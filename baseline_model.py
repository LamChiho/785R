"""
REINFORCE with Learned Baseline + PSL Regularization
====================================================

Differences from RLOO:
1. Adds a separate value network V(s) to predict expected rewards
2. Baseline is learned (not empirical batch mean)
3. Works with batch_size=1 (no need for K>1)
4. Requires training both policy and value network
"""

import torch
from torch import nn
from transformers import GPT2Model
from logic_table import construct_table
from psl_loss import NaturalLogicPSLLoss
from fix2 import fix_nnl_path

label2rel = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6], 3: [2]}
rel2label = {0: 0, 1: 0, 3: 1, 4: 1, 2: 2, 5: 2, 6: 2}

class ValueNetwork(nn.Module):
    """
    Value network V(s) for baseline estimation.
    
    Predicts expected cumulative reward from current state.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Scalar output: V(s)
        )
    
    def forward(self, state_representation):
        """
        Args:
            state_representation: [batch, hidden_dim] - State embedding
        
        Returns:
            value: [batch, 1] - Predicted value V(s)
        """
        return self.value_head(state_representation)


class BaselineGPT2Classifier(nn.Module):
    """
    GPT2-based Natural Logic Classifier with Learned Baseline + PSL.
    
    Architecture:
    1. GPT2 encoder (fine-tuned)
    2. Local relation prediction layer (7 relations)
    3. VALUE NETWORK (NEW) - predicts V(s)
    4. PSL regularization on distributions
    5. REINFORCE with learned baseline
    6. Natural logic program execution
    7. Introspective revision
    """

    def __init__(self, n_class, lambda_psl=0.5, use_psl=True, use_baseline=True):
        super(BaselineGPT2Classifier, self).__init__()
        
        # Original NS-NLI components
        self.model = GPT2Model.from_pretrained('gpt2')
        self.local_relation_layer = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(2*768, 768, bias=True), 
            nn.ReLU(), 
            nn.Linear(768, 7)
        )
        self.classification_layer = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(2*768, n_class, bias=True)
        )
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.logic = construct_table(map=True).cuda()
        
        # Mask to remove negation and cover relations
        self.rm = torch.tensor([0, 0, 0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        
        # NEW: Value network for baseline
        self.use_baseline = use_baseline
        if use_baseline:
            self.value_network = ValueNetwork(hidden_dim=768).cuda()
        
        # PSL loss module
        self.use_psl = use_psl
        self.lambda_psl = lambda_psl
        if use_psl:
            self.psl_loss_fn = NaturalLogicPSLLoss(
                lambda_transitivity=0.7,
                lambda_symmetry=0.3,
                device='cuda'
            )
        
    def forward(self, input_idxs, masks, segment_idxs, projections, hypothesis_len, 
                phrase_start, phrase_end, ys=None, hints=None, train=False):
        """
        Forward pass with learned baseline and PSL.
        
        Key Modifications:
        1. Compute state representation for value network
        2. Predict baseline V(s) from value network
        3. Compute advantages: A = R - V(s)
        4. Update both policy and value network
        """
        
        def get_vectors(tensor, index):
            ind1 = torch.arange(index.shape[0], dtype=torch.int64).unsqueeze(-1).repeat([1, index.shape[1]]).view(-1)
            ind2 = index.view(-1)
            return tensor[ind1, ind2].view(index.shape[0], index.shape[1], -1)
        
        batch_size = ys.shape[0]
        max_agg = torch.max(hypothesis_len).item()
        
        # 1. GPT2 Encoding
        outputs = self.model(input_idxs, token_type_ids=segment_idxs, attention_mask=masks)
        hidden_states = outputs[0]  # [batch, seq_len, 768]
        
        # NEW: Extract state representation for value network
        # Use [CLS] token (first token) as state representation
        state_representation = hidden_states[:, 0, :]  # [batch, 768]
        
        # NEW: Predict baseline V(s)
        baseline_values = None
        if self.use_baseline and train:
            baseline_values = self.value_network(state_representation).squeeze(-1)  # [batch]
        
        # 2. Extract phrase representations
        phrase_start_rep = get_vectors(hidden_states, phrase_start)
        phrase_end_rep = get_vectors(hidden_states, phrase_end)
        phrase_rep = torch.cat([phrase_start_rep, phrase_end_rep], dim=-1)
        
        # 3. Compute relation distributions
        local_relations = torch.softmax(
            self.local_relation_layer(phrase_rep) - self.rm * 1e06, 
            dim=-1
        )  # [batch, max_phrases, 7]
        
        # 4. PSL LOSS (on distributions, BEFORE sampling)
        psl_loss = torch.tensor(0.0).cuda()
        psl_metrics = {}
        
        if self.use_psl and train:
            psl_loss, psl_metrics = self.psl_loss_fn(local_relations, hypothesis_len)
        
        # 5. Sample relations for REINFORCE
        prev_relation = torch.zeros([batch_size], dtype=torch.int64).cuda()
        relation_action = prev_relation.unsqueeze(-1).cuda()
        relation_history = prev_relation.unsqueeze(-1).cuda()
        probs = 0.5 * torch.ones([batch_size, 1]).cuda()
        used_projections = torch.zeros([batch_size, 1, 7], dtype=torch.int64).cuda()
        
        for i in range(max_agg):
            transition = local_relations[:, i]
            
            if train:
                sampled_relation = torch.multinomial(transition.detach(), num_samples=1).squeeze(-1)
            else:
                sampled_relation = torch.argmax(transition, dim=-1)
            
            relation_action = torch.cat([relation_action, sampled_relation.unsqueeze(-1)], dim=-1)
            sampled_prob = transition[range(batch_size), sampled_relation]
            probs = torch.cat([probs, sampled_prob.unsqueeze(-1)], dim=-1)
            
            proj = projections[range(batch_size), phrase_start[:, i]]
            used_projections = torch.cat([used_projections, proj.unsqueeze(1)], dim=1)
            agg_result = self.logic[prev_relation, self.project(sampled_relation, proj)]
            relation_history = torch.cat([relation_history, agg_result.unsqueeze(-1)], dim=-1)
            prev_relation = agg_result
        
        last_state = relation_history[range(batch_size), hypothesis_len]
        
        # 6. Introspective Revision
        reward_raw = self.shape_reward(relation_history, ys, hypothesis_len, 0.5)
        policy_gradient_loss_raw = self.loss(probs, reward_raw, hypothesis_len, baseline_values)
        
        if train:
            fixed_actions, relation_history, last_state = self.back_search(
                relation_action, relation_history, local_relations, 
                used_projections, ys, hypothesis_len, hints
            )
            probs = 0.5 * torch.ones([batch_size, 1]).cuda()
            for i in range(max_agg):
                transition = local_relations[:, i]
                sampled_prob = transition[range(batch_size), fixed_actions[:, i]]
                probs = torch.cat([probs, sampled_prob.unsqueeze(-1)], dim=-1)
        
        # 7. Compute final reward and REINFORCE loss
        reward = self.shape_reward(relation_history, ys, hypothesis_len, gamma=1.0)
        policy_gradient_loss = self.loss(probs, reward, hypothesis_len, baseline_values)
        
        # 8. COMBINED LOSS (REINFORCE + PSL)
        policy_gradient_loss = 0.5 * policy_gradient_loss_raw + 0.5 * policy_gradient_loss
        total_loss = policy_gradient_loss + self.lambda_psl * psl_loss
        
        # NEW: Value network loss (MSE between V(s) and actual returns)
        value_loss = torch.tensor(0.0).cuda()
        if self.use_baseline and train:
            # Compute actual returns (sum of rewards)
            actual_returns = torch.zeros(batch_size, device=reward.device)
            for i in range(batch_size):
                actual_returns[i] = reward[i, 1:hypothesis_len[i]+1].sum()
            
            # MSE loss: (V(s) - R)^2
            value_loss = ((baseline_values - actual_returns) ** 2).mean()
        
        # 9. Compute logits
        logit_entail = (last_state == 0) + (last_state == 1)
        logit_contradiction = (last_state == 3) + (last_state == 4)
        logit_neutral = (last_state == 2) + (last_state == 5) + (last_state == 6)
        logits = torch.cat([
            logit_entail.unsqueeze(-1), 
            logit_contradiction.unsqueeze(-1), 
            logit_neutral.unsqueeze(-1)
        ], dim=-1).type(torch.int64)
        
        metrics = {
            'psl_loss': psl_loss.item(),
            'reinforce_loss': policy_gradient_loss.mean().item(),
            'value_loss': value_loss.item(),
            **psl_metrics
        }
        
        return logits, total_loss, value_loss, (relation_history, relation_action), metrics
    
    def project(self, relation, projection):
        return projection[range(relation.shape[0]), relation]
    
    def back_search(self, relation_action, relation_history, local_relations, projections, y, hypo_len, hints):
        """Introspective revision (same as original)."""
        fixed_actions = torch.zeros(relation_action.shape, dtype=torch.int64).cuda()
        fixed_history = torch.zeros(relation_action.shape, dtype=torch.int64).cuda()
        last_states = torch.zeros(relation_action.shape[0], dtype=torch.int64).cuda()
        
        for i in range(relation_history.shape[0]):
            f_rel, f_hist, lst_state = fix_nnl_path(
                hints[i], relation_action[i], local_relations[i], 
                relation_history[i], projections[i], y[i], hypo_len[i]
            )
            fixed_actions[i, 1:hypo_len[i]+1] = torch.tensor(f_rel, dtype=torch.int64).cuda()
            fixed_history[i, 0:hypo_len[i]+1] = torch.tensor(f_hist, dtype=torch.int64).cuda()
            last_states[i] = lst_state
        
        return fixed_actions[:, 1:], fixed_history, last_states
    
    def shape_reward(self, relation_history, ys, hypo_len, gamma=0.5):
        """Reward shaping (same as original)."""
        def compute_reward(history, y, hp_len):
            rwd = torch.zeros([history.shape[0]])
            label_rel = label2rel[y.item()]
            hit_obj = history[hp_len].item() in label_rel
            final_rwd = 2 * int(hit_obj) - 1
            gamma_ = 1 if hit_obj else gamma
            
            for i in range(1, hp_len + 1):
                rwd[i] = final_rwd * pow(gamma_, hp_len - i)
                if history[i] in [2, 5, 6] and not hit_obj:
                    rwd[i] = -1
                    break
                if hit_obj and history[hp_len] == 0:
                    rwd[i] = min(0, rwd[i])
                if history[i] in label_rel:
                    rwd[i] = max(0, rwd[i])
            
            return rwd
        
        reward = torch.zeros(relation_history.shape, dtype=torch.float32).cuda()
        for i in range(relation_history.shape[0]):
            reward[i, :] = compute_reward(relation_history[i], ys[i], hypo_len[i])
        
        return reward
    
    def loss(self, probs, reward, hypo_len, baseline_values=None):
        """
        REINFORCE loss with LEARNED baseline.
        
        Args:
            probs: [batch, max_len+1] - Sampled probabilities
            reward: [batch, max_len+1] - Shaped rewards
            hypo_len: [batch] - Hypothesis lengths
            baseline_values: [batch] - Predicted baselines V(s) (optional)
        
        Returns:
            loss: [batch] - Per-sample losses
        """
        # Compute advantages
        if baseline_values is not None:
            # Advantage: A = R - V(s)
            # Expand baseline to match reward shape
            baseline_expanded = baseline_values.unsqueeze(1).expand_as(reward)
            advantages = reward - baseline_expanded
        else:
            advantages = reward
        
        # REINFORCE loss with advantages
        mask = torch.ones_like(probs).cuda()
        for i in range(hypo_len.shape[0]):
            mask[i, hypo_len[i]+1:] = 0
        
        loss_ = (
            - torch.log(probs) * advantages * (advantages > 0).type(torch.float32) * mask 
            - torch.log(1 - probs + 0.001) * torch.abs(advantages) * (advantages < 0).type(torch.float32) * mask
        )
        
        return torch.sum(loss_, dim=-1)