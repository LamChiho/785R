"""
Enhanced GPT2 Classifier with PSL Regularization and RLOO Support
==================================================================

Modifications from original sampling_model_fix3.py:
1. Added PSL loss computation on probability distributions (BEFORE sampling)
2. Added RLOO baseline support for variance reduction
3. Maintained full backward compatibility with introspective revision
4. Added detailed logging for PSL metrics

Key Changes:
- forward() now computes PSL loss on relation distributions
- loss() function modified to use RLOO advantages
- New method: compute_rloo_baseline()
"""

import torch
from torch import nn
from transformers import GPT2Model
from logic_table import construct_table
from psl_loss import NaturalLogicPSLLoss
from fix2 import fix_nnl_path

label2rel = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6], 3: [2]}
rel2label = {0: 0, 1: 0, 3: 1, 4: 1, 2: 2, 5: 2, 6: 2}

class EnhancedGPT2Classifier(nn.Module):
    """
    Enhanced GPT2-based Natural Logic Classifier with PSL and RLOO.
    
    Architecture:
    1. GPT2 encoder (frozen or fine-tuned)
    2. Local relation prediction layer (7 relations)
    3. PSL regularization on distributions
    4. REINFORCE with RLOO baseline
    5. Natural logic program execution
    6. Introspective revision (optional)
    """

    def __init__(self, n_class, lambda_psl=0.5, use_psl=True, use_rloo=True):
        super(EnhancedGPT2Classifier, self).__init__()
        
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
        
        # Mask to remove negation and cover relations (as in original code)
        self.rm = torch.tensor([0, 0, 0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        
        # NEW: PSL loss module
        self.use_psl = use_psl
        self.lambda_psl = lambda_psl
        if use_psl:
            self.psl_loss_fn = NaturalLogicPSLLoss(
                lambda_transitivity=0.7,
                lambda_symmetry=0.3,
                device='cuda'
            )
        
        # NEW: RLOO flag
        self.use_rloo = use_rloo
        
    def forward(self, input_idxs, masks, segment_idxs, projections, hypothesis_len, 
                phrase_start, phrase_end, ys=None, hints=None, train=False, batch_rewards=None):
        """
        Forward pass with PSL and RLOO integration.
        
        Key Modifications:
        1. Compute PSL loss on distributions BEFORE sampling
        2. Use .detach() on distributions for sampling (stop gradient)
        3. Compute RLOO baseline if batch_rewards provided
        4. Combine REINFORCE + PSL losses
        
        Args:
            input_idxs: [batch, seq_len] - Token IDs
            masks: [batch, seq_len] - Attention masks
            segment_idxs: [batch, seq_len] - Segment IDs (premise=0, hypothesis=1)
            projections: [batch, seq_len, 7] - Monotonicity projections
            hypothesis_len: [batch] - Hypothesis lengths
            phrase_start: [batch, max_phrases] - Phrase start positions
            phrase_end: [batch, max_phrases] - Phrase end positions
            ys: [batch] - Ground truth labels
            hints: List[Dict] - Introspective revision hints
            train: bool - Training mode flag
            batch_rewards: [batch] - Rewards from previous batch samples (for RLOO)
        
        Returns:
            logits: [batch, 3] - Predicted labels
            total_loss: Scalar - Combined REINFORCE + PSL loss
            proof: Tuple - (relation_history, sampled_actions)
            metrics: Dict - PSL and RLOO metrics
        """
        
        def get_vectors(tensor, index):
            """Extract phrase representations from hidden states."""
            ind1 = torch.arange(index.shape[0], dtype=torch.int64).unsqueeze(-1).repeat([1, index.shape[1]]).view(-1)
            ind2 = index.view(-1)
            return tensor[ind1, ind2].view(index.shape[0], index.shape[1], -1)
        
        batch_size = ys.shape[0]
        max_agg = torch.max(hypothesis_len).item()
        
        # 1. GPT2 Encoding (original)
        outputs = self.model(input_idxs, token_type_ids=segment_idxs, attention_mask=masks)
        hidden_states = outputs[0]  # [batch, seq_len, 768]
        
        # 2. Extract phrase representations (original)
        phrase_start_rep = get_vectors(hidden_states, phrase_start)
        phrase_end_rep = get_vectors(hidden_states, phrase_end)
        phrase_rep = torch.cat([phrase_start_rep, phrase_end_rep], dim=-1)  # [batch, max_phrases, 1536]
        
        # 3. Compute relation distributions (original)
        local_relations = torch.softmax(
            self.local_relation_layer(phrase_rep) - self.rm * 1e06, 
            dim=-1
        )  # [batch, max_phrases, 7]
        
        # ---------------------------------------------------------------
        # NEW: PSL LOSS COMPUTATION (on distributions, BEFORE sampling)
        # ---------------------------------------------------------------
        psl_loss = torch.tensor(0.0).cuda()
        psl_metrics = {}
        
        if self.use_psl and train:
            psl_loss, psl_metrics = self.psl_loss_fn(local_relations, hypothesis_len)
        
        # 4. Sample relations for REINFORCE (original, with .detach() for PSL)
        # CRITICAL: Use .detach() so PSL gradients don't flow through sampling
        prev_relation = torch.zeros([batch_size], dtype=torch.int64).cuda()
        relation_action = prev_relation.unsqueeze(-1).cuda()
        relation_history = prev_relation.unsqueeze(-1).cuda()
        probs = 0.5 * torch.ones([batch_size, 1]).cuda()
        used_projections = torch.zeros([batch_size, 1, 7], dtype=torch.int64).cuda()
        
        for i in range(max_agg):
            transition = local_relations[:, i]  # [batch, 7]
            
            if train:
                # Sample from DETACHED distribution (stop gradient for PSL)
                sampled_relation = torch.multinomial(transition.detach(), num_samples=1).squeeze(-1)
            else:
                sampled_relation = torch.argmax(transition, dim=-1)
            
            relation_action = torch.cat([relation_action, sampled_relation.unsqueeze(-1)], dim=-1)
            sampled_prob = transition[range(batch_size), sampled_relation]
            probs = torch.cat([probs, sampled_prob.unsqueeze(-1)], dim=-1)
            
            # Apply projection and compose
            proj = projections[range(batch_size), phrase_start[:, i]]
            used_projections = torch.cat([used_projections, proj.unsqueeze(1)], dim=1)
            agg_result = self.logic[prev_relation, self.project(sampled_relation, proj)]
            relation_history = torch.cat([relation_history, agg_result.unsqueeze(-1)], dim=-1)
            prev_relation = agg_result
        
        last_state = relation_history[range(batch_size), hypothesis_len]
        
        # 5. Introspective Revision (original)
        relation_history_backup = relation_history.clone().detach()
        reward_raw = self.shape_reward(relation_history, ys, hypothesis_len, 0.5)
        policy_gradient_loss_raw = self.loss(probs, reward_raw, hypothesis_len, relation_history, ys, local_relations, batch_rewards)
        
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
        
        # 6. Compute final reward and REINFORCE loss
        reward = self.shape_reward(relation_history, ys, hypothesis_len, gamma=1.0, fix=False)
        policy_gradient_loss = self.loss(probs, reward, hypothesis_len, relation_history, ys, local_relations, batch_rewards)
        
        # ---------------------------------------------------------------
        # NEW: COMBINED LOSS (REINFORCE + PSL)
        # ---------------------------------------------------------------
        policy_gradient_loss = 0.5 * policy_gradient_loss_raw + 0.5 * policy_gradient_loss
        total_loss = policy_gradient_loss + self.lambda_psl * psl_loss
        
        # 7. Compute logits (original)
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
            'reinforce_loss': policy_gradient_loss.mean().item(),  # ? FIXED
            **psl_metrics
        }
        
        return logits, total_loss, (relation_history, relation_action), metrics
    
    def project(self, relation, projection):
        """Apply monotonicity projection (original)."""
        return projection[range(relation.shape[0]), relation]
    
    def back_search(self, relation_action, relation_history, local_relations, projections, y, hypo_len, hints):
        """Introspective revision (original, from fix2.py)."""
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
    
    def shape_reward(self, relation_history, ys, hypo_len, gamma=0.5, fix=False):
        """Reward shaping (original)."""
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
    
    def loss(self, probs, reward, hypo_len, relation_history, ys, local_relations, batch_rewards=None):
        """
        REINFORCE loss with optional RLOO baseline.
        
        MODIFICATION: If batch_rewards is provided, compute RLOO baseline:
        baseline_i = mean(batch_rewards[j != i])
        
        Args:
            probs: [batch, max_len+1] - Sampled probabilities
            reward: [batch, max_len+1] - Shaped rewards
            hypo_len: [batch] - Hypothesis lengths
            batch_rewards: [batch] - Final rewards for RLOO (optional)
        
        Returns:
            loss: [batch] - Per-sample losses
        """
        # ---------------------------------------------------------------
        # NEW: RLOO BASELINE COMPUTATION
        # ---------------------------------------------------------------
        if self.use_rloo and batch_rewards is not None:
            # Compute leave-one-out baseline for each sample
            batch_size = reward.shape[0]
            baselines = torch.zeros(batch_size, device=reward.device)
            
            for i in range(batch_size):
                # Mean of all rewards except current sample
                mask = torch.ones(batch_size, dtype=torch.bool, device=reward.device)
                mask[i] = False
                baselines[i] = batch_rewards[mask].mean()
            
            # Subtract baseline from reward (advantage)
            # Expand baseline to match reward shape [batch, max_len+1]
            baseline_expanded = baselines.unsqueeze(1).expand_as(reward)
            reward = reward - baseline_expanded
        
        # Original REINFORCE loss computation
        mask = torch.ones_like(probs).cuda()
        for i in range(hypo_len.shape[0]):
            mask[i, hypo_len[i]+1:] = 0
        
        loss_ = (
            - torch.log(probs) * reward * (reward > 0).type(torch.float32) * mask 
            - torch.log(1 - probs + 0.001) * torch.abs(reward) * (reward < 0).type(torch.float32) * mask
        )
        
        return torch.sum(loss_, dim=-1)