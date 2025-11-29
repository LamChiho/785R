"""
Enhanced Model WITHOUT Introspective Revision
==============================================

Key Modification:
- Removed back_search() call in forward pass
- All other components (PSL, RLOO) remain intact
"""

import torch
from torch import nn
from transformers import GPT2Model
from logic_table import construct_table
from psl_loss import NaturalLogicPSLLoss

label2rel = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6], 3: [2]}
rel2label = {0: 0, 1: 0, 3: 1, 4: 1, 2: 2, 5: 2, 6: 2}

class EnhancedGPT2ClassifierNoIR(nn.Module):
    """Enhanced model WITHOUT introspective revision."""

    def __init__(self, n_class, lambda_psl=0.5, use_psl=True, use_rloo=True):
        super(EnhancedGPT2ClassifierNoIR, self).__init__()
        
        self.model = GPT2Model.from_pretrained('gpt2')
        self.local_relation_layer = nn.Sequential(
            nn. Dropout(0.1), 
            nn.Linear(2*768, 768, bias=True), 
            nn.ReLU(), 
            nn.Linear(768, 7)
        )
        self.classification_layer = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(2*768, n_class, bias=True)
        )
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.logic = construct_table(map=True). cuda()
        self.rm = torch.tensor([0, 0, 0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        
        self.use_psl = use_psl
        self.lambda_psl = lambda_psl
        if use_psl:
            self.psl_loss_fn = NaturalLogicPSLLoss(
                lambda_transitivity=0.7,
                lambda_symmetry=0.3,
                device='cuda'
            )
        
        self.use_rloo = use_rloo
        
    def forward(self, input_idxs, masks, segment_idxs, projections, hypothesis_len, 
                phrase_start, phrase_end, ys=None, hints=None, train=False, batch_rewards=None):
        
        def get_vectors(tensor, index):
            ind1 = torch.arange(index.shape[0], dtype=torch.int64).unsqueeze(-1).repeat([1, index.shape[1]]).view(-1)
            ind2 = index.view(-1)
            return tensor[ind1, ind2]. view(index.shape[0], index. shape[1], -1)
        
        batch_size = ys.shape[0]
        max_agg = torch.max(hypothesis_len). item()
        
        # GPT2 encoding
        outputs = self.model(input_idxs, token_type_ids=segment_idxs, attention_mask=masks)
        hidden_states = outputs[0]
        
        # Extract phrase representations
        phrase_start_rep = get_vectors(hidden_states, phrase_start)
        phrase_end_rep = get_vectors(hidden_states, phrase_end)
        phrase_rep = torch.cat([phrase_start_rep, phrase_end_rep], dim=-1)
        
        # Compute relation distributions
        local_relations = torch.softmax(
            self.local_relation_layer(phrase_rep) - self.rm * 1e06, 
            dim=-1
        )
        
        # PSL loss (on distributions, before sampling)
        psl_loss = torch.tensor(0.0).cuda()
        psl_metrics = {}
        if self.use_psl and train:
            psl_loss, psl_metrics = self.psl_loss_fn(local_relations, hypothesis_len)
        
        # Sample relations
        prev_relation = torch.zeros([batch_size], dtype=torch.int64).cuda()
        relation_action = prev_relation.unsqueeze(-1).cuda()
        relation_history = prev_relation.unsqueeze(-1).cuda()
        probs = 0.5 * torch.ones([batch_size, 1]).cuda()
        used_projections = torch.zeros([batch_size, 1, 7], dtype=torch.int64).cuda()
        
        for i in range(max_agg):
            transition = local_relations[:, i]
            
            if train:
                sampled_relation = torch.multinomial(transition. detach(), num_samples=1). squeeze(-1)
            else:
                sampled_relation = torch.argmax(transition, dim=-1)
            
            relation_action = torch.cat([relation_action, sampled_relation. unsqueeze(-1)], dim=-1)
            sampled_prob = transition[range(batch_size), sampled_relation]
            probs = torch.cat([probs, sampled_prob. unsqueeze(-1)], dim=-1)
            
            proj = projections[range(batch_size), phrase_start[:, i]]
            used_projections = torch.cat([used_projections, proj.unsqueeze(1)], dim=1)
            agg_result = self.logic[prev_relation, self.project(sampled_relation, proj)]
            relation_history = torch.cat([relation_history, agg_result. unsqueeze(-1)], dim=-1)
            prev_relation = agg_result
        
        last_state = relation_history[range(batch_size), hypothesis_len]
        
        # ================================================================
        # KEY MODIFICATION: NO INTROSPECTIVE REVISION (no back_search)
        # ================================================================
        # Compute reward directly from sampled path (no fixing)
        reward = self.shape_reward(relation_history, ys, hypothesis_len, gamma=1.0)
        policy_gradient_loss = self.loss(probs, reward, hypothesis_len, relation_history, ys, local_relations, batch_rewards)
        
        # Combined loss
        total_loss = policy_gradient_loss + self.lambda_psl * psl_loss
        
        # Compute logits
        logit_entail = (last_state == 0) + (last_state == 1)
        logit_contradiction = (last_state == 3) + (last_state == 4)
        logit_neutral = (last_state == 2) + (last_state == 5) + (last_state == 6)
        logits = torch.cat([
            logit_entail.unsqueeze(-1), 
            logit_contradiction. unsqueeze(-1), 
            logit_neutral.unsqueeze(-1)
        ], dim=-1). type(torch.int64)
        
        metrics = {
            'psl_loss': psl_loss.item(),
            'reinforce_loss': policy_gradient_loss.mean().item(),
            **psl_metrics
        }
        
        return logits, total_loss, (relation_history, relation_action), metrics
    
    def project(self, relation, projection):
        return projection[range(relation.shape[0]), relation]
    
    def shape_reward(self, relation_history, ys, hypo_len, gamma=0.5):
        def compute_reward(history, y, hp_len):
            rwd = torch.zeros([history.shape[0]])
            label_rel = label2rel[y. item()]
            hit_obj = history[hp_len]. item() in label_rel
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
        # RLOO baseline
        if self.use_rloo and batch_rewards is not None:
            batch_size = reward.shape[0]
            baselines = torch.zeros(batch_size, device=reward.device)
            
            for i in range(batch_size):
                mask = torch.ones(batch_size, dtype=torch.bool, device=reward.device)
                mask[i] = False
                baselines[i] = batch_rewards[mask].mean()
            
            baseline_expanded = baselines.unsqueeze(1).expand_as(reward)
            reward = reward - baseline_expanded
        
        # REINFORCE loss
        mask = torch.ones_like(probs). cuda()
        for i in range(hypo_len.shape[0]):
            mask[i, hypo_len[i]+1:] = 0
        
        loss_ = (
            - torch.log(probs) * reward * (reward > 0).type(torch.float32) * mask 
            - torch.log(1 - probs + 0.001) * torch.abs(reward) * (reward < 0).type(torch.float32) * mask
        )
        
        return torch.sum(loss_, dim=-1)