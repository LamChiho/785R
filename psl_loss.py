"""
PSL (Probabilistic Soft Logic) Regularization for Natural Logic Reasoning
==========================================================================

This module implements PSL soft constraints to enforce composition consistency
in natural logic reasoning paths. The PSL loss operates on PROBABILITY DISTRIBUTIONS
(not sampled actions) to maintain differentiability.

Key Concepts:
- Lukasiewicz t-norm: P(A AND B) = max(0, P(A) + P(B) - 1)
- PSL distance: d = max(0, body_satisfaction - head_satisfaction)
- Composition consistency: P(r_t) AND P(r_{t+1}) IMPLIES P(compose(r_t, r_{t+1}))

Integration with NS-NLI:
- Called BEFORE sampling in forward pass
- Uses 7-relation composition table from logic_table.py
- Compatible with introspective revision (operates independently)
"""

import torch
import torch.nn as nn
from logic_table import construct_table

class NaturalLogicPSLLoss(nn.Module):
    """
    PSL regularization for natural logic reasoning paths.
    Enforces composition consistency: compose(r_t, r_{t+1}) should match predicted r_{t+2}.
    """
    
    def __init__(self, lambda_transitivity=0.7, lambda_symmetry=0.3, device='cuda'):
        super().__init__()
        self.lambda_trans = lambda_transitivity
        self.lambda_sym = lambda_symmetry
        self.device = device
        
        # Load 7-relation composition table from logic_table.py
        # Relations: eq=0, ent=1, rent=2, neg=3, alt=4, cov=5, ind=6
        self.composition_table = construct_table(map=True).to(device)  # [7, 7] -> result
        
    def compute_composed_distribution(self, p_r1, p_r2):
        """
        Compute expected probability distribution of compose(r_t, r_{t+1})
        given distributions P(r_t) and P(r_{t+1}).
        
        Args:
            p_r1: [batch, 7] - P(r_t)
            p_r2: [batch, 7] - P(r_{t+1})
        
        Returns:
            p_composed: [batch, 7] - P(compose(r_t, r_{t+1}))

        """
        batch_size = p_r1.shape[0]
        p_composed = torch.zeros(batch_size, 7, device=self.device)
        
        # Iterate over all relation pairs
        for i in range(7):  # r_t
            for j in range(7):  # r_{t+1}
                # Lookup composition result: compose(i, j) = ?
                result_rel = self.composition_table[i, j].item()
                
                # Probability mass from this pair
                # P(r_t=i AND r_{t+1}=j) contributes to P(result_rel)
                p_composed[:, result_rel] += p_r1[:, i] * p_r2[:, j]
        
        return p_composed
    
    def transitivity_loss(self, distributions, hypo_len):
        """
        Enforce transitivity/composition consistency:
        P(r_t) AND P(r_{t+1}) IMPLIES P(compose(r_t, r_{t+1})) should match P(r_{t+2})
        
        Uses Lukasiewicz t-norm for conjunction:
        P(A AND B) = max(0, P(A) + P(B) - 1)
        
        PSL distance (how much rule is violated):
        d = max(0, P(A AND B) - P(C))  where rule is A AND B IMPLIES C
        
        Args:
            distributions: [batch, max_len, 7] - Policy distributions at each step
            hypo_len: [batch] - Actual hypothesis lengths (for masking)
        
        Returns:
            loss: Scalar tensor (mean PSL distance over all valid triplets)
        """
        batch_size, max_len, num_relations = distributions.shape
        total_loss = 0.0
        count = 0
        
        # Check each consecutive triplet (r_t, r_{t+1}, r_{t+2})
        for t in range(max_len - 2):
            p_t = distributions[:, t, :]    # P(r_t)   [batch, 7]
            p_t1 = distributions[:, t+1, :] # P(r_{t+1}) [batch, 7]
            p_t2 = distributions[:, t+2, :] # P(r_{t+2}) [batch, 7]
            
            # Compute expected distribution of compose(r_t, r_{t+1})
            p_expected_composed = self.compute_composed_distribution(p_t, p_t1)
            
            # PSL distance: How much does p_t2 deviate from p_expected_composed?
            # We want p_expected_composed approximately equal to p_t2
            # Distance = sum of violations across all relations
            distance = torch.sum(torch.clamp(p_expected_composed - p_t2, min=0), dim=1)
            
            # Mask out samples beyond hypothesis length
            mask = (t + 2 < hypo_len).float()  # [batch]
            masked_distance = distance * mask
            
            total_loss += masked_distance.sum()
            count += mask.sum()
        
        # Average over all valid triplets
        return total_loss / (count + 1e-8)
    
    def symmetry_loss(self, distributions):
        """
        Enforce symmetry rules:
        - Forward(A,B) IMPLIES Reverse(B,A)
        - Equiv(A,B) IMPLIES Equiv(B,A)
        
        NOTE: This requires bidirectional relation tracking (premise-to-hypothesis)
        For now, this is a placeholder. Can be extended if phrase pairs are tracked.
        """
        # Placeholder - implement if needed
        return torch.tensor(0.0, device=self.device)
    
    def forward(self, distributions, hypo_len):
        """
        Compute total PSL loss.
        
        Args:
            distributions: [batch, max_len, 7] - Softmax distributions from policy network
            hypo_len: [batch] - Actual hypothesis lengths
        
        Returns:
            psl_loss: Scalar tensor
            metrics: Dict with loss components
        """
        trans_loss = self.transitivity_loss(distributions, hypo_len)
        sym_loss = self.symmetry_loss(distributions)
        
        total_loss = self.lambda_trans * trans_loss + self.lambda_sym * sym_loss
        
        return total_loss, {
            'transitivity_loss': trans_loss.item(),
            'symmetry_loss': sym_loss.item()
        }