

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactHybridLoss(nn.Module):

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0, 
                 nuc_pref_weight: float = 1.0, contact_threshold: float = 0.1, 
                 pos_weight: float = 5.0, topk: int = 10):
        
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.nuc_pref_weight = nuc_pref_weight
        self.contact_threshold = contact_threshold
        self.topk = topk
        
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_cls = predictions['contact_cls']
        pred_nuc_pref = predictions['nuc_preference']
        pred_importance = predictions['binding_importance']
        contact_values = batch['contact_tensor']
        
        mask = self._create_sequence_mask(
            batch['protein_lengths'],
            pred_cls.size(1)
        ).unsqueeze(-1)
        mask = mask.expand(-1, -1, 4)
        
        target_cls = (contact_values > 0).float()
        
        valid_pred_cls = pred_cls[mask.bool()]
        valid_target_cls = target_cls[mask.bool()]
        cls_loss = F.binary_cross_entropy_with_logits(
            valid_pred_cls,
            valid_target_cls,
            pos_weight=self.pos_weight
        )
        
        nuc_pref_loss = self._calculate_nucleotide_preference_loss(
            pred_nuc_pref, contact_values, mask
        )
        
        rank_loss = self._lambda_ndcg_topk_loss(
            pred_importance, contact_values, mask[:, :, 0]
        )
        
        total_loss = (self.cls_weight * cls_loss + 
                     self.nuc_pref_weight * nuc_pref_loss + 
                     self.reg_weight * rank_loss)
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'nuc_pref_loss': nuc_pref_loss,
            'rank_loss': rank_loss
        }
    
    def _calculate_nucleotide_preference_loss(self, 
                                           pred_nuc_pref: torch.Tensor,
                                           contact_values: torch.Tensor,
                                           mask: torch.Tensor) -> torch.Tensor:
        
        true_max_nuc = contact_values.argmax(dim=-1)
        
        aa_mask = mask[:, :, 0] & (contact_values.max(dim=-1)[0] > 0)
        
        if not aa_mask.any():
            return torch.tensor(0.0, device=pred_nuc_pref.device)
        
        valid_pred = pred_nuc_pref[aa_mask]
        valid_true_max = true_max_nuc[aa_mask]
        
        return F.cross_entropy(valid_pred, valid_true_max)

    def _ranknet_topk_loss(self,
                           pred_importance: torch.Tensor,
                           contact_values: torch.Tensor,
                           seq_mask: torch.Tensor) -> torch.Tensor:
        
        true_importance = contact_values.max(dim=-1)[0]

        valid_mask = seq_mask & (true_importance > 0)

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_importance.device)

        total_loss = pred_importance.new_tensor(0.0)
        total_pairs = pred_importance.new_tensor(0.0)

        for b in range(pred_importance.size(0)):
            valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            true_scores_b = true_importance[b, valid_idx]
            k = min(self.topk, true_scores_b.numel())
            topk_vals, topk_order = torch.topk(true_scores_b, k=k, largest=True, sorted=True)
            topk_idx = valid_idx[topk_order]

            s = pred_importance[b, topk_idx]
            y = true_importance[b, topk_idx]

            if s.numel() <= 1:
                continue
            diff_pred = s[:, None] - s[None, :]
            diff_true = y[:, None] - y[None, :]
            positive_pairs = diff_true > 0
            if not positive_pairs.any():
                continue
            pair_losses = F.softplus(-diff_pred)
            loss_mat = pair_losses * positive_pairs.float()
            total_loss = total_loss + loss_mat.sum()
            total_pairs = total_pairs + positive_pairs.float().sum()

        if total_pairs.item() == 0:
            return torch.tensor(0.0, device=pred_importance.device)
        return total_loss / total_pairs
    
    def _lambda_ndcg_topk_loss(self,
                               pred_importance: torch.Tensor,
                               contact_values: torch.Tensor,
                               seq_mask: torch.Tensor) -> torch.Tensor:
        
        true_importance = contact_values.max(dim=-1)[0]

        valid_mask = seq_mask.bool()

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_importance.device)

        total_loss = pred_importance.new_tensor(0.0)
        total_weight = pred_importance.new_tensor(0.0)

        B = pred_importance.size(0)
        for b in range(B):
            valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            s_valid = pred_importance[b, valid_idx]
            y_valid = true_importance[b, valid_idx]

            T = int(min(self.topk, s_valid.numel()))
            if T <= 0:
                continue

            pred_top_rel = torch.topk(s_valid, k=T, largest=True, sorted=False).indices
            true_top_rel = torch.topk(y_valid, k=T, largest=True, sorted=False).indices
            cand_rel = torch.unique(torch.cat([pred_top_rel, true_top_rel], dim=0), sorted=True)

            s = s_valid[cand_rel]
            y = y_valid[cand_rel]

            M = int(s.numel())
            if M <= 1:
                continue

            k_focus = int(min(5, y_valid.numel()))
            if k_focus <= 0:
                continue

            ideal_gains = torch.sort(y_valid, descending=True).values[:k_focus]
            discounts_k = 1.0 / torch.log2(torch.arange(2, 2 + k_focus, device=y.device).float())
            idcg_at_k = (ideal_gains * discounts_k).sum()
            if idcg_at_k.item() <= 0:
                continue

            order = torch.argsort(s, descending=True)
            ranks = torch.empty(M, dtype=torch.long, device=s.device)
            ranks[order] = torch.arange(1, M + 1, device=s.device, dtype=torch.long)

            ranks_float = ranks.to(dtype=s.dtype)
            discounts_all = 1.0 / torch.log2(ranks_float + 1.0)

            diff_pred = s[:, None] - s[None, :]
            diff_true = y[:, None] - y[None, :]
            positive_pairs = diff_true > 0
            if not positive_pairs.any():
                continue

            topk_mask_elem = ranks <= k_focus
            topk_pair_mask = topk_mask_elem[:, None] | topk_mask_elem[None, :]

            delta_discount = torch.abs(discounts_all[:, None] - discounts_all[None, :])
            delta_gain_pos = torch.clamp(diff_true, min=0.0)
            delta_dcg = delta_gain_pos * delta_discount
            weights = delta_dcg / idcg_at_k
            weights = weights * positive_pairs.float() * topk_pair_mask.float()

            sum_weights = weights.sum()
            if sum_weights.item() <= 0:
                continue

            pair_losses = F.softplus(-diff_pred)
            weighted_loss = (pair_losses * weights).sum()

            total_loss = total_loss + weighted_loss
            total_weight = total_weight + sum_weights

        if total_weight.item() == 0:
            return torch.tensor(0.0, device=pred_importance.device)
        return total_loss / total_weight
    
    def _calculate_nucleotide_preference_accuracy(self,
                                              pred_nuc_pref: torch.Tensor,
                                              contact_values: torch.Tensor,
                                              mask: torch.Tensor) -> torch.Tensor:
        
        pred_max_nuc = pred_nuc_pref.argmax(dim=-1)
        true_max_nuc = contact_values.argmax(dim=-1)
        
        aa_mask = mask[:, :, 0] & (contact_values.max(dim=-1)[0] > 0)
        
        if not aa_mask.any():
            return torch.tensor(0.0, device=pred_nuc_pref.device)
        
        correct = (pred_max_nuc[aa_mask] == true_max_nuc[aa_mask]).float().sum()
        total = aa_mask.sum()
        
        return correct / total
    
    @staticmethod
    def _create_sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        
        return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
