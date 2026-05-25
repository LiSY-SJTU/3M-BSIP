

from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve)


class ContactMetricsCalculator:
    def __init__(self, contact_threshold: float = 0.5):
        self.contact_threshold = contact_threshold
        self.reset()
        self.topk_list = (5, 10)
        self.all_probs = []
        self.all_labels = []
        self.all_masks = []
        
        self.all_interface_probs = []
        self.all_interface_labels = []
        self.all_interface_masks = []
        
        self.nuc_pref_correct = 0
        self.nuc_pref_total = 0
    
    def reset(self):
        
        self.classification_stats = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_samples': 0
        }
        
        self.interface_stats = {
            'interface_true_positives': 0,
            'interface_true_negatives': 0,
            'interface_false_positives': 0,
            'interface_false_negatives': 0
        }
        
        self.loss_stats = {
            'loss': 0.0,
            'cls_loss': 0.0,
            'nuc_pref_loss': 0.0,
            'rank_loss': 0.0
        }
        
        self.total_stats = defaultdict(float)
        self.num_samples = 0
        
        self.all_probs = []
        self.all_labels = []
        self.all_masks = []
        self.all_interface_probs = []
        self.all_interface_labels = []
        self.all_interface_masks = []
        
        self.nuc_pref_correct = 0
        self.nuc_pref_total = 0
    
    def update(self, predictions: Dict[str, torch.Tensor],
              targets: Dict[str, torch.Tensor],
              mask: torch.Tensor,
              loss_dict: Dict[str, float],
              batch_size: int):
        
        pred_probs = torch.sigmoid(predictions['contact_cls'])
        target_cls = (targets['contact_tensor'] > 0).float()

        self.all_probs.append(pred_probs.detach().cpu())
        self.all_labels.append(target_cls.detach().cpu())
        self.all_masks.append(mask.detach().cpu())
        
        pred_interface_probs = pred_probs.max(dim=-1)[0]
        true_interface = target_cls.max(dim=-1)[0]
        interface_mask = mask[:, :, 0]
        
        self.all_interface_probs.append(pred_interface_probs.detach().cpu())
        self.all_interface_labels.append(true_interface.detach().cpu())
        self.all_interface_masks.append(interface_mask.detach().cpu())
        
        batch_metrics = self.calculate_metrics(predictions, targets, mask)
        
        for k in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives', 'total_samples']:
            self.classification_stats[k] += int(batch_metrics[k])
        
        for k in ['interface_true_positives', 'interface_true_negatives', 
                 'interface_false_positives', 'interface_false_negatives']:
            self.interface_stats[k] += int(batch_metrics[k])
        
        for k in ['loss', 'cls_loss', 'rank_loss', 'nuc_pref_loss']:
            if k in loss_dict:
                if isinstance(loss_dict[k], torch.Tensor):
                    self.loss_stats[k] += loss_dict[k].detach().item() * batch_size
                else:
                    self.loss_stats[k] += loss_dict[k] * batch_size
        
        for k, v in batch_metrics.items():
            if k not in self.classification_stats and k not in self.interface_stats and k not in ['nuc_pref_accuracy']:
                if isinstance(v, torch.Tensor):
                    self.total_stats[k] += v.detach().item() * batch_size
                else:
                    self.total_stats[k] += v * batch_size
        
        self.num_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        
        avg_stats = {}
        if self.num_samples > 0:
            for k in self.loss_stats:
                avg_stats[k] = self.loss_stats[k] / self.num_samples
            
            for k, v in self.total_stats.items():
                avg_stats[k] = v / self.num_samples
        
        total = self.classification_stats['total_samples']
        if total > 0:
            tp = self.classification_stats['true_positives']
            tn = self.classification_stats['true_negatives']
            fp = self.classification_stats['false_positives']
            fn = self.classification_stats['false_negatives']
            
            avg_stats['accuracy'] = (tp + tn) / total
            avg_stats['precision'] = tp / (tp + fp + 1e-6)
            avg_stats['recall'] = tp / (tp + fn + 1e-6)
            avg_stats['f1'] = 2 * (avg_stats['precision'] * avg_stats['recall']) / (avg_stats['precision'] + avg_stats['recall'] + 1e-6)
            
            avg_stats.update(self.classification_stats)
        
        tp = self.interface_stats['interface_true_positives']
        tn = self.interface_stats['interface_true_negatives']
        fp = self.interface_stats['interface_false_positives']
        fn = self.interface_stats['interface_false_negatives']
        total = tp + tn + fp + fn
        
        if total > 0:
            avg_stats['interface_acc'] = (tp + tn) / total
            avg_stats['interface_precision'] = tp / (tp + fp + 1e-6)
            avg_stats['interface_recall'] = tp / (tp + fn + 1e-6)
            avg_stats['interface_f1'] = 2 * (avg_stats['interface_precision'] * avg_stats['interface_recall']) / (avg_stats['interface_precision'] + avg_stats['interface_recall'] + 1e-6)
            
            avg_stats.update(self.interface_stats)
        
        if self.nuc_pref_total > 0:
            avg_stats['nuc_pref_accuracy'] = self.nuc_pref_correct / self.nuc_pref_total
        else:
            avg_stats['nuc_pref_accuracy'] = 0.0
        
        all_probs = []
        all_labels = []
        
        for probs, labels, mask in zip(self.all_probs, self.all_labels, self.all_masks):
            valid_mask = mask.bool()
            all_probs.extend(probs[valid_mask].numpy().flatten())
            all_labels.extend(labels[valid_mask].numpy().flatten())
        
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            valid_indices = [i for i, (p, l) in enumerate(zip(all_probs, all_labels)) 
                           if not (np.isnan(p) or np.isnan(l))]
            if len(valid_indices) > 0:
                filtered_probs = [all_probs[i] for i in valid_indices]
                filtered_labels = [all_labels[i] for i in valid_indices]
                if len(set(filtered_labels)) > 1:
                    precision, recall, _ = precision_recall_curve(filtered_labels, filtered_probs)
                    pr_auc = auc(recall, precision)
                else:
                    pr_auc = 0.0
            else:
                pr_auc = 0.0
        else:
            pr_auc = 0.0
        
        all_interface_probs = []
        all_interface_labels = []
        
        for probs, labels, mask in zip(self.all_interface_probs, self.all_interface_labels, self.all_interface_masks):
            valid_mask = mask.bool()
            all_interface_probs.extend(probs[valid_mask].numpy().flatten())
            all_interface_labels.extend(labels[valid_mask].numpy().flatten())
        
        if len(all_interface_labels) > 0 and len(set(all_interface_labels)) > 1:
            valid_indices = [i for i, (p, l) in enumerate(zip(all_interface_probs, all_interface_labels)) 
                           if not (np.isnan(p) or np.isnan(l))]
            if len(valid_indices) > 0:
                filtered_probs = [all_interface_probs[i] for i in valid_indices]
                filtered_labels = [all_interface_labels[i] for i in valid_indices]
                if len(set(filtered_labels)) > 1:
                    interface_precision, interface_recall, _ = precision_recall_curve(filtered_labels, filtered_probs)
                    interface_pr_auc = auc(interface_recall, interface_precision)
                else:
                    interface_pr_auc = 0.0
            else:
                interface_pr_auc = 0.0
        else:
            interface_pr_auc = 0.0
        
        avg_stats['pr_auc'] = pr_auc
        avg_stats['interface_pr_auc'] = interface_pr_auc
        
        return avg_stats
    
    def calculate_metrics(self, predictions: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor],
                         mask: torch.Tensor) -> Dict[str, float]:
        
        pred_probs = torch.sigmoid(predictions['contact_cls'])
        pred_binary = (pred_probs > self.contact_threshold).float()
        target_cls = (targets['contact_tensor'] > 0).float()
        
        classification_stats = self._calculate_classification_stats(pred_binary, target_cls, mask)
        
        interface_stats = self._calculate_interface_stats(pred_binary, target_cls, mask)
        
        nuc_pref_stats = {}
        if 'nuc_preference' in predictions:
            nuc_pref_stats = self._calculate_nucleotide_preference_stats(
                predictions['nuc_preference'], 
                targets['contact_tensor'], 
                mask
            )
            if 'nuc_pref_accuracy' in nuc_pref_stats:
                aa_mask = mask[:, :, 0] & (targets['contact_tensor'].max(dim=-1)[0] > 0)
                if aa_mask.any():
                    pred_max_nuc = predictions['nuc_preference'].argmax(dim=-1)
                    true_max_nuc = targets['contact_tensor'].argmax(dim=-1)
                    correct = (pred_max_nuc[aa_mask] == true_max_nuc[aa_mask]).sum().item()
                    total = aa_mask.sum().item()
                    self.nuc_pref_correct += correct
                    self.nuc_pref_total += total
        
        ranking_stats = {}
        if 'binding_importance' in predictions:
            ranking_stats = self._calculate_ranking_metrics(
                predictions['binding_importance'],
                targets['contact_tensor'],
                mask[:, :, 0]
            )
        
        metrics = {}
        metrics.update(classification_stats)
        metrics.update(interface_stats)
        metrics.update(nuc_pref_stats)
        metrics.update(ranking_stats)
        
        return metrics
    
    def _update_nucleotide_preference_stats(self, 
                                         pred_nuc_pref: torch.Tensor,
                                         contact_values: torch.Tensor,
                                         mask: torch.Tensor):
        
        pred_max_nuc = pred_nuc_pref.argmax(dim=-1)
        true_max_nuc = contact_values.argmax(dim=-1)
        
        aa_mask = mask[:, :, 0] & (contact_values.max(dim=-1)[0] > 0)
        
        if not aa_mask.any():
            return
        
        correct = (pred_max_nuc[aa_mask] == true_max_nuc[aa_mask]).sum().item()
        total = aa_mask.sum().item()
        
        self.nuc_pref_correct += correct
        self.nuc_pref_total += total
    
    def _calculate_classification_stats(self, pred_binary, target_cls, mask):
        
        valid_mask = mask.bool()
        
        true_positives = ((pred_binary == 1) & (target_cls == 1) & valid_mask).sum().item()
        true_negatives = ((pred_binary == 0) & (target_cls == 0) & valid_mask).sum().item()
        false_positives = ((pred_binary == 1) & (target_cls == 0) & valid_mask).sum().item()
        false_negatives = ((pred_binary == 0) & (target_cls == 1) & valid_mask).sum().item()
        
        total = valid_mask.sum().item()
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_samples': total
        }
    
    def _calculate_interface_stats(self, pred_binary, target_cls, mask):
        
        pred_interface = (pred_binary.max(dim=-1)[0] > 0).float()
        true_interface = (target_cls.max(dim=-1)[0] > 0).float()
        
        valid_mask = mask[:, :, 0].bool()
        valid_pred_interface = pred_interface[valid_mask]
        valid_true_interface = true_interface[valid_mask]
        
        tp = ((valid_pred_interface == 1) & (valid_true_interface == 1)).sum().item()
        tn = ((valid_pred_interface == 0) & (valid_true_interface == 0)).sum().item()
        fp = ((valid_pred_interface == 1) & (valid_true_interface == 0)).sum().item()
        fn = ((valid_pred_interface == 0) & (valid_true_interface == 1)).sum().item()
        
        total = tp + tn + fp + fn
        interface_acc = (tp + tn) / total if total > 0 else 0.0
        interface_precision = tp / (tp + fp + 1e-8)
        interface_recall = tp / (tp + fn + 1e-8)
        interface_f1 = 2 * (interface_precision * interface_recall) / (interface_precision + interface_recall + 1e-8)
        
        return {
            'interface_acc': interface_acc,
            'interface_precision': interface_precision,
            'interface_recall': interface_recall,
            'interface_f1': interface_f1,
            'interface_true_positives': tp,
            'interface_true_negatives': tn,
            'interface_false_positives': fp,
            'interface_false_negatives': fn
        }
    
    
    def _calculate_nucleotide_preference_stats(self, 
                                            pred_nuc_pref: torch.Tensor,
                                            contact_values: torch.Tensor,
                                            mask: torch.Tensor) -> Dict[str, float]:
        
        pred_max_nuc = pred_nuc_pref.argmax(dim=-1)
        true_max_nuc = contact_values.argmax(dim=-1)
        
        aa_mask = mask[:, :, 0] & (contact_values.max(dim=-1)[0] > 0)
        
        if not aa_mask.any():
            return {'nuc_pref_accuracy': 0.0}
        
        correct = (pred_max_nuc[aa_mask] == true_max_nuc[aa_mask]).float().sum().item()
        total = aa_mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {'nuc_pref_accuracy': accuracy}

    def _calculate_ranking_metrics(self,
                                   pred_importance: torch.Tensor,
                                   contact_values: torch.Tensor,
                                   seq_mask: torch.Tensor) -> Dict[str, float]:
        
        true_importance = contact_values.max(dim=-1)[0]
        
        sums = {
            'precision_at_5': 0.0, 'precision_at_10': 0.0,
            'recall_at_5': 0.0, 'recall_at_10': 0.0,
            'ndcg_at_5': 0.0, 'ndcg_at_10': 0.0,
            'spearman': 0.0,
        }
        counts = {k: 0 for k in sums.keys()}
        
        B = pred_importance.size(0)
        for b in range(B):
            valid = seq_mask[b].bool()
            if valid.sum().item() == 0:
                continue
            s = pred_importance[b, valid]
            y = true_importance[b, valid]
            num_valid = int(s.numel())
            
            pos = (y > 0)
            num_pos = int(pos.sum().item())
            pred_sorted_idx = torch.argsort(s, descending=True)
            
            for K in self.topk_list:
                k_eff = int(min(K, num_valid))
                if k_eff <= 0:
                    continue
                topk_pred_idx = pred_sorted_idx[:k_eff]
                
                prec_correct = pos[topk_pred_idx].float().sum().item()
                sums[f'precision_at_{K}'] += prec_correct / float(k_eff)
                counts[f'precision_at_{K}'] += 1
                
                if num_pos > 0:
                    recall_correct = pos[topk_pred_idx].float().sum().item()
                    sums[f'recall_at_{K}'] += recall_correct / float(num_pos)
                    counts[f'recall_at_{K}'] += 1
                
                gains = y[pred_sorted_idx[:k_eff]]
                discounts = 1.0 / torch.log2(torch.arange(2, 2 + k_eff, device=y.device).float())
                dcg = (gains * discounts).sum().item()
                ideal_gains = torch.sort(y, descending=True).values[:k_eff]
                idcg = (ideal_gains * discounts).sum().item()
                ndcg = (dcg / idcg) if idcg > 0 else 0.0
                sums[f'ndcg_at_{K}'] += ndcg
                counts[f'ndcg_at_{K}'] += 1
            
            if num_valid >= 2 and pos.sum().item() > 0:
                def rankify(v: torch.Tensor) -> torch.Tensor:
                    order = torch.argsort(v)
                    ranks = torch.empty_like(order, dtype=torch.float)
                    ranks[order] = torch.arange(1, v.numel() + 1, device=v.device, dtype=torch.float)
                    return ranks
                rx = rankify(s)
                ry = rankify(y)
                rx = rx - rx.mean()
                ry = ry - ry.mean()
                rx_std = rx.std(unbiased=False)
                ry_std = ry.std(unbiased=False)
                if rx_std > 1e-8 and ry_std > 1e-8:
                    spearman_val = float((rx * ry).mean().item() / (rx_std * ry_std).item())
                    sums['spearman'] += spearman_val
                    counts['spearman'] += 1
        
        out = {}
        for k in sums:
            out[k] = sums[k] / counts[k] if counts[k] > 0 else 0.0
        return out
    
    