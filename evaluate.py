

import shutup; shutup.please()
import argparse
import csv
import json
import logging
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import TFDNADataset
from src.data import collate_fn as original_collate_fn
from src.metrics.contact_metrics import ContactMetricsCalculator
from src.models import DBP2Predictor
from src.models.loss import ContactHybridLoss
from src.utils.logger import setup_logger

torch.backends.cudnn.benchmark = True

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_mcc(tp, tn, fp, fn):
    
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def move_to_device(batch, device):
    
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    elif hasattr(batch, 'to'):
        return batch.to(device)
    return batch

def load_complex_labels(apo_pdb_id, complex_cache_dir, device, matching_map=None):
    
    logger = logging.getLogger('evaluate')

    if matching_map is not None:
        if apo_pdb_id not in matching_map:
            raise KeyError(f"Apo id '{apo_pdb_id}' not found in matching report. Provide a correct mapping or unset matching_report_path.")
        complex_name = matching_map[apo_pdb_id]
        complex_cache_path = os.path.join(complex_cache_dir, complex_name)
        if not os.path.exists(complex_cache_path):
            raise FileNotFoundError(f"Complex cache file for '{apo_pdb_id}' not found: {complex_cache_path}")
        complex_data = torch.load(complex_cache_path, map_location=device)
    else:
        base_name = apo_pdb_id.replace('_apo_model', '')
        complex_name = f"{base_name}.pt"
        complex_cache_path = os.path.join(complex_cache_dir, complex_name)
        if not os.path.exists(complex_cache_path):
            raise FileNotFoundError(f"Expected complex cache for '{apo_pdb_id}' not found at: {complex_cache_path}")
        complex_data = torch.load(complex_cache_path, map_location=device)

    if 'contact_tensor' not in complex_data:
        raise KeyError(f"'contact_tensor' missing in complex cache file: {complex_cache_path}")

    return {
        'contact_tensor': complex_data['contact_tensor'].to(device),
        'preference_tensor': complex_data.get('preference_tensor', None).to(device) if complex_data.get('preference_tensor') is not None else None,
        'strength_tensor': complex_data.get('strength_tensor', None).to(device) if complex_data.get('strength_tensor') is not None else None
    }

def parse_matching_report(report_path):
    
    logger = logging.getLogger('evaluate')
    if not report_path:
        raise ValueError("matching_report_path is empty or not provided.")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Matching report not found at: {report_path}")

    mapping = {}
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split('Monomer PDB ID:')[1:]

    for block in blocks:
        apo_file_match = re.search(r'Monomer file:\s*(\S+\.pdb)', block)
        complex_file_match = re.search(r'Complex file:\s*(\S+\.pt)', block)

        if apo_file_match and complex_file_match:
            apo_filename_base = os.path.splitext(apo_file_match.group(1))[0]
            complex_filename = complex_file_match.group(1)
            mapping[apo_filename_base] = complex_filename
    
    if not mapping:
        raise ValueError(f"Could not parse any valid apo-complex pairs from {report_path}.")
        
    return mapping

class CollateWrapper:
    def __init__(self, core_length):
        self.core_length = core_length

    def __call__(self, batch):
        return original_collate_fn(batch, self.core_length)

def log_stats(logger, stats, prefix):
    
    protein_count = stats.get('num_protein_samples', 0)
    residue_count = stats.get('num_amino_acid_samples', 0)
    interface_pr_auc = stats.get('interface_pr_auc', float('nan'))
    interface_mcc = stats.get('interface_mcc', float('nan'))
    interface_base_pref_Acc = stats.get('interface_base_pref_Acc', float('nan'))
    hotspot_base_pref_Acc = stats.get('hotspot_base_pref_Acc', float('nan'))
    ndcg5 = stats.get('ndcg_at_5', float('nan'))
    ndcg10 = stats.get('ndcg_at_10', float('nan'))

    logger.info(
        f"{prefix}: "
        f"number of proteins: {protein_count} | "
        f"number of residues: {residue_count} | "
        f"interface_pr_auc: {interface_pr_auc:.3f} | "
        f"interface_mcc: {interface_mcc:.3f} | "
        f"interface_base_pref_Acc: {interface_base_pref_Acc:.3f} | "
        f"hotspot_base_pref_Acc: {hotspot_base_pref_Acc:.3f} | "
        f"ndcg_at_5: {ndcg5:.3f} | ndcg_at_10: {ndcg10:.3f}"
    )

def log_classification_stats(logger, stats, prefix=""):
    
    total = stats.get('total_samples', 0)
    if total == 0:
        logger.info(f"{prefix} Classification Stats: No valid samples.")
        return
        
    tp = stats.get('true_positives', 0)
    tn = stats.get('true_negatives', 0)
    fp = stats.get('false_positives', 0)
    fn = stats.get('false_negatives', 0)

    logger.info(f"{prefix} Classification Stats:")
    logger.info(f"├─ Total Valid Samples (contact_cls): {total:,}")
    logger.info("├─ Correct Classifications:")
    logger.info(f"│  ├─ True Positives (TP): {tp:,} ({tp/total*100:.2f}%)")
    logger.info(f"│  └─ True Negatives (TN): {tn:,} ({tn/total*100:.2f}%)")
    logger.info("└─ Incorrect Classifications:")
    logger.info(f"   ├─ False Positives (FP): {fp:,} ({fp/total*100:.2f}%)")
    logger.info(f"   └─ False Negatives (FN): {fn:,} ({fn/total*100:.2f}%)")

def log_interface_stats(logger, stats, prefix=""):
    
    itp = stats.get('interface_true_positives', 0)
    itn = stats.get('interface_true_negatives', 0)
    ifp = stats.get('interface_false_positives', 0)
    ifn = stats.get('interface_false_negatives', 0)
    total_interface = itp + itn + ifp + ifn

    if total_interface == 0:
        logger.info(f"{prefix} Interface Prediction Stats: No valid interface samples.")
        return
        
    logger.info(f"{prefix} Interface Prediction Stats:")
    logger.info(f"├─ Total Interface Samples: {int(total_interface):,}")
    logger.info("├─ Correct Classifications:")
    logger.info(f"│  ├─ True Positives (TP): {int(itp):,} ({itp/total_interface*100:.2f}%)")
    logger.info(f"│  └─ True Negatives (TN): {int(itn):,} ({itn/total_interface*100:.2f}%)")
    logger.info("└─ Incorrect Classifications:")
    logger.info(f"   ├─ False Positives (FP): {int(ifp):,} ({ifp/total_interface*100:.2f}%)")
    logger.info(f"   └─ False Negatives (FN): {int(ifn):,} ({ifn/total_interface*100:.2f}%)")

def load_trained_model(checkpoint_path, device, args):
    
    logger = logging.getLogger('evaluate')
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'args' not in checkpoint:
        logger.warning("Checkpoint does not contain 'args'. Model hyperparameters (hidden_dim, dropout, dna_core_length) must be provided via command line or defaults will be used.")
        cp_args_namespace = argparse.Namespace()
    else:
        cp_args_namespace = checkpoint['args']

    hidden_dim = args.hidden_dim if args.hidden_dim is not None else getattr(cp_args_namespace, 'hidden_dim', 256)
    dropout = args.dropout if args.dropout is not None else getattr(cp_args_namespace, 'dropout', 0.25)
    if hasattr(cp_args_namespace, 'dna_core_length') and cp_args_namespace.dna_core_length != args.dna_core_length:
         logger.warning(f"Model in checkpoint was trained with dna_core_length={cp_args_namespace.dna_core_length}, "
                        f"but evaluation is using dna_core_length={args.dna_core_length} for data processing.")

    model = DBP2Predictor(
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully.")
    logger.info(f"  Model hidden_dim: {hidden_dim}")
    logger.info(f"  Model dropout: {dropout}")
    if hasattr(cp_args_namespace, 'dna_core_length'):
         logger.info(f"  Model was trained with DNA core length: {cp_args_namespace.dna_core_length}")
    
    return model, cp_args_namespace

def create_test_dataloader(test_dataset, args):
    
    collate_instance = CollateWrapper(args.dna_core_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instance,
        pin_memory=True
    )
    return test_loader

def evaluate_epoch(model, dataloader, criterion, device, logger, args, test_pdb_files, matching_map=None):
    
    model.eval()
    metrics_calculator = ContactMetricsCalculator(contact_threshold=criterion.contact_threshold)
    
    use_complex_labels = args.complex_cache_dir is not None
    if use_complex_labels:
        logger.info(f"Using Apo Evaluation Mode: Loading labels from complex cache directory: {args.complex_cache_dir}")
    
    aa_types = "ACDEFGHIKLMNPQRSTVWY"
    aa_metrics_calculators = {aa: ContactMetricsCalculator(contact_threshold=criterion.contact_threshold) 
                             for aa in aa_types}
    aa_strong_contacts = {aa: 0 for aa in aa_types}
    aa_correct_interface = {aa: 0 for aa in aa_types}
    aa_correct_nuc_pref = {aa: 0 for aa in aa_types}
    
    total_strong_contact_residues = 0
    total_protein_count = 0
    total_residue_count = 0
    correct_strong_interface_sites = 0
    correct_strong_nuc_pref_sites = 0
    per_protein_results = {}
    
    dataset = dataloader.dataset

    metrics_to_remove = ['loss', 'cls_loss', 'nuc_pref_loss', 'rank_loss', 'accuracy', 'max_nuc_accuracy']

    aa_to_id = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21, '-': 0
    }
    id_to_aa = {v: k for k, v in aa_to_id.items()}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = move_to_device(batch, device)
                protein_features_input = batch['protein_features']
                
                protein_sequences = []
                batch_size = batch['protein_lengths'].size(0)
                total_protein_count += batch_size
                total_residue_count += int(batch['protein_lengths'].sum().item())
                
                if args.debug_mode and batch_idx == 0:
                    logger.debug(f"Protein features keys: {protein_features_input.keys()}")
                    logger.debug(f"Sequence data keys: {protein_features_input['sequence'].keys()}")
                
                for i in range(batch_size):
                    length = batch['protein_lengths'][i].item()
                    if 'combined_seq' in protein_features_input['sequence']:
                        seq_field = protein_features_input['sequence']['combined_seq']
                        if isinstance(seq_field, (list, tuple)):
                            if len(seq_field) != batch_size:
                                raise ValueError(f"combined_seq length {len(seq_field)} does not match batch size {batch_size} for sample index {i}")
                            seq = seq_field[i]
                        elif isinstance(seq_field, str):
                            if batch_size != 1:
                                raise ValueError("combined_seq is a single string but batch_size > 1")
                            seq = seq_field
                        else:
                            raise TypeError(f"combined_seq must be list/tuple of strings or string, got {type(seq_field)}")
                        if len(seq) < length:
                            raise ValueError(f"Protein sequence length {len(seq)} is shorter than declared length {length} at index {i}")
                        protein_sequences.append(seq[:length])
                        continue
                    
                    raise KeyError("Neither 'combined_seq' nor 'sequence_ids' provided in protein_features_input['sequence']")
                
                predictions = model(protein_features_input)
                
                if use_complex_labels:
                    batch_size = batch['protein_lengths'].size(0)
                    complex_contact_tensors = []
                    
                    for i in range(batch_size):
                        dataset_idx = batch_idx * dataloader.batch_size + i
                        if dataset_idx >= len(dataset):
                            raise IndexError(f"Dataset index {dataset_idx} out of range for dataset length {len(dataset)}")
                        
                        original_idx = dataset.get_original_index(dataset_idx)
                        pdb_path = test_pdb_files[original_idx]
                        apo_pdb_id = os.path.splitext(os.path.basename(str(pdb_path)))[0]
                        
                        complex_labels_data = load_complex_labels(apo_pdb_id, args.complex_cache_dir, device, matching_map)
                        complex_contact_tensors.append(complex_labels_data['contact_tensor'])
                    
                    if complex_contact_tensors:
                        max_len = max(tensor.size(0) for tensor in complex_contact_tensors)
                        padded_tensors = []
                        for tensor in complex_contact_tensors:
                            if tensor.size(0) < max_len:
                                padding = torch.zeros(max_len - tensor.size(0), tensor.size(1), 
                                                    dtype=tensor.dtype, device=tensor.device)
                                padded_tensor = torch.cat([tensor, padding], dim=0)
                            else:
                                padded_tensor = tensor
                            padded_tensors.append(padded_tensor)
                        
                        batch['contact_tensor'] = torch.stack(padded_tensors)
                
                if 'contact_tensor' not in batch or batch['contact_tensor'] is None:
                    raise KeyError(f"Batch {batch_idx} is missing 'contact_tensor'.")
                
                labels = {
                    'contact_tensor': batch['contact_tensor'],
                    'protein_lengths': batch['protein_lengths']
                }
                
                mask = criterion._create_sequence_mask(
                    batch['protein_lengths'],
                    predictions['contact_cls'].size(1)
                ).unsqueeze(-1).expand(-1, -1, 4).to(device)
                
                metrics_calculator.update(
                    predictions=predictions,
                    targets=labels,
                    mask=mask,
                    loss_dict={},
                    batch_size=batch['protein_lengths'].size(0)
                )
                
                contact_tensor_true = labels['contact_tensor']
                contact_cls_pred_logits = predictions['contact_cls']
                nuc_preference_pred_logits = predictions['nuc_preference']
                protein_lengths = labels['protein_lengths']
                batch_size = contact_tensor_true.size(0)
                
                for i in range(batch_size):
                    dataset_idx = batch_idx * dataloader.batch_size + i
                    if dataset_idx >= len(dataset):
                        raise IndexError(f"Dataset index {dataset_idx} out of range for dataset length {len(dataset)}")
                    
                    original_idx = dataset.get_original_index(dataset_idx)
                    pdb_path = test_pdb_files[original_idx]
                    pdb_id = os.path.splitext(os.path.basename(str(pdb_path)))[0]
                    
                    protein_calculator = ContactMetricsCalculator(contact_threshold=criterion.contact_threshold)
                    length = protein_lengths[i].item()
                    single_protein_mask = mask[i:i+1, :length, :].clone()
                    
                    protein_seq = protein_sequences[i] if i < len(protein_sequences) else ""
                    if len(protein_seq) < length:
                        raise ValueError(f"Protein sequence for {pdb_id} is too short ({len(protein_seq)} < {length}).")
                    protein_seq = protein_seq[:length]
                    
                    single_protein_predictions = {
                        'contact_cls': contact_cls_pred_logits[i:i+1, :length, :],
                        'binding_importance': predictions['binding_importance'][i:i+1, :length],
                        'nuc_preference': nuc_preference_pred_logits[i:i+1, :length, :]
                    }
                    
                    if use_complex_labels:
                        dataset_idx = batch_idx * dataloader.batch_size + i
                        original_idx = dataset.get_original_index(dataset_idx)
                        pdb_path = test_pdb_files[original_idx]
                        apo_pdb_id = os.path.splitext(os.path.basename(str(pdb_path)))[0]
                        complex_labels_data = load_complex_labels(apo_pdb_id, args.complex_cache_dir, device, matching_map)
                        single_protein_contact_tensor = complex_labels_data['contact_tensor'][:length, :]
                    else:
                        single_protein_contact_tensor = contact_tensor_true[i, :length, :]
                    
                    single_protein_labels = {
                        'contact_tensor': single_protein_contact_tensor.unsqueeze(0),
                        'protein_lengths': torch.tensor([length], device=device)
                    }
                    
                    protein_calculator.update(
                        predictions=single_protein_predictions,
                        targets=single_protein_labels,
                        mask=single_protein_mask,
                        loss_dict={},
                        batch_size=1
                    )
                    
                    true_contacts_sample = contact_tensor_true[i, :length, :]
                    pred_cls_logits_sample = contact_cls_pred_logits[i, :length, :]
                    pred_nuc_pref_logits_sample = nuc_preference_pred_logits[i, :length, :]
                    
                    if use_complex_labels:
                        true_contacts_sample = single_protein_contact_tensor
                    else:
                        true_contacts_sample = contact_tensor_true[i, :length, :]
                    
                    strong_contact_mask_per_nuc = (true_contacts_sample > args.strong_contact_threshold)
                    strong_contact_residue_mask = strong_contact_mask_per_nuc.any(dim=-1)
                    num_strong_res_in_sample = strong_contact_residue_mask.sum().item()
                    protein_strong_interface_correct = 0
                    protein_strong_nuc_pref_correct = 0
                    
                    if num_strong_res_in_sample > 0:
                        total_strong_contact_residues += num_strong_res_in_sample
                        pred_interface_prob_sample = torch.sigmoid(pred_cls_logits_sample)
                        pred_interface_for_residue = (pred_interface_prob_sample > 0.5).any(dim=-1)
                        protein_strong_interface_correct = (pred_interface_for_residue & strong_contact_residue_mask).sum().item()
                        correct_strong_interface_sites += protein_strong_interface_correct
                        true_contacts_strong_residues = true_contacts_sample[strong_contact_residue_mask]
                        true_max_nuc_indices = true_contacts_strong_residues.argmax(dim=-1)
                        pred_nuc_pref_logits_strong_residues = pred_nuc_pref_logits_sample[strong_contact_residue_mask]
                        pred_max_nuc_indices = pred_nuc_pref_logits_strong_residues.argmax(dim=-1)
                        protein_strong_nuc_pref_correct = (true_max_nuc_indices == pred_max_nuc_indices).sum().item()
                        correct_strong_nuc_pref_sites += protein_strong_nuc_pref_correct
                        
                        for pos, is_strong in enumerate(strong_contact_residue_mask):
                            if is_strong:
                                aa = protein_seq[pos] if pos < len(protein_seq) else 'X'
                                if aa in aa_types:
                                    aa_strong_contacts[aa] += 1
                                    if pred_interface_for_residue[pos]:
                                        aa_correct_interface[aa] += 1
                                    
                                    strong_idx = (strong_contact_residue_mask[:pos+1].sum() - 1).item()
                                    if strong_idx >= 0 and true_max_nuc_indices[strong_idx] == pred_max_nuc_indices[strong_idx]:
                                        aa_correct_nuc_pref[aa] += 1
                    
                    for pos in range(length):
                        aa = protein_seq[pos] if pos < len(protein_seq) else 'X'
                        if aa in aa_types:
                            aa_predictions = {
                                'contact_cls': contact_cls_pred_logits[i:i+1, pos:pos+1, :],
                                'binding_importance': predictions['binding_importance'][i:i+1, pos:pos+1],
                                'nuc_preference': nuc_preference_pred_logits[i:i+1, pos:pos+1, :]
                            }
                            
                            if use_complex_labels:
                                aa_contact_tensor = single_protein_contact_tensor[pos:pos+1, :]
                            else:
                                aa_contact_tensor = contact_tensor_true[i, pos:pos+1, :]
                            
                            aa_labels = {
                                'contact_tensor': aa_contact_tensor.unsqueeze(0),
                                'protein_lengths': torch.tensor([1], device=device)
                            }
                            
                            aa_mask = single_protein_mask[:, pos:pos+1, :]
                            
                            aa_metrics_calculators[aa].update(
                                predictions=aa_predictions,
                                targets=aa_labels,
                                mask=aa_mask,
                                loss_dict={},
                                batch_size=1
                            )
                    
                    protein_stats = protein_calculator.compute()

                    contact_tp = protein_stats.get('true_positives', 0)
                    contact_tn = protein_stats.get('true_negatives', 0)
                    contact_fp = protein_stats.get('false_positives', 0)
                    contact_fn = protein_stats.get('false_negatives', 0)
                    protein_stats['contact_mcc'] = calculate_mcc(contact_tp, contact_tn, contact_fp, contact_fn)

                    interface_tp = protein_stats.get('interface_true_positives', 0)
                    interface_tn = protein_stats.get('interface_true_negatives', 0)
                    interface_fp = protein_stats.get('interface_false_positives', 0)
                    interface_fn = protein_stats.get('interface_false_negatives', 0)
                    protein_stats['interface_mcc'] = calculate_mcc(interface_tp, interface_tn, interface_fp, interface_fn)
                    
                    for metric_key in metrics_to_remove:
                        protein_stats.pop(metric_key, None)

                    protein_stats['strong_contact_threshold_value'] = args.strong_contact_threshold
                    protein_stats['strong_total_sites'] = num_strong_res_in_sample
                    protein_stats['strong_correct_interface_sites'] = protein_strong_interface_correct
                    protein_stats['strong_correct_nuc_pref_sites'] = protein_strong_nuc_pref_correct
                    if num_strong_res_in_sample > 0:
                        protein_stats['strong_interface_accuracy'] = protein_strong_interface_correct / num_strong_res_in_sample
                        protein_stats['strong_nuc_pref_accuracy'] = protein_strong_nuc_pref_correct / num_strong_res_in_sample
                    else:
                        protein_stats['strong_interface_accuracy'] = 0.0
                        protein_stats['strong_nuc_pref_accuracy'] = 0.0
                    protein_stats['protein_length'] = length
                    per_protein_results[pdb_id] = protein_stats
                
                if (batch_idx + 1) % args.log_interval == 0:
                    logger.info(f"Evaluated batch {batch_idx + 1}/{len(dataloader)}")
            
            except Exception as e:
                raise
    
    avg_stats = metrics_calculator.compute()

    contact_tp_avg = avg_stats.get('true_positives', 0)
    contact_tn_avg = avg_stats.get('true_negatives', 0)
    contact_fp_avg = avg_stats.get('false_positives', 0)
    contact_fn_avg = avg_stats.get('false_negatives', 0)
    avg_stats['contact_mcc'] = calculate_mcc(contact_tp_avg, contact_tn_avg, contact_fp_avg, contact_fn_avg)

    interface_tp_avg = avg_stats.get('interface_true_positives', 0)
    interface_tn_avg = avg_stats.get('interface_true_negatives', 0)
    interface_fp_avg = avg_stats.get('interface_false_positives', 0)
    interface_fn_avg = avg_stats.get('interface_false_negatives', 0)
    avg_stats['interface_mcc'] = calculate_mcc(interface_tp_avg, interface_tn_avg, interface_fp_avg, interface_fn_avg)

    for metric_key in metrics_to_remove:
        avg_stats.pop(metric_key, None)
        
    avg_stats['strong_contact_threshold_value'] = args.strong_contact_threshold
    avg_stats['strong_total_sites'] = total_strong_contact_residues
    avg_stats['strong_correct_interface_sites'] = correct_strong_interface_sites
    avg_stats['strong_correct_nuc_pref_sites'] = correct_strong_nuc_pref_sites
    if total_strong_contact_residues > 0:
        avg_stats['strong_interface_accuracy'] = correct_strong_interface_sites / total_strong_contact_residues
        avg_stats['strong_nuc_pref_accuracy'] = correct_strong_nuc_pref_sites / total_strong_contact_residues
    else:
        avg_stats['strong_interface_accuracy'] = 0.0
        avg_stats['strong_nuc_pref_accuracy'] = 0.0
    
    avg_stats['num_protein_samples'] = total_protein_count
    avg_stats['num_amino_acid_samples'] = total_residue_count

    if 'nuc_pref_accuracy' in avg_stats:
        avg_stats['interface_base_pref_Acc'] = avg_stats['nuc_pref_accuracy']
    if 'strong_nuc_pref_accuracy' in avg_stats:
        avg_stats['hotspot_base_pref_Acc'] = avg_stats['strong_nuc_pref_accuracy']
    
    aa_results = {}
    for aa in aa_types:
        aa_stats = aa_metrics_calculators[aa].compute()
        
        aa_contact_tp = aa_stats.get('true_positives', 0)
        aa_contact_tn = aa_stats.get('true_negatives', 0)
        aa_contact_fp = aa_stats.get('false_positives', 0)
        aa_contact_fn = aa_stats.get('false_negatives', 0)
        aa_stats['contact_mcc'] = calculate_mcc(aa_contact_tp, aa_contact_tn, aa_contact_fp, aa_contact_fn)
        
        aa_interface_tp = aa_stats.get('interface_true_positives', 0)
        aa_interface_tn = aa_stats.get('interface_true_negatives', 0)
        aa_interface_fp = aa_stats.get('interface_false_positives', 0)
        aa_interface_fn = aa_stats.get('interface_false_negatives', 0)
        aa_stats['interface_mcc'] = calculate_mcc(aa_interface_tp, aa_interface_tn, aa_interface_fp, aa_interface_fn)
        
        for metric_key in metrics_to_remove:
            aa_stats.pop(metric_key, None)
        
        aa_stats['strong_contact_threshold_value'] = args.strong_contact_threshold
        aa_stats['strong_total_sites'] = aa_strong_contacts[aa]
        aa_stats['strong_correct_interface_sites'] = aa_correct_interface[aa]
        aa_stats['strong_correct_nuc_pref_sites'] = aa_correct_nuc_pref[aa]
        
        if aa_strong_contacts[aa] > 0:
            aa_stats['strong_interface_accuracy'] = aa_correct_interface[aa] / aa_strong_contacts[aa]
            aa_stats['strong_nuc_pref_accuracy'] = aa_correct_nuc_pref[aa] / aa_strong_contacts[aa]
        else:
            aa_stats['strong_interface_accuracy'] = 0.0
            aa_stats['strong_nuc_pref_accuracy'] = 0.0
        
        aa_results[aa] = aa_stats
    
    return avg_stats, per_protein_results, aa_results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DBP2DNA model on a test set')
    
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory (containing PDBs for feature extraction and .npz ground truths)')
    parser.add_argument('--cache_dir', type=str, default='./feature_cache', help='Cache directory for test features')
    parser.add_argument('--complex_cache_dir', type=str, default=None, help='Cache directory for complex structure labels (for apo evaluation mode)')
    parser.add_argument('--matching_report_path', type=str, default=None, help='Path to the matching report file for apo-complex mapping.')
    parser.add_argument('--dna_core_length', type=int, default=8, help='DNA core length (must match training setup for data processing)')

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pt)')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension of the model (loaded from checkpoint by default)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate of the model (loaded from checkpoint by default)')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation (recommend 1 for accurate amino acid statistics)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers (0 for main process)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (if available)')
    parser.add_argument('--seed', type=int, default=9909, help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='logs/eval_logs', help='Directory to save evaluation logs and results')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging progress')
    parser.add_argument('--results_file_name', type=str, default='evaluation_results.json', help='File name to save evaluation metrics as JSON')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for more verbose error reporting on problematic PDBs.')
    parser.add_argument('--protein_list', type=str, default=None, 
                    help='Path to a text file containing list of protein IDs to evaluate (one ID per line)')

    parser.add_argument('--cls_weight', type=float, default=2.0)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--nuc_pref_weight', type=float, default=4.0)
    parser.add_argument('--contact_threshold', type=float, default=0.5, help="Threshold for defining positive contacts in metrics")
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--strong_contact_threshold', type=float, default=0.25, help='Threshold for defining strong contacts for specific metrics')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, f'evaluate_{timestamp}.log')
    logger = setup_logger('evaluate', log_file_path) 
    logger.info("========== Evaluation Configuration ==========")
    for arg, value in vars(args).items():
        logger.info(f"├─ {arg}: {value}")
    logger.info("============================================\n")
    if torch.cuda.is_available() and args.gpu_id >= 0 and args.gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        if torch.cuda.is_available():
            logger.warning(f"GPU id {args.gpu_id} is invalid. Defaulting to cuda:0 or CPU if no GPUs.")
            device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
        else:
            device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    model, _ = load_trained_model(args.checkpoint_path, device, args)
    criterion = ContactHybridLoss(
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        nuc_pref_weight=args.nuc_pref_weight,
        contact_threshold=args.contact_threshold,
        pos_weight=args.pos_weight
    )
    logger.info(f"Using general contact threshold for metrics: {criterion.contact_threshold}")
    logger.info(f"Using strong contact threshold for specific metrics: {args.strong_contact_threshold}")
    
    matching_map = None
    if args.complex_cache_dir and args.matching_report_path:
        matching_map = parse_matching_report(args.matching_report_path)
        logger.info(f"Loaded matching report with {len(matching_map)} entries.")

    if args.complex_cache_dir:
        logger.info("==================== Apo Evaluation Mode ====================")
        logger.info(f"├─ Apo Structure Directory: {args.test_dir}")
        logger.info(f"├─ Complex Label Cache Directory: {args.complex_cache_dir}")
        if matching_map:
            logger.info(f"├─ Using Matching Report: {args.matching_report_path}")
        logger.info("└─ Using apo structures for prediction, complex labels for evaluation")
        logger.info("==================================================\n")
    else:
        logger.info("Using Standard Evaluation Mode")
    
    logger.info(f"Loading test data from: {args.test_dir}")
    
    test_pdb_files = sorted(list(Path(args.test_dir).glob('*.pdb'))) 
    if not test_pdb_files:
        logger.error(f"No PDB files found in {args.test_dir}. Ensure PDBs and their corresponding .npz label files are present.")
        return
    logger.info(f"Found {len(test_pdb_files)} PDB files for evaluation.")
    
    target_proteins = None
    if args.protein_list:
        try:
            with open(args.protein_list, 'r') as f:
                target_proteins = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(target_proteins)} target proteins from {args.protein_list}")
        except Exception as e:
            logger.error(f"Failed to load protein list from {args.protein_list}: {e}")
            return
            
    if target_proteins:
        filtered_pdb_files = []
        for pdb_file in test_pdb_files:
            pdb_id = os.path.splitext(os.path.basename(str(pdb_file)))[0]
            if pdb_id in target_proteins:
                filtered_pdb_files.append(pdb_file)
        test_pdb_files = filtered_pdb_files
        logger.info(f"Filtered to {len(test_pdb_files)} PDB files matching the target list")
        
        if not test_pdb_files:
            logger.error("No PDB files matched the provided protein list. Please check your protein IDs.")
            return
    
    test_dataset = TFDNADataset(
        pdb_paths=test_pdb_files,
        cache_dir=args.cache_dir,
        core_length=args.dna_core_length,
        num_workers=args.num_workers if args.num_workers > 0 else 1, 
        is_training=False,
        build_eval_labels=(args.complex_cache_dir is None)
    )
    
    if len(test_dataset) == 0: 
        logger.error(f"Test dataset is empty after attempting to load {len(test_pdb_files)} PDBs. "
                     f"Check PDB file validity, .npz cache in '{args.cache_dir}', and logs for errors during dataset creation.")
        return
    
    logger.info(f"Test dataset loaded with {len(test_dataset)} valid samples.")
    test_loader = create_test_dataloader(test_dataset, args)
    logger.info("Starting model evaluation on the test set...")
    
    eval_stats, per_protein_results, aa_type_results = evaluate_epoch(model, test_loader, criterion, device, logger, args, test_pdb_files, matching_map)
    
    logger.info("\n========== Overall Test Set Metrics ==========")
    log_stats(logger, eval_stats, "Test Set")
    logger.info("============================================\n")
    
    logger.info("\n========== Amino Acid Type Metrics ==========")
    for aa, stats in aa_type_results.items():
        logger.info(f"Amino Acid: {aa}")
        log_stats(logger, stats, f"   {aa}")
    logger.info("============================================\n")
    
    results_path = os.path.join(args.log_dir, args.results_file_name)
    try:
        json_compatible_stats = {}
        keys_to_keep = [
            'num_protein_samples',
            'num_amino_acid_samples',
            'interface_pr_auc',
            'interface_mcc',
            'interface_base_pref_Acc',
            'hotspot_base_pref_Acc',
            'ndcg_at_5',
            'ndcg_at_10'
        ]
        for k in keys_to_keep:
            if k in eval_stats:
                v = eval_stats[k]
                if isinstance(v, torch.Tensor):
                    json_compatible_stats[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, np.ndarray):
                    json_compatible_stats[k] = v.item() if v.size == 1 else v.tolist()
                elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    json_compatible_stats[k] = v.item() 
                else:
                    json_compatible_stats[k] = v
        with open(results_path, 'w') as f:
            json.dump(json_compatible_stats, f, indent=4)
        logger.info(f"Overall evaluation metrics saved to {results_path}")

        protein_results_path = os.path.join(args.log_dir, 'protein_results.csv')
        if per_protein_results:
            keep_and_rename = {
                'interface_pr_auc': 'interface_pr_auc',
                'interface_mcc': 'interface_mcc',
                'nuc_pref_accuracy': 'interface_base_pref_Acc',
                'strong_nuc_pref_accuracy': 'hotspot_base_pref_Acc',
                'ndcg_at_5': 'ndcg_at_5',
                'ndcg_at_10': 'ndcg_at_10'
            }
            fieldnames = ['pdb_id'] + list(keep_and_rename.values())
            
            with open(protein_results_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for pdb_path in test_pdb_files:
                    pdb_id = os.path.splitext(os.path.basename(str(pdb_path)))[0]
                    if pdb_id in per_protein_results:
                        stats = per_protein_results[pdb_id]
                        row_data = {'pdb_id': pdb_id}
                        for src_key, dst_key in keep_and_rename.items():
                            if src_key in stats:
                                v = stats[src_key]
                                if isinstance(v, torch.Tensor):
                                    row_data[dst_key] = v.item() if v.numel() == 1 else v.tolist()
                                elif isinstance(v, np.ndarray):
                                    row_data[dst_key] = v.item() if v.size == 1 else v.tolist()
                                elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                                    row_data[dst_key] = v.item()
                                else:
                                    row_data[dst_key] = v
                            else:
                                row_data[dst_key] = ''
                        writer.writerow(row_data)
            logger.info(f"Per-protein evaluation metrics saved to {protein_results_path}")
        else:
            logger.info("No per-protein results to save.")

        aa_type_path = os.path.join(args.log_dir, 'aa_type_results.csv')
        if aa_type_results:
            keep_and_rename = {
                'interface_pr_auc': 'interface_pr_auc',
                'interface_mcc': 'interface_mcc',
                'nuc_pref_accuracy': 'interface_base_pref_Acc',
                'strong_nuc_pref_accuracy': 'hotspot_base_pref_Acc',
                'ndcg_at_5': 'ndcg_at_5',
                'ndcg_at_10': 'ndcg_at_10'
            }
            fieldnames = ['amino_acid_type'] + list(keep_and_rename.values())
            
            with open(aa_type_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for aa in sorted(aa_type_results.keys()):
                    stats = aa_type_results[aa]
                    row_data = {'amino_acid_type': aa}
                    for src_key, dst_key in keep_and_rename.items():
                        if src_key in stats:
                            v = stats[src_key]
                            if isinstance(v, torch.Tensor):
                                row_data[dst_key] = v.item() if v.numel() == 1 else v.tolist()
                            elif isinstance(v, np.ndarray):
                                row_data[dst_key] = v.item() if v.size == 1 else v.tolist()
                            elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                                row_data[dst_key] = v.item()
                            else:
                                row_data[dst_key] = v
                        else:
                            row_data[dst_key] = ''
                    writer.writerow(row_data)
            logger.info(f"Amino acid type evaluation metrics saved to {aa_type_path}")
        else:
            logger.info("No amino acid type results to save.")

    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
    logger.info("Evaluation finished.")

if __name__ == '__main__':
    main()
