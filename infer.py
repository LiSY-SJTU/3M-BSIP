import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.direct_processor import DirectProcessor
from src.models import DBP2Predictor
from src.utils.logger import setup_logger


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def move_to_device(batch, device):
    
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    elif hasattr(batch, 'to'):
        return batch.to(device)
    return batch

def load_model(checkpoint_path, device, hidden_dim):
    
    model = DBP2Predictor(
        hidden_dim=hidden_dim,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model

 
def _squeeze_batch_dim(tensor):
    """Remove batch dimension (dim 0) if it exists and has size 1."""
    if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def plot_contact_specificity(contact_cls, nuc_preference, protein_sequence, save_path, logger, threshold=0.5, residue_range=None, binding_importance=None):
    
    plt.rcParams['font.family'] = 'Arial'

    model_nt_order = ['A', 'T', 'G', 'C']
    display_nt_order = ['A', 'T', 'C', 'G']
    nt_colors = {
        'A': '#58976b',
        'T': '#f17463',
        'G': '#fdbd71',
        'C': '#707db9'
    }
    # Map model nucleotide channels to display rows.
    model_index_to_row = [display_nt_order.index(nt) for nt in model_nt_order]

    contact_cls = _squeeze_batch_dim(contact_cls)
    nuc_preference = _squeeze_batch_dim(nuc_preference)
    if binding_importance is not None:
        binding_importance = _squeeze_batch_dim(binding_importance)
    
    contact_probs = torch.sigmoid(contact_cls)
    logger.info(f"Classification probabilities range: {contact_probs.min().item():.4f} to {contact_probs.max().item():.4f}")
    
    index_offset = 0
    protein_sequence_local = protein_sequence
    if residue_range is not None:
        parts = residue_range.strip().split(':')
        start = int(parts[0])
        end = int(parts[1])
        start0 = start - 1
        end0 = end
        contact_cls = contact_cls[start0:end0]
        nuc_preference = nuc_preference[start0:end0]
        contact_probs = contact_probs[start0:end0]
        if binding_importance is not None:
            binding_importance = binding_importance[start0:end0]
        protein_sequence_local = protein_sequence[start0:end0]
        index_offset = start0

    nuc_pref_probs = torch.softmax(nuc_preference, dim=-1)
    logger.info(f"Nucleotide preference probabilities range: {nuc_pref_probs.min().item():.4f} to {nuc_pref_probs.max().item():.4f}")

    aa_contact_mask = (contact_probs > threshold).any(dim=1)
    logger.info(f"Found {aa_contact_mask.sum().item()} amino acids with any contact probability > {threshold}")
    
    num_residues = nuc_pref_probs.size(0)

    global_max_pref = torch.max(nuc_pref_probs).item()
    if global_max_pref == 0:
        global_max_pref = 1.0

    # Fold long non-interface spans while preserving terminal residues.
    display_items = []  # int residue index or 'ELLIPSIS'
    i = 0
    while i < num_residues:
        if not aa_contact_mask[i]:
            j = i
            while j < num_residues and not aa_contact_mask[j]:
                j += 1
            run_len = j - i
            is_leading = (i == 0)
            is_trailing = (j == num_residues)
            if run_len > 8:
                if is_leading and is_trailing:
                    display_items.append(0)
                    if num_residues > 1:
                        display_items.append('ELLIPSIS')
                        display_items.append(num_residues - 1)
                    break
                if is_leading:
                    display_items.append(0)
                    display_items.append('ELLIPSIS')
                    i = j
                    continue
                if is_trailing:
                    display_items.append('ELLIPSIS')
                    display_items.append(num_residues - 1)
                    break
                display_items.append('ELLIPSIS')
                i = j
                continue
            else:
                for k in range(i, j):
                    display_items.append(k)
                i = j
                continue
        else:
            display_items.append(i)
            i += 1

    num_cols = len(display_items)
    logger.info(f"Compressed to {num_cols} columns with ellipsis folding")

    img = np.ones((4, num_cols, 4), dtype=float)
    img[:, :, :3] = 1.0  # White background
    img[:, :, 3] = 1.0

    def hex_to_rgb01(h):
        h = h.lstrip('#')
        return (int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0)

    max_pref_values, max_pref_indices = torch.max(nuc_pref_probs, dim=1)
    # Use binding-importance rank to scale interface opacity.
    alpha_map = None
    if binding_importance is not None and bool(aa_contact_mask.any()):
        imp = binding_importance.float()
        alpha_map = torch.zeros_like(imp, dtype=torch.float32)
        valid_idx = aa_contact_mask.nonzero(as_tuple=False).squeeze(-1)
        valid_imp = imp[valid_idx]
        n_valid = int(valid_imp.numel())
        order_desc = torch.argsort(valid_imp, descending=True)
        ranks = torch.empty(n_valid, dtype=torch.float32, device=valid_imp.device)
        ranks[order_desc] = torch.arange(1, n_valid + 1, dtype=torch.float32, device=valid_imp.device)
        alpha_vals = torch.clamp(1.0 - 0.07 * (ranks - 1.0), min=0.3, max=1.0)
        alpha_map[valid_idx] = alpha_vals
    for col_idx, item in enumerate(display_items):
        if isinstance(item, int):
            res_idx = item
            if not bool(aa_contact_mask[res_idx]):
                continue
            model_nt_idx = int(max_pref_indices[res_idx].item())
            disp_row = model_index_to_row[model_nt_idx]
            if alpha_map is not None:
                alpha = float(alpha_map[res_idx].item())
            else:
                alpha = float(max_pref_values[res_idx].item() / max(1e-9, global_max_pref))
            nt_char = model_nt_order[model_nt_idx]
            r, g, b = hex_to_rgb01(nt_colors[nt_char])
            img[disp_row, col_idx, 0] = r
            img[disp_row, col_idx, 1] = g
            img[disp_row, col_idx, 2] = b
            img[disp_row, col_idx, 3] = alpha

    fig_w = max(4.0, min(24.0, num_cols * 0.2 + 2.0))
    fig = plt.figure(figsize=(fig_w, 2.2))
    ax = fig.add_axes([0.08, 0.34, 0.90, 0.56])

    ax.imshow(img, aspect='auto', interpolation='nearest', origin='upper')

    for y in range(5):
        ax.axhline(y=y-0.5, color='#A0A0A0', linewidth=1.5)
    for x in range(num_cols + 1):
        ax.axvline(x=x-0.5, color='#A0A0A0', linewidth=1.5)

    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_edgecolor('#A0A0A0')
        ax.spines[side].set_linewidth(1.5)

    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(display_nt_order)
    for lbl in ax.get_yticklabels():
        nt = lbl.get_text()
        if nt in nt_colors:
            lbl.set_color(nt_colors[nt])
        lbl.set_fontweight('bold')
        lbl.set_fontsize(14)
    ax.tick_params(axis='y', which='major', pad=5)

    # Label residues and folded spans on the top axis.
    ax.set_xticks(np.arange(num_cols))
    xticklabels = []
    tick_colors = []
    column_is_residue = []
    column_global_pos = []
    for item in display_items:
        if isinstance(item, int):
            aa = protein_sequence_local[item]
            xticklabels.append(aa)
            tick_colors.append('black' if bool(aa_contact_mask[item]) else 'gray')
            column_is_residue.append(True)
            column_global_pos.append(item + 1 + index_offset)
        else:
            xticklabels.append('...')
            tick_colors.append('gray')
            column_is_residue.append(False)
            column_global_pos.append(None)
    ax.set_xticklabels(xticklabels)
    ax.xaxis.set_ticks_position('top')
    for lbl, c in zip(ax.get_xticklabels(), tick_colors):
        lbl.set_color(c)
        lbl.set_fontsize(12)

    # Add residue-number anchors after folded spans.
    anchor = None
    for col_idx in range(num_cols):
        if not column_is_residue[col_idx]:
            anchor = None
            continue
        gpos = column_global_pos[col_idx]
        if anchor is None:
            anchor = gpos
            ax.text(col_idx, 1.18, f"{gpos}", ha='center', va='bottom', fontsize=6,
                    color='black', transform=ax.get_xaxis_transform(), clip_on=False)
        else:
            if (gpos - anchor) % 5 == 0:
                ax.text(col_idx, 1.18, f"{gpos}", ha='center', va='bottom', fontsize=6,
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False)

    # Highlight top binding-importance residues.
    if binding_importance is not None:
        candidates = [(float(binding_importance[i].item()), i) for i in range(num_residues) if bool(aa_contact_mask[i])]
        candidates.sort(key=lambda x: x[0], reverse=True)
        top5 = [idx for _, idx in candidates[:5]]
        residue_to_col = {res_idx: col_idx for col_idx, res_idx in enumerate(display_items) if isinstance(res_idx, int)}
        for rank, res_idx in enumerate(top5):
            if res_idx not in residue_to_col:
                continue
            col_idx = residue_to_col[res_idx]
            model_nt_idx = int(max_pref_indices[res_idx].item())
            row = model_index_to_row[model_nt_idx]
            highlight = plt.Rectangle((col_idx-0.5, row-0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
            ax.add_patch(highlight)
            ax.text(col_idx, row, f"{rank+1}", ha='center', va='center', color='#f8f9fa', fontsize=8, weight='bold')

    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='both', length=0)
    
    logger.info(f"Saving image to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Image saved successfully: {save_path}")

def extract_protein_sequence(batch, logger):
    
    if 'protein_features' in batch and 'sequence' in batch['protein_features'] and \
       'sequence' in batch['protein_features']['sequence']:
        return batch['protein_features']['sequence']['sequence']
    
    protein_length = batch['protein_lengths'].item()
    logger.warning(f"Using placeholder sequence, protein length: {protein_length}")
    return ''.join(['X' for _ in range(protein_length)])

def save_results(results, output_dir, logger, residue_range=None):
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    
    for result in results:
        pdb_id = result['pdb_id']
        logger.info(f"Processing results for {pdb_id}")
        
        pdb_output_dir = output_dir_path / pdb_id
        pdb_output_dir.mkdir(exist_ok=True)
        
        output_path = pdb_output_dir / f"{pdb_id}_pred.pt"
        save_dict = {
            'protein_length': result['protein_lengths']
        }
        
        if 'contact_cls' in result:
            save_dict['contact_cls'] = result['contact_cls']
        if 'nuc_preference' in result:
            save_dict['nuc_preference'] = result['nuc_preference']
        if 'residue_feat' in result:
            save_dict['residue_feat'] = result['residue_feat']
        if 'protein_sequence' in result:
            save_dict['protein_sequence'] = result['protein_sequence']
        
        torch.save(save_dict, output_path)
        logger.info(f"Saving results to: {output_path}")
        
        if all(k in result for k in ['contact_cls', 'nuc_preference']):
            protein_length = result['protein_lengths'].item()
            contact_cls = result['contact_cls']
            nuc_preference = result['nuc_preference']
            binding_importance = result.get('binding_importance', None)
            
            protein_sequence = result.get('protein_sequence', 
                                        ''.join(['X' for _ in range(protein_length)]))
            
            viz_path = pdb_output_dir / f"{pdb_id}_contact_specificity.png"

            logger.info(f"Starting to plot contact graph...")
            plot_contact_specificity(
                contact_cls,
                nuc_preference,
                protein_sequence,
                viz_path,
                logger,
                threshold=0.28,
                residue_range=residue_range,
                binding_importance=binding_importance
            )
        else:
            logger.warning("Missing required predictions for visualization")
            missing = [k for k in ['contact_cls', 'nuc_preference'] if k not in result]
            logger.warning(f"Missing components: {missing}")


def parse_args():
    
    parser = argparse.ArgumentParser(description='Single-GPU Inference for DBP2DNA Model - Optimized Version')
    
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension')
    parser.add_argument('--dna_core_length', type=int, default=8,
                      help='DNA core region length')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Trained model checkpoint path')
    parser.add_argument('--input_dir', type=str, default='test_case',
                      help='Input data directory (containing PDB files)')
    parser.add_argument('--input_pdb', type=str, default=None,
                      help='Single PDB file path for inference (takes precedence over --input_dir)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Inference results save directory')
    parser.add_argument('--cache_dir', type=str, default='./feature_cache/infer_feature_cache',
                      help='Feature cache directory')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of data loading workers')
    parser.add_argument('--gpu_id', type=int, default=0,
                      help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=3407,
                      help='Random seed')
    parser.add_argument('--residue_range', type=str, default=None,
                      help='Plot residue range, format start:end (1-based, inclusive interval)')
    
    return parser.parse_args()


def infer_single_pdb(pdb_file_path, model, device, processor, logger):
    
    logger.info(f"Starting inference: {pdb_file_path}")
    
    sample = processor.process_pdb(pdb_file_path)
    if sample is None:
        logger.warning(f"Skipping invalid file: {pdb_file_path}")
        return None
    
    pdb_id = sample['pdb_id']
    
    batch = {
        'protein_features': sample['protein_features'],
        'protein_lengths': torch.tensor([sample['contact_tensor'].size(0)]),
        'pdb_ids': [pdb_id]
    }
    
    batch = move_to_device(batch, device)
    
    with torch.no_grad():
        outputs = model(batch['protein_features'])
    
    result = {
        'protein_lengths': batch['protein_lengths'].cpu(),
        'pdb_id': pdb_id
    }
    
    protein_sequence = extract_protein_sequence(batch, logger)
    result['protein_sequence'] = protein_sequence

    if 'contact_cls' in outputs:
        result['contact_cls'] = outputs['contact_cls'].cpu()
    if 'nuc_preference' in outputs:
        result['nuc_preference'] = outputs['nuc_preference'].cpu()
    if 'binding_importance' in outputs:
        result['binding_importance'] = outputs['binding_importance'].cpu()
    if 'residue_feat' in outputs:
        result['residue_feat'] = outputs['residue_feat'].cpu()
    
    logger.info(f"Finished inference: {pdb_id}")
    return result
        

def create_inference_components(checkpoint_path, cache_dir, dna_core_length, device, hidden_dim):
    
    model = load_model(checkpoint_path, device, hidden_dim)
    
    processor = DirectProcessor(
        cache_dir=cache_dir,
        core_length=dna_core_length
    )
    
    return model, processor

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(exist_ok=True)
    log_file = output_dir_path / f'infer_{timestamp}.log'
    logger = setup_logger('infer', log_file)
    
    logger.info("======== Inference Configuration ========")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"Number of Workers: {args.num_workers}")
    if args.input_pdb:
        logger.info(f"Input PDB File: {args.input_pdb}")
    else:
        logger.info(f"Input Directory: {args.input_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Model Checkpoint: {args.checkpoint_path}")
    logger.info("=========================")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    logger.info(f'Loading model from {args.checkpoint_path}')
    model, processor = create_inference_components(
        args.checkpoint_path, args.cache_dir, args.dna_core_length, device, args.hidden_dim
    )
    
    if args.input_pdb:
        input_pdb_path = Path(args.input_pdb)
        if not input_pdb_path.exists():
            logger.error(f"PDB file not found: {args.input_pdb}")
            return
        if not input_pdb_path.is_file():
            logger.error(f"Path is not a file: {args.input_pdb}")
            return
        input_files = [input_pdb_path]
        logger.info(f'Processing single PDB file: {args.input_pdb}')
    else:
        logger.info(f'Loading input data from {args.input_dir}')
        input_files = list(Path(args.input_dir).glob('*.pdb'))
        logger.info(f'Found {len(input_files)} PDB files')
    
    results = []
    for i, pdb_file in enumerate(input_files):
        logger.info(f"Processing file {i+1}/{len(input_files)}: {pdb_file.name}")
        
        result = infer_single_pdb(pdb_file, model, device, processor, logger)
        if result is None:
            logger.warning(f"Skipping invalid file: {pdb_file}")
            continue
        
        results.append(result)
        logger.info(f"Finished processing: {pdb_file.name}")
    
    logger.info(f'Saving results and visualizations to {args.output_dir}')
    save_results(results, args.output_dir, logger, residue_range=args.residue_range)
    
    logger.info('Inference completed!')

if __name__ == '__main__':
    main()
