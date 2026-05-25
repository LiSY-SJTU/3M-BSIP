

import shutup; shutup.please()
import argparse
import logging
import multiprocessing as mp
import os
import random
import signal
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data import TFDNADataset
from src.data import collate_fn as original_collate_fn
from src.data.dynamic_sampler import BucketBatchSampler
from src.metrics.contact_metrics import ContactMetricsCalculator
from src.models import DBP2Predictor
from src.models.loss import ContactHybridLoss
from src.utils.logger import setup_logger

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'



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

class CollateWrapper:
    def __init__(self, core_length):
        self.core_length = core_length

    def __call__(self, batch):
        return original_collate_fn(batch, self.core_length)

def create_dataloaders(train_dataset, valid_dataset, args):

    collate_instance = CollateWrapper(args.dna_core_length)

    batch_sampler = BucketBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        bucket_size=50
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_instance,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instance,
        pin_memory=True
    )

    return train_loader, valid_loader

def log_stats(logger, stats, prefix):
    logger.info(
        f"{prefix}: "
        f"Loss: {stats['loss']:.4f} (Cls: {stats['cls_loss']:.4f}, "
        f"NucPref: {stats['nuc_pref_loss']:.4f}, Rank: {stats['rank_loss']:.4f}) | "
        f"Acc: {stats['accuracy']:.3f} | "
        f"F1: {stats['f1']:.3f} | "
        f"PR-AUC: {stats['pr_auc']:.3f} | "
        f"Nucleotide Preference Accuracy: {stats['nuc_pref_accuracy']:.3f} | "
        f"Interface Accuracy: {stats['interface_acc']:.3f} | "
        f"Interface Recall: {stats['interface_recall']:.3f} | "
        f"Interface PR-AUC: {stats['interface_pr_auc']:.3f} | "
        f"P@5: {stats['precision_at_5']:.3f} R@5: {stats['recall_at_5']:.3f} "
        f"NDCG@5: {stats['ndcg_at_5']:.3f} Spearman: {stats['spearman']:.3f}"
    )

def calculate_classification_stats(pred_binary, target_cls, mask):

    valid_mask = mask.bool()

    true_positives = ((pred_binary == 1) & (target_cls == 1) & valid_mask).sum().item()
    true_negatives = ((pred_binary == 0) & (target_cls == 0) & valid_mask).sum().item()
    false_positives = ((pred_binary == 1) & (target_cls == 0) & valid_mask).sum().item()
    false_negatives = ((pred_binary == 0) & (target_cls == 1) & valid_mask).sum().item()

    total_samples = valid_mask.sum().item()

    return {
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total_samples
    }

def log_classification_stats(logger, stats, prefix=""):

    total = stats['total_samples']
    if total == 0:
        logger.info(f"{prefix} Classification Stats: No valid samples.")
        return

    logger.info(f"{prefix} Classification Stats:")
    logger.info(f"├─ Total Samples: {total:,}")
    logger.info("├─ Correct Classifications:")
    logger.info(f"│  ├─ True Positives (TP): {stats['true_positives']:,} ({stats['true_positives']/total*100:.2f}%)")
    logger.info(f"│  └─ True Negatives (TN): {stats['true_negatives']:,} ({stats['true_negatives']/total*100:.2f}%)")
    logger.info("└─ Incorrect Classifications:")
    logger.info(f"   ├─ False Positives (FP): {stats['false_positives']:,} ({stats['false_positives']/total*100:.2f}%)")
    logger.info(f"   └─ False Negatives (FN): {stats['false_negatives']:,} ({stats['false_negatives']/total*100:.2f}%)")

def log_interface_stats(logger, stats, prefix=""):

    total = (stats['interface_true_positives'] + stats['interface_true_negatives'] +
             stats['interface_false_positives'] + stats['interface_false_negatives'])
    if total == 0:
        logger.info(f"{prefix} Interface Prediction Stats: No valid interface samples.")
        return

    logger.info(f"{prefix} Interface Prediction Stats:")
    logger.info(f"├─ Total Samples: {int(total):,}")
    logger.info("├─ Correct Classifications:")
    logger.info(f"│  ├─ True Positives (TP): {int(stats['interface_true_positives']):,} ({stats['interface_true_positives']/total*100:.2f}%)")
    logger.info(f"│  └─ True Negatives (TN): {int(stats['interface_true_negatives']):,} ({stats['interface_true_negatives']/total*100:.2f}%)")
    logger.info("└─ Incorrect Classifications:")
    logger.info(f"   ├─ False Positives (FP): {int(stats['interface_false_positives']):,} ({stats['interface_false_positives']/total*100:.2f}%)")
    logger.info(f"   └─ False Negatives (FN): {int(stats['interface_false_negatives']):,} ({stats['interface_false_negatives']/total*100:.2f}%)")

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, logger):
    model.train()
    metrics_calculator = ContactMetricsCalculator(contact_threshold=criterion.contact_threshold)

    accumulation_steps = args.gradient_accumulation_steps
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    optimizer.zero_grad()
    batch_losses = []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        try:
            group_start = (batch_idx // accumulation_steps) * accumulation_steps
            group_end = min(group_start + accumulation_steps, total_batches)
            current_accumulation_steps = group_end - group_start

            batch = move_to_device(batch, device)
            predictions = model(batch['protein_features'])
            labels = {
                'contact_tensor': batch['contact_tensor'],
                'protein_lengths': batch['protein_lengths']
            }
            loss_dict = criterion(predictions=predictions, batch=labels)
            loss = loss_dict['loss']
            batch_losses.append(loss.item())
            loss = loss / current_accumulation_steps

            loss.backward()

            if batch_idx + 1 == group_end:
                optimizer.step()
                optimizer.zero_grad()

                if (batch_idx + 1) % args.log_interval == 0:
                    avg_loss = sum(batch_losses[-current_accumulation_steps:]) / current_accumulation_steps
                    logger.info(
                        f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] | "
                        f"Total Loss: {avg_loss:.4f} = Cls({loss_dict['cls_loss']:.4f}) + "
                        f"NucPref({loss_dict['nuc_pref_loss']:.4f}) + "
                        f"Rank({loss_dict['rank_loss']:.4f})"
                    )

            mask = move_to_device(criterion._create_sequence_mask(
                batch['protein_lengths'],
                predictions['contact_cls'].size(1)
            ).unsqueeze(-1).expand(-1, -1, 4), device)

            metrics_calculator.update(
                predictions=predictions,
                targets=labels,
                mask=mask,
                loss_dict=loss_dict,
                batch_size=batch['protein_lengths'].size(0)
            )

        except Exception:
            logger.exception(f"Training failed at epoch {epoch}, batch {batch_idx + 1}")
            raise

    avg_stats = metrics_calculator.compute()

    log_stats(logger, avg_stats, f"Epoch {epoch} Summary")
    log_classification_stats(logger, avg_stats, f"Epoch {epoch}")
    log_interface_stats(logger, avg_stats, f"Epoch {epoch}")

    return avg_stats

def validate(model, dataloader, criterion, device, args, logger, epoch):
    model.eval()
    metrics_calculator = ContactMetricsCalculator(contact_threshold=criterion.contact_threshold)

    with torch.no_grad():
        for batch in dataloader:
            try:
                batch = move_to_device(batch, device)
                predictions = model(batch['protein_features'])
                labels = {
                    'contact_tensor': batch['contact_tensor'],
                    'protein_lengths': batch['protein_lengths']
                }
                loss_dict = criterion(predictions=predictions, batch=labels)

                mask = criterion._create_sequence_mask(
                    batch['protein_lengths'],
                    predictions['contact_cls'].size(1)
                ).unsqueeze(-1).expand(-1, -1, 4).to(device)

                metrics_calculator.update(
                    predictions=predictions,
                    targets=labels,
                    mask=mask,
                    loss_dict=loss_dict,
                    batch_size=batch['protein_lengths'].size(0)
                )

            except Exception:
                logger.exception(f"Validation failed at epoch {epoch}")
                raise

    avg_stats = metrics_calculator.compute()

    logger.info(f"Validation:")
    log_stats(logger, avg_stats, "Validation")

    log_classification_stats(logger, avg_stats, "Validation")
    log_interface_stats(logger, avg_stats, "Validation")

    return avg_stats

def calculate_score(stats):

    cls_weight = 0.8
    rank_weight = 0.2

    f1 = stats.get('f1', 0.0)
    rank_loss = stats.get('rank_loss', 0.0)
    loss = stats.get('loss', 0.0)

    cls_score = 1 - f1
    rank_score = rank_loss
    loss_penalty = 0.1 * loss

    final_score = (cls_weight * cls_score + rank_weight * rank_score + loss_penalty)
    return final_score

def create_model(args, device):
    model = DBP2Predictor(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    return model

def create_loss(args):
    criterion = ContactHybridLoss(
        cls_weight=2.0,
        reg_weight=1.0,
        nuc_pref_weight=4.0,
        contact_threshold=0.5,
        pos_weight=1
    )
    return criterion

def create_optimizer(model, args):
    saprot_param_ids = {id(param) for param in model.saprot.parameters()}
    saprot_params = [param for param in model.saprot.parameters() if param.requires_grad]
    non_saprot_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in saprot_param_ids
    ]

    param_groups = []
    if saprot_params:
        param_groups.append({
            'params': saprot_params,
            'lr': args.lr * 0.1,
            'initial_lr': args.lr * 0.1
        })
    if non_saprot_params:
        param_groups.append({
            'params': non_saprot_params,
            'lr': args.lr,
            'initial_lr': args.lr
        })

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = torch.optim.Adam(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    return optimizer

def train(args):
    set_seed(args.seed)

    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.warning(f"--gpu_id not specified, defaulting to cuda:0")
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        global TARGET_GPU_ID
        TARGET_GPU_ID = device.index

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger('train', os.path.join(args.log_dir, f'train_{timestamp}.log'))

    logger.info("========== Training Configuration ==========")
    logger.info("Data Configuration:")
    logger.info(f"├─ Batch Size: {args.batch_size}")
    logger.info(f"├─ Number of Workers: {args.num_workers}")
    logger.info(f"├─ Gradient Accumulation Steps: {args.gradient_accumulation_steps}")

    logger.info("\nModel Configuration:")
    logger.info(f"├─ Hidden Dimension: {args.hidden_dim}")
    logger.info(f"├─ Dropout Rate: {args.dropout}")
    logger.info(f"└─ Freeze SaProt: {args.freeze_saprot}")

    logger.info("\nTraining Configuration:")
    logger.info(f"├─ Number of Epochs: {args.epochs}")
    logger.info(f"├─ Learning Rate: {args.lr}")
    logger.info(f"├─ Warmup Epochs: {args.warmup_epochs}")
    logger.info(f"├─ Log Interval: {args.log_interval}")
    logger.info(f"├─ Early Stopping Patience: {args.early_stopping_patience}")
    logger.info(f"├─ Random Seed: {args.seed}")
    logger.info(f"└─ Device: {device}")
    logger.info("============================\n")

    logger.info(f'Using device: {device}')

    train_pdb_files = list(Path(args.train_dir).glob('*.pdb'))
    valid_pdb_files = list(Path(args.valid_dir).glob('*.pdb'))

    logger.info(f'Found {len(train_pdb_files)} training files')
    logger.info(f'Found {len(valid_pdb_files)} validation files')

    train_dataset = TFDNADataset(
        pdb_paths=train_pdb_files,
        cache_dir='./feature_cache',
        core_length=args.dna_core_length,
        num_workers=args.num_workers
    )

    valid_dataset = TFDNADataset(
        pdb_paths=valid_pdb_files,
        cache_dir='./feature_cache',
        core_length=args.dna_core_length,
        num_workers=args.num_workers
    )

    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = create_model(args, device)

    if args.freeze_saprot:
        for param in model.saprot.parameters():
            param.requires_grad = False

    criterion = create_loss(args).to(device)
    optimizer = create_optimizer(model, args)

    logger.info("\nLoss Function Configuration:")
    logger.info(f"├─ Classification Loss Weight: {criterion.cls_weight}")
    logger.info(f"├─ Ranking Loss Weight: {criterion.reg_weight}")
    logger.info(f"├─ Nucleotide Preference Loss Weight: {criterion.nuc_pref_weight}")
    logger.info(f"├─ Contact Threshold: {criterion.contact_threshold}")
    logger.info(f"└─ Positive Sample Weight: {criterion.pos_weight.item()}")

    logger.info("\nOptimizer Configuration:")
    logger.info(f"├─ Type: Adam")
    logger.info(f"├─ Base Learning Rate: {args.lr}")
    if args.freeze_saprot:
        logger.info(f"├─ SaProt: Frozen (not participating in training)")
    else:
        logger.info(f"├─ SaProt Learning Rate: {args.lr * 0.1}")
    logger.info(f"├─ Beta1: 0.9")
    logger.info(f"├─ Beta2: 0.999")
    logger.info(f"├─ Epsilon: 1e-8")
    logger.info(f"└─ Weight Decay: 1e-5")
    logger.info("============================\n")


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    best_score = float('inf')
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args, logger
        )

        valid_stats = validate(
            model, valid_loader, criterion, device, args, logger, epoch
        )

        if epoch < args.warmup_epochs:
            warmup_scale = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_scale
        else:
            scheduler.step(valid_stats['loss'])

        try:
            current_score = calculate_score(valid_stats)
            if current_score < best_score:
                best_score = current_score
                no_improve_epochs = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_score': best_score,
                    'metrics': valid_stats,
                    'args': args
                }
                best_model_path = os.path.join(args.save_dir, f'best_model_{timestamp}.pt')
                torch.save(checkpoint, best_model_path)
                logger.info(f"Saved Best Model to {best_model_path}: Score: {best_score:.4f}")
                log_stats(logger, valid_stats, "Best Validation Stats")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        except Exception as e:
            logger.error(f"Error during score calculation or checkpoint saving: {str(e)}", exc_info=True)
            continue


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3M-BSIP Model')
    parser.add_argument('--train_dir', type=str, default='./dataset/train', help='Training data directory')
    parser.add_argument('--valid_dir', type=str, default='./dataset/valid', help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dna_core_length', type=int, default=8, help='DNA core region length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs/training_logs', help='Log save directory')
    parser.add_argument('--freeze_saprot', action='store_true', help='Whether to freeze SaProt pre-trained model parameters')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='Early stopping patience')



    return parser.parse_args()

def main():
    args = parse_args()
    if args.num_workers > 0:
        try:
            current_start_method = mp.get_start_method(allow_none=True)
            if current_start_method != 'spawn':
                mp.set_start_method('spawn', force=True)
                print(f"Multiprocessing start method set to 'spawn'. Previous method was: {current_start_method}")
            else:
                print("Multiprocessing start method already set to 'spawn'.")
        except RuntimeError as e:
            print(f"Warning: Could not set start method to 'spawn': {e}. Using current method: {mp.get_start_method(allow_none=True)}")

    train(args)

if __name__ == '__main__':
    main()
