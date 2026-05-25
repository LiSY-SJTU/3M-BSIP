# 3M-BSIP

3M-BSIP is a multimodal deep learning framework for DNA-binding protein specificity prediction. Given a protein structure, the model predicts residue-level DNA-binding interface, nucleotide preference, and residue-importance scores.



## Repository Structure

```text
.
|-- train_amp.py                  # Training entry point
|-- evaluate.py                   # Evaluation entry point
|-- infer.py                      # Single-structure inference and visualization
|-- src/
|   |-- data/                     # Dataset processing and feature extraction
|   |-- metrics/                  # Evaluation metrics
|   |-- models/                   # Model, encoders, losses, and modules
|   `-- utils/                    # Logging utilities
|-- models/
|   `-- saprot/                   # Local SaProt model files

```

## Environment

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate 3M-BSIP
```

This repository depends on SaProt for structure-aware protein sequence representation:

```text
https://github.com/westlake-repl/SaProt
```

Place the local SaProt model files under `models/saprot/`.

## Pre-trained Weights

The pre-trained weight file `best_model.pt` is available at:

```text
https://pan.sjtu.edu.cn/web/share/6305a7ef5e9c698525ec3e316590300a
```

## Training

```bash
python train_amp.py \
  --train_dir Dataset/train \
  --valid_dir Dataset/valid \
  --save_dir checkpoints \
  --log_dir logs/training_logs \
  --gpu_id 0
```

Useful options:

- `--batch_size`: training batch size.
- `--gradient_accumulation_steps`: gradient accumulation steps.
- `--epochs`: number of training epochs.
- `--hidden_dim`: model hidden dimension.
- `--freeze_saprot`: freeze SaProt parameters.

## Evaluation

```bash
python evaluate.py \
  --test_dir Dataset/test \
  --checkpoint_path checkpoints/best_model.pt \
  --cache_dir feature_cache \
```

## Inference

```bash
python infer.py \
  --input_pdb path/to/input.pdb \
  --checkpoint_path checkpoints/best_model.pt \
  --output_dir results \
  --cache_dir feature_cache/infer_feature_cache \
```

Outputs include prediction tensors and contact-specificity visualizations.
