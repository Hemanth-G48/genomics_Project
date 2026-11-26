# Genomics Project — Exon/Intron Segmentation with a CNN-BiLSTM

Per-base prediction of exon vs. intron structure along human transcripts using a
convolutional–recurrent network trained on GRCh38 / GENCODE v50 annotations.

## Overview

The pipeline converts annotated human transcripts into overlapping windows, encodes each
base with 87 biological features, and trains a CNN-BiLSTM that labels every position as
exon (class 0), intron (class 1), or padding (class 2, excluded from metrics).

**Working set:** chromosome 22, protein-coding transcripts (GENCODE v50).

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_prepare_transcripts.py` | Extract chr22 transcript sequences and exon/intron labels from FASTA + GTF |
| 2 | `02_generate_windows.py` | Split transcripts into 2048 bp windows with 1024 bp stride |
| 3 | `03_split_dataset.py` | Gene-level train/validation/test split (prevents transcript leakage) |
| 4 | `04_extract_features.py` | Encode sequence + biological features into an HDF5 dataset |
| 5 | `05_train_model.py` | Train the feature-enhanced CNN-BiLSTM (cross-entropy) |
| 5 | `05_train_model_focal.py` | Same architecture, focal loss for exon-class imbalance |
| 6 | `06_evaluate_model.py` | Select exon threshold on validation, evaluate test once |
| 7 | `07_error_analysis.py` | Validation-only error analysis by exon fraction |
| 8 | `08_compare_checkpoints.py` | Compare baseline vs. focal checkpoints on validation |
| 8 | `08_compare_checkpoints_v2.py` | Same, with a sorted-threshold sweep (equivalent, much faster) |

Run every script from the project root, e.g.:

```bash
python Scripts/01_prepare_transcripts.py
python Scripts/08_compare_checkpoints_v2.py
```

## Features (87 channels per base)

| Channels | Feature |
|---|---|
| 5 | DNA one-hot (A, C, G, T, N) |
| 2 | GC content and GC skew (per-base) |
| 16 | Dinucleotide frequency channels |
| 64 | Trinucleotide frequency channels |

## Model

```
Input (87 × 2048)
  → Conv1d(87 → 32, kernel 9, same padding) → ReLU
  → BiLSTM(32 → 32, bidirectional) → 64
  → Linear(64 → 2)   # exon / intron per base
```

Training: Adam, 12M windows (2048 bp, 1024 bp stride), early stopping on validation macro-F1.

## Results

Evaluated on the untouched test partition (119,476,696 valid bases) at a threshold of
0.75 selected on validation only.

| Metric | Test |
|---|---|
| Accuracy | 0.9262 |
| Macro F1 | 0.7165 |
| Exon precision | 0.4236 |
| Exon recall | 0.5346 |
| Exon F1 | 0.4727 |
| Intron F1 | 0.9603 |

Baseline vs. focal-loss checkpoints (validation only, 58,360 windows):

| Model | Threshold | Accuracy | Macro F1 | Exon F1 | Intron F1 |
|---|---|---|---|---|---|
| Baseline (cross-entropy) | 0.75 | 0.9365 | 0.7525 | 0.5390 | 0.9659 |
| Focal loss | 0.60 | 0.9354 | 0.7473 | 0.5292 | 0.9653 |

The baseline model outperforms the focal-loss variant on every metric. Exon prediction
remains the hard part: exons are a small minority of bases, so recall is limited even
though intron classification is near-perfect.

## Repository contents

- `Scripts/` — full pipeline (stages 1–8)
- `models/` — trained checkpoints (`*_best.pt`)
- `results/` — metrics, threshold sweeps, error analysis, training history
- `prepared_dataset/`, `partitioned_dataset/`, `window_dataset/` — metadata CSVs (large sequence archives excluded)

## Data not included

The genome FASTA (3.0 GB), GENCODE GTF (4.4 GB), `feature_dataset/features.h5` (14 GB)
and the window sequence archives are excluded — they exceed GitHub's file limits. To
regenerate them, run stages 1–4 against the public inputs:

- **Genome:** GRCh38 primary assembly (GENCODE / Ensembl)
- **Annotation:** GENCODE v50 primary assembly GTF

## Environment

Python 3.12 with `torch 2.15.0`, `numpy 2.5.4`, `h5py 3.16.0`, `scipy 1.18.1`,
`scikit-learn 1.9.1`. CUDA 13.2 build; runs on CPU but training expects a CUDA GPU.
