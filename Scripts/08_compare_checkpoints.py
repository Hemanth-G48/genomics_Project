#!/usr/bin/env python3
"""
Compare the baseline and focal-loss CNN-BiLSTM checkpoints on the VALIDATION
partition only. Select the best exon-probability threshold for each model.

Expected HDF5 layout (as produced by the project feature-extraction script):
  sequence       [N, L]       encoded DNA
  composition    [N, L, 2]    GC content and GC skew, uint8
  kmer           [N, L, 80]   16 dinucleotide + 64 trinucleotide channels
  labels         [N, L]       E=0, I=1, X=2 (padding)
  metadata/Split [N]          "Training", "Validation", "Test"

Run from the project root:
  python Scripts/08_compare_checkpoints.py

Or specify paths:
  python Scripts/08_compare_checkpoints.py --h5 feature_dataset/features.h5 \
      --baseline models/cnn_bilstm_features_best.pt \
      --focal models/cnn_bilstm_focal_best.pt

This script does NOT read or evaluate the Test partition.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class ValidationDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        self.h5 = h5py.File(self.h5_path, "r")
        required = ["sequence", "composition", "kmer", "labels", "metadata/Split"]
        missing = [key for key in required if key not in self.h5]
        if missing:
            raise KeyError(
                f"Missing HDF5 datasets: {missing}. "
                "Check the feature_dataset/features.h5 layout."
            )

        split = self.h5["metadata/Split"][:]
        split = np.array([
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in split
        ])
        self.indices = np.flatnonzero(np.char.lower(split) == "validation")
        if len(self.indices) == 0:
            raise RuntimeError(
                "No rows marked 'Validation' in metadata/Split. "
                "Inspect the actual split labels before proceeding."
            )

        self.seq = self.h5["sequence"]
        self.comp = self.h5["composition"]
        self.kmer = self.h5["kmer"]
        self.labels = self.h5["labels"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        seq = np.asarray(self.seq[i], dtype=np.int64)
        comp = np.asarray(self.comp[i], dtype=np.float32)
        kmer = np.asarray(self.kmer[i], dtype=np.float32)
        labels = np.asarray(self.labels[i], dtype=np.int64)

        # Match training preprocessing: 5 DNA one-hot channels.
        # The sequence encoding is expected to be A=0,C=1,G=2,T=3,N=4.
        seq = np.clip(seq, 0, 4)
        onehot = np.eye(5, dtype=np.float32)[seq]

        # Match training scaling for engineered channels.
        comp[:, 0] /= 255.0                 # GC content: [0,1]
        comp[:, 1] = 2.0 * (comp[:, 1] / 255.0) - 1.0  # GC skew: [-1,1]
        kmer /= 255.0

        x = np.concatenate([onehot, comp, kmer], axis=1).T
        return torch.from_numpy(x), torch.from_numpy(labels)


class CNNBiLSTM(nn.Module):
    def __init__(self, input_channels=87, cnn_filters=32, kernel_size=9,
                 bilstm_hidden_size=32, num_classes=2, bidirectional=True):
        super().__init__()
        self.cnn = nn.Conv1d(
            input_channels, cnn_filters, kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=bilstm_hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out = bilstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_out, num_classes)

    def forward(self, x):
        x = self.relu(self.cnn(x))
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        return self.classifier(x)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("model_config", {})
    model = CNNBiLSTM(
        input_channels=cfg.get("input_channels", 87),
        cnn_filters=cfg.get("cnn_filters", 32),
        kernel_size=cfg.get("cnn_kernel_size", cfg.get("kernel_size", 9)),
        bilstm_hidden_size=cfg.get("bilstm_hidden_size", 32),
        num_classes=cfg.get("num_classes", 2),
        bidirectional=cfg.get("bidirectional", True),
    )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(
            f"{checkpoint_path}: checkpoint has no 'model_state_dict'. "
            "Please send the checkpoint and training script so the loader can be adapted."
        )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, loader, device):
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[..., 0]  # exon probability, class E=0
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

    probs = np.concatenate([p.reshape(-1) for p in all_probs])
    labels = np.concatenate([y.reshape(-1) for y in all_labels])
    valid = labels != 2  # exclude padding X
    return probs[valid], labels[valid]


def threshold_metrics(probs, labels, threshold):
    # At/above the exon probability threshold, predict exon (class 0).
    pred = np.where(probs >= threshold, 0, 1).astype(np.int64)  # 0=Exon, 1=Intron
    # The binary target is 0 for exon, 1 for intron.
    # Explicit labels keep the class ordering stable.
    macro = f1_score(labels, pred, labels=[0, 1], average="macro", zero_division=0)
    exon_p = precision_score(labels, pred, pos_label=0, zero_division=0)
    exon_r = recall_score(labels, pred, pos_label=0, zero_division=0)
    exon_f = f1_score(labels, pred, labels=[0], average="macro", zero_division=0)
    intron_f = f1_score(labels, pred, labels=[1], average="macro", zero_division=0)
    acc = accuracy_score(labels, pred)
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "exon_precision": float(exon_p),
        "exon_recall": float(exon_r),
        "exon_f1": float(exon_f),
        "intron_f1": float(intron_f),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="feature_dataset/features.h5")
    parser.add_argument("--baseline", default="models/cnn_bilstm_features_best.pt")
    parser.add_argument("--focal", default="models/cnn_bilstm_focal_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/model_comparison")
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    for p in [args.h5, args.baseline, args.focal]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = ValidationDataset(args.h5)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    print(f"Validation windows: {len(dataset):,}")
    print("Test partition: NOT LOADED / NOT EVALUATED")

    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2,
                  args.threshold_step), 4
    )
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    sweep_rows = []

    for name, path in [("baseline", args.baseline), ("focal", args.focal)]:
        print(f"\nLoading {name}: {path}")
        model, checkpoint = load_model(path, device)
        probs, labels = predict(model, loader, device)

        best = None
        for threshold in thresholds:
            m = threshold_metrics(probs, labels, threshold)
            m["model"] = name
            sweep_rows.append(dict(m))
            if best is None or m["macro_f1"] > best["macro_f1"]:
                best = m

        best["checkpoint"] = str(path)
        best["checkpoint_epoch"] = checkpoint.get("epoch", "unknown")
        best["checkpoint_recorded_val_macro_f1"] = checkpoint.get(
            "best_validation_macro_f1", "unknown"
        )
        summary.append(best)
        print(
            f"{name}: best threshold={best['threshold']:.2f}, "
            f"macro F1={best['macro_f1']:.4f}, "
            f"exon F1={best['exon_f1']:.4f}, "
            f"exon precision={best['exon_precision']:.4f}, "
            f"exon recall={best['exon_recall']:.4f}"
        )
        del model

    with open(outdir / "validation_threshold_comparison.csv", "w", newline="") as f:
        fields = ["model", "threshold", "accuracy", "macro_f1",
                  "exon_precision", "exon_recall", "exon_f1", "intron_f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sweep_rows)

    with open(outdir / "validation_model_comparison.csv", "w", newline="") as f:
        fields = ["model", "checkpoint", "checkpoint_epoch",
                  "checkpoint_recorded_val_macro_f1", "threshold", "accuracy",
                  "macro_f1", "exon_precision", "exon_recall", "exon_f1", "intron_f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    with open(outdir / "validation_model_comparison.txt", "w") as f:
        f.write("VALIDATION-ONLY MODEL COMPARISON\n")
        f.write("The Test partition was not loaded or evaluated.\n\n")
        for row in summary:
            f.write(f"{row['model'].upper()}\n")
            for key in ["checkpoint", "checkpoint_epoch",
                        "checkpoint_recorded_val_macro_f1", "threshold",
                        "accuracy", "macro_f1", "exon_precision",
                        "exon_recall", "exon_f1", "intron_f1"]:
                f.write(f"  {key}: {row[key]}\n")
            f.write("\n")

    print(f"\nSaved results in: {outdir}")
    print("Send these files back:")
    print("  validation_model_comparison.csv")
    print("  validation_threshold_comparison.csv")
    print("  validation_model_comparison.txt")


if __name__ == "__main__":
    main()
