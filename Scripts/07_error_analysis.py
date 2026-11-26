#!/usr/bin/env python3
"""
Stage 7: Validation-only error analysis for the trained CNN-BiLSTM.

This script does NOT train or modify the model and does NOT read the test split.
It loads the Stage 5 checkpoint, evaluates Validation windows, selects a
threshold using Validation only, and reports where exon errors occur.

Inputs:
    feature_dataset/features.h5
    models/cnn_bilstm_features_best.pt

Outputs:
    results/error_analysis/validation_threshold_sweep.csv
    results/error_analysis/validation_error_summary.txt
    results/error_analysis/validation_window_errors.csv
    results/error_analysis/validation_exon_fraction_summary.csv

Run from the project root:
    python Scripts/07_error_analysis.py --batch-size 64
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


INPUT_CHANNELS = 87
NUM_CLASSES = 2
CLASS_NAMES = ["Exon", "Intron"]

# Optional per-window metadata fields. Missing fields are simply omitted.
METADATA_FIELDS = [
    "TranscriptID", "GeneID", "Strand", "WindowNumber",
    "TranscriptStart", "TranscriptEnd", "GenomicStart", "GenomicEnd",
    "ValidLength", "ExonBases", "IntronBases", "OverlapGroup",
]


class ValidationDataset(Dataset):
    """Lazy HDF5 reader using exactly the Stage 5 feature preprocessing."""

    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        self.file = None

        with h5py.File(self.h5_path, "r") as h5:
            raw = h5["metadata/Split"][:]
            splits = np.asarray([
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in raw
            ])
            self.indices = np.flatnonzero(splits == "Validation")

            self.metadata = {}
            if "metadata" in h5:
                for field in METADATA_FIELDS:
                    key = f"metadata/{field}"
                    if key in h5:
                        self.metadata[field] = h5[key][self.indices]

        if len(self.indices) == 0:
            raise ValueError("No Validation windows found in features.h5.")

        print(f"Validation windows: {len(self.indices):,}")
        print("Available metadata:", ", ".join(self.metadata) or "none")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        row = int(self.indices[index])

        sequence = self.file["sequence"][row]
        composition = self.file["composition"][row]
        kmer = self.file["kmer"][row]
        labels = self.file["labels"][row]

        sequence_tensor = torch.from_numpy(sequence.astype(np.int64))
        one_hot = F.one_hot(sequence_tensor, num_classes=5).float()

        composition_tensor = torch.from_numpy(
            composition.astype(np.float32) / 255.0
        )
        composition_tensor[:, 1] = (
            composition_tensor[:, 1] * 2.0 - 1.0
        )

        kmer_tensor = torch.from_numpy(kmer.astype(np.float32) / 255.0)

        features = torch.cat(
            [one_hot, composition_tensor, kmer_tensor], dim=1
        )
        features = features.transpose(0, 1).contiguous()
        target = torch.from_numpy(labels.astype(np.int64))

        return features, target, index


class CNNBiLSTM(nn.Module):
    """Architecture copied from Stage 5; must match the saved checkpoint."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv1d(87, 32, kernel_size=9, padding="same")
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.cnn(x))
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        return self.classifier(x)


def load_checkpoint(model, path, device):
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is not in the expected Stage 5 format.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def calculate_metrics(confusion):
    total = int(confusion.sum())
    result = {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "valid_bases": total,
    }
    f1s = []

    for i, name in enumerate(CLASS_NAMES):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }
        f1s.append(f1)

    result["macro_f1"] = float(np.mean(f1s))
    return result


def safe_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Analyse exon/intron errors on Validation only."
    )
    parser.add_argument(
        "--data", default="feature_dataset/features.h5"
    )
    parser.add_argument(
        "--checkpoint", default="models/cnn_bilstm_features_best.pt"
    )
    parser.add_argument(
        "--output-dir", default="results/error_analysis"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.batch_size < 1 or args.threshold_step <= 0:
        raise ValueError("Batch size and threshold step must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = ValidationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CNNBiLSTM().to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}."
    )

    # Store validation probabilities and labels for threshold and error analysis.
    all_probabilities = []
    all_labels = []
    model.eval()

    print("\nRunning inference on Validation only...")
    with torch.inference_mode():
        for batch_number, (features, targets, _) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=True)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=-1)[..., 0]
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(targets.numpy())

            if batch_number % 100 == 0:
                print(f"  Processed {batch_number:,} batches")

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    thresholds = np.round(
        np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step * 0.5,
            args.threshold_step,
        ),
        6,
    )

    sweep_rows = []
    best_threshold = None
    best_metrics = None
    best_confusion = None
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = np.where(probabilities >= threshold, 0, 1)
        valid = labels != 2
        true_valid = labels[valid].astype(np.int64)
        pred_valid = predictions[valid].astype(np.int64)
        confusion = np.bincount(
            2 * true_valid + pred_valid, minlength=4
        ).reshape(2, 2)
        metrics = calculate_metrics(confusion)

        sweep_rows.append({
            "Threshold": float(threshold),
            "Accuracy": metrics["accuracy"],
            "MacroF1": metrics["macro_f1"],
            "ExonPrecision": metrics["Exon"]["precision"],
            "ExonRecall": metrics["Exon"]["recall"],
            "ExonF1": metrics["Exon"]["f1"],
            "IntronPrecision": metrics["Intron"]["precision"],
            "IntronRecall": metrics["Intron"]["recall"],
            "IntronF1": metrics["Intron"]["f1"],
        })

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
            best_confusion = confusion

    with open(
        output_dir / "validation_threshold_sweep.csv",
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Per-window error statistics at the validation-selected threshold.
    predictions = np.where(probabilities >= best_threshold, 0, 1)
    metadata = dataset.metadata
    window_rows = []
    exon_fraction_bins = [
        ("0%", 0.0, 0.0),
        (">0-1%", 0.0, 0.01),
        (">1-5%", 0.01, 0.05),
        (">5-10%", 0.05, 0.10),
        (">10-25%", 0.10, 0.25),
        (">25%", 0.25, 1.0000001),
    ]

    # Aggregate by exon proportion in the true labels.
    bin_stats = {
        name: {
            "windows": 0, "valid_bases": 0, "true_exon": 0,
            "true_intron": 0, "exon_tp": 0, "exon_fn": 0,
            "intron_fp": 0, "intron_tn": 0,
        }
        for name, _, _ in exon_fraction_bins
    }

    valid = labels != 2
    for i in range(labels.shape[0]):
        valid_i = valid[i]
        y = labels[i, valid_i].astype(np.int64)
        p = predictions[i, valid_i].astype(np.int64)

        if y.size == 0:
            continue

        exon_tp = int(np.sum((y == 0) & (p == 0)))
        exon_fn = int(np.sum((y == 0) & (p == 1)))
        intron_fp = int(np.sum((y == 1) & (p == 0)))
        intron_tn = int(np.sum((y == 1) & (p == 1)))
        true_exon = exon_tp + exon_fn
        true_intron = intron_fp + intron_tn
        exon_fraction = true_exon / y.size

        row = {
            "ValidationWindowIndex": i,
            "ValidBases": int(y.size),
            "TrueExonBases": true_exon,
            "TrueIntronBases": true_intron,
            "TrueExonFraction": exon_fraction,
            "ExonTP": exon_tp,
            "ExonFN": exon_fn,
            "IntronFP": intron_fp,
            "IntronTN": intron_tn,
            "ExonRecall": exon_tp / true_exon if true_exon else "",
            "ExonPrecision": (
                exon_tp / (exon_tp + intron_fp)
                if exon_tp + intron_fp else ""
            ),
            "MeanExonProbabilityOnTrueExon": (
                float(probabilities[i][valid_i & (labels[i] == 0)].mean())
                if true_exon else ""
            ),
            "MeanExonProbabilityOnTrueIntron": (
                float(probabilities[i][valid_i & (labels[i] == 1)].mean())
                if true_intron else ""
            ),
        }

        for field, values in metadata.items():
            row[field] = safe_text(values[i])

        window_rows.append(row)

        selected_bin = None
        for name, low, high in exon_fraction_bins:
            if name == "0%" and exon_fraction == 0:
                selected_bin = name
                break
            if name != "0%" and low < exon_fraction <= high:
                selected_bin = name
                break
        if selected_bin is not None:
            s = bin_stats[selected_bin]
            s["windows"] += 1
            s["valid_bases"] += int(y.size)
            s["true_exon"] += true_exon
            s["true_intron"] += true_intron
            s["exon_tp"] += exon_tp
            s["exon_fn"] += exon_fn
            s["intron_fp"] += intron_fp
            s["intron_tn"] += intron_tn

    window_path = output_dir / "validation_window_errors.csv"
    if window_rows:
        fields = list(window_rows[0].keys())
        with open(window_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(window_rows)

    summary_rows = []
    for name, _, _ in exon_fraction_bins:
        s = bin_stats[name]
        precision = (
            s["exon_tp"] / (s["exon_tp"] + s["intron_fp"])
            if s["exon_tp"] + s["intron_fp"] else 0.0
        )
        recall = (
            s["exon_tp"] / (s["exon_tp"] + s["exon_fn"])
            if s["exon_tp"] + s["exon_fn"] else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        summary_rows.append({
            "TrueExonFractionBin": name,
            "Windows": s["windows"],
            "ValidBases": s["valid_bases"],
            "TrueExonBases": s["true_exon"],
            "TrueIntronBases": s["true_intron"],
            "ExonTP": s["exon_tp"],
            "ExonFN": s["exon_fn"],
            "IntronFP": s["intron_fp"],
            "IntronTN": s["intron_tn"],
            "ExonPrecision": precision,
            "ExonRecall": recall,
            "ExonF1": f1,
        })

    with open(
        output_dir / "validation_exon_fraction_summary.csv",
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Probability separation on all valid validation positions.
    true_exon_probs = probabilities[(labels == 0)]
    true_intron_probs = probabilities[(labels == 1)]

    summary_path = output_dir / "validation_error_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("VALIDATION-ONLY ERROR ANALYSIS\n")
        handle.write("=" * 44 + "\n")
        handle.write(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}\n")
        handle.write(f"Selected validation threshold: {best_threshold:.4f}\n")
        handle.write(f"Validation accuracy: {best_metrics['accuracy']:.6f}\n")
        handle.write(f"Validation macro F1: {best_metrics['macro_f1']:.6f}\n")
        handle.write(
            f"Exon precision/recall/F1: "
            f"{best_metrics['Exon']['precision']:.6f} / "
            f"{best_metrics['Exon']['recall']:.6f} / "
            f"{best_metrics['Exon']['f1']:.6f}\n"
        )
        handle.write(
            f"Intron precision/recall/F1: "
            f"{best_metrics['Intron']['precision']:.6f} / "
            f"{best_metrics['Intron']['recall']:.6f} / "
            f"{best_metrics['Intron']['f1']:.6f}\n"
        )
        handle.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        handle.write(
            f"True exon:   predicted exon={best_confusion[0,0]:,}, "
            f"predicted intron={best_confusion[0,1]:,}\n"
        )
        handle.write(
            f"True intron: predicted exon={best_confusion[1,0]:,}, "
            f"predicted intron={best_confusion[1,1]:,}\n"
        )
        handle.write("\nExon-probability distribution on valid bases:\n")
        for label, values in [
            ("True exon", true_exon_probs),
            ("True intron", true_intron_probs),
        ]:
            handle.write(
                f"{label}: mean={values.mean():.6f}, "
                f"median={np.median(values):.6f}, "
                f"p10={np.quantile(values, 0.10):.6f}, "
                f"p90={np.quantile(values, 0.90):.6f}\n"
            )
        handle.write(
            "\nThis analysis used Validation only. The Testing partition "
            "was not loaded or evaluated.\n"
        )

    print("\n" + "=" * 56)
    print("VALIDATION ERROR ANALYSIS COMPLETED")
    print("=" * 56)
    print(f"Selected validation threshold: {best_threshold:.2f}")
    print(f"Validation macro F1: {best_metrics['macro_f1']:.4f}")
    print(
        f"Exon F1: {best_metrics['Exon']['f1']:.4f} | "
        f"Intron F1: {best_metrics['Intron']['f1']:.4f}"
    )
    print(f"Saved reports to: {output_dir}")
    print("Testing partition was not accessed.")
    print("=" * 56)


if __name__ == "__main__":
    main()
