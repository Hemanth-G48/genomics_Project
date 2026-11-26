#!/usr/bin/env python3
"""
Stage 6: Evaluate the trained feature-enhanced CNN-BiLSTM.

Uses:
    feature_dataset/features.h5
    models/cnn_bilstm_features_best.pt

Procedure:
    1. Load the checkpoint saved by 05_train_model.py.
    2. Select an exon-probability threshold using VALIDATION only.
    3. Apply that fixed threshold once to the untouched TEST partition.
    4. Save test metrics, confusion matrix, threshold, and classification report.

Labels:
    E = 0
    I = 1
    X = 2 (padding; excluded from metrics)
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


WINDOW_LENGTH = 2048
INPUT_CHANNELS = 87
NUM_CLASSES = 2
CLASS_NAMES = ["Exon", "Intron"]


class GenomicDataset(Dataset):
    """Lazy HDF5 reader matching the Stage 5 preprocessing exactly."""

    def __init__(self, h5_path, split):
        self.h5_path = str(h5_path)
        self.split = split
        self.file = None

        with h5py.File(self.h5_path, "r") as h5:
            if "metadata/Split" not in h5:
                raise KeyError("HDF5 file is missing metadata/Split.")
            raw_splits = h5["metadata/Split"][:]
            splits = np.asarray([
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in raw_splits
            ])

        available = sorted(set(splits.tolist()))
        self.indices = np.flatnonzero(splits == split)

        if len(self.indices) == 0:
            raise ValueError(
                f"No windows found for split '{split}'. "
                f"Split values in HDF5: {available}"
            )

        print(f"{split}: {len(self.indices):,} windows")

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

        # Match Stage 5: A=0, C=1, G=2, T=3, N=4.
        sequence_tensor = torch.from_numpy(sequence.astype(np.int64))
        one_hot = F.one_hot(sequence_tensor, num_classes=5).float()

        composition_tensor = torch.from_numpy(
            composition.astype(np.float32) / 255.0
        )
        # Decode GC skew from [0,1] to [-1,1].
        composition_tensor[:, 1] = (
            composition_tensor[:, 1] * 2.0 - 1.0
        )

        kmer_tensor = torch.from_numpy(
            kmer.astype(np.float32) / 255.0
        )

        # (L,5) + (L,2) + (L,80) = (L,87)
        features = torch.cat(
            [one_hot, composition_tensor, kmer_tensor], dim=1
        )

        # Conv1d input: (channels, sequence_length).
        features = features.transpose(0, 1).contiguous()
        target = torch.from_numpy(labels.astype(np.int64))

        return features, target


class CNNBiLSTM(nn.Module):
    """Must match the architecture in 05_train_model.py."""

    def __init__(self):
        super().__init__()

        self.cnn = nn.Conv1d(
            in_channels=INPUT_CHANNELS,
            out_channels=32,
            kernel_size=9,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        return self.classifier(x)


def confusion_from_predictions(targets, predictions):
    """Rows are true classes; columns are predicted classes."""
    valid = targets != 2
    true_valid = targets[valid].astype(np.int64)
    pred_valid = predictions[valid].astype(np.int64)

    return np.bincount(
        NUM_CLASSES * true_valid + pred_valid,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confusion):
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    result = {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": 0.0,
        "valid_bases": total,
    }

    f1_values = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }
        f1_values.append(f1)

    result["macro_f1"] = float(np.mean(f1_values))
    return result


def load_checkpoint(model, checkpoint_path, device):
    # The checkpoint is a trusted file produced by our own Stage 5 script.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint format: expected a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "Check that this is cnn_bilstm_features_best.pt from Stage 5."
        )

    config = checkpoint.get("model_config", {})
    expected = {
        "input_channels": INPUT_CHANNELS,
        "cnn_filters": 32,
        "cnn_kernel_size": 9,
        "bilstm_hidden_size": 32,
        "bidirectional": True,
        "num_classes": NUM_CLASSES,
        "window_length": WINDOW_LENGTH,
    }
    for key, expected_value in expected.items():
        if key in config and config[key] != expected_value:
            raise ValueError(
                f"Checkpoint architecture mismatch for {key}: "
                f"expected {expected_value}, found {config[key]}."
            )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def validation_threshold_search(model, loader, device, thresholds):
    """
    Select the threshold maximizing validation macro-F1.
    No test data is read in this function.
    """
    confusions = {
        float(t): np.zeros((2, 2), dtype=np.int64)
        for t in thresholds
    }

    model.eval()
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            exon_prob = torch.softmax(logits, dim=-1)[..., 0].cpu().numpy()
            target_np = targets.numpy()

            for threshold in thresholds:
                pred_np = np.where(exon_prob >= threshold, 0, 1)
                confusions[float(threshold)] += confusion_from_predictions(
                    target_np, pred_np
                )

    rows = []
    best_threshold = None
    best_metrics = None
    best_f1 = -1.0

    for threshold in thresholds:
        confusion = confusions[float(threshold)]
        metrics = metrics_from_confusion(confusion)
        rows.append({
            "Threshold": float(threshold),
            "ValidationAccuracy": metrics["accuracy"],
            "ValidationMacroF1": metrics["macro_f1"],
            "ExonF1": metrics["Exon"]["f1"],
            "IntronF1": metrics["Intron"]["f1"],
            "ValidBases": metrics["valid_bases"],
        })

        # np.arange is ascending, so ties retain the lower threshold.
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics, rows


def evaluate_test_once(model, loader, device, threshold):
    """Apply the already-selected validation threshold to test exactly once."""
    confusion = np.zeros((2, 2), dtype=np.int64)
    model.eval()

    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            exon_prob = torch.softmax(logits, dim=-1)[..., 0].cpu().numpy()
            target_np = targets.numpy()
            predictions = np.where(exon_prob >= threshold, 0, 1)
            confusion += confusion_from_predictions(target_np, predictions)

    return confusion, metrics_from_confusion(confusion)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Stage 5 CNN-BiLSTM checkpoint."
    )
    parser.add_argument(
        "--data", default="feature_dataset/features.h5"
    )
    parser.add_argument(
        "--checkpoint", default="models/cnn_bilstm_features_best.pt"
    )
    parser.add_argument(
        "--output-dir", default="results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8
    )
    parser.add_argument(
        "--num-workers", type=int, default=4
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.50
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.98
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"Feature dataset not found: {data_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if args.threshold_min < 0 or args.threshold_max > 1:
        raise ValueError("Threshold limits must be between 0 and 1.")
    if args.threshold_min > args.threshold_max:
        raise ValueError("--threshold-min must not exceed --threshold-max.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Confirm required HDF5 arrays exist before starting evaluation.
    with h5py.File(data_path, "r") as h5:
        required = [
            "sequence", "composition", "kmer", "labels",
            "metadata/Split",
        ]
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"HDF5 file is missing required datasets: {missing}")

    # Load the exact Stage 5 architecture and weights.
    model = CNNBiLSTM().to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    print(
        "Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')} "
        f"(best validation macro F1 recorded by Stage 5: "
        f"{checkpoint.get('best_validation_macro_f1', 'unknown')})"
    )

    # Threshold selection uses validation only.
    val_dataset = GenomicDataset(data_path, "Validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step * 0.5,
        args.threshold_step,
    )
    thresholds = np.round(thresholds, 6)

    print("\nSelecting exon-probability threshold on VALIDATION...")
    best_threshold, val_metrics, threshold_rows = validation_threshold_search(
        model, val_loader, device, thresholds
    )

    threshold_path = output_dir / "validation_threshold.csv"
    write_csv(
        threshold_path,
        [
            "Threshold", "ValidationAccuracy", "ValidationMacroF1",
            "ExonF1", "IntronF1", "ValidBases",
        ],
        threshold_rows,
    )

    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation macro F1 at selected threshold: {val_metrics['macro_f1']:.4f}")

    # Only now load and evaluate the untouched test partition.
    test_dataset = GenomicDataset(data_path, "Testing")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\nEvaluating TEST partition once...")
    confusion, test_metrics = evaluate_test_once(
        model, test_loader, device, best_threshold
    )

    # Save the test confusion matrix: rows=true, columns=predicted.
    confusion_path = output_dir / "test_confusion_matrix.csv"
    with open(confusion_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["TrueClass", "Predicted_Exon", "Predicted_Intron"])
        writer.writerow(["True_Exon", int(confusion[0, 0]), int(confusion[0, 1])])
        writer.writerow(["True_Intron", int(confusion[1, 0]), int(confusion[1, 1])])

    metric_rows = [
        {"Metric": "SelectedValidationThreshold", "Value": best_threshold},
        {"Metric": "TestAccuracy", "Value": test_metrics["accuracy"]},
        {"Metric": "TestMacroF1", "Value": test_metrics["macro_f1"]},
        {"Metric": "TestValidBases", "Value": test_metrics["valid_bases"]},
    ]
    for class_name in CLASS_NAMES:
        for metric_name in ["precision", "recall", "f1", "support"]:
            metric_rows.append({
                "Metric": f"{class_name}{metric_name.title()}",
                "Value": test_metrics[class_name][metric_name],
            })

    metrics_path = output_dir / "test_metrics.csv"
    write_csv(metrics_path, ["Metric", "Value"], metric_rows)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("INDEPENDENT TEST-SET EVALUATION\n")
        handle.write("=" * 42 + "\n")
        handle.write(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}\n")
        handle.write(f"Validation-selected exon threshold: {best_threshold:.4f}\n")
        handle.write("Confusion matrix (rows=true, columns=predicted):\n")
        handle.write("                 Pred Exon   Pred Intron\n")
        handle.write(
            f"True Exon        {confusion[0, 0]:10d}   {confusion[0, 1]:11d}\n"
        )
        handle.write(
            f"True Intron      {confusion[1, 0]:10d}   {confusion[1, 1]:11d}\n"
        )
        handle.write("\n")
        handle.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        handle.write(f"Macro F1: {test_metrics['macro_f1']:.6f}\n")
        handle.write(f"Valid bases: {test_metrics['valid_bases']:,}\n\n")

        for class_name in CLASS_NAMES:
            values = test_metrics[class_name]
            handle.write(f"{class_name}:\n")
            handle.write(f"  Precision: {values['precision']:.6f}\n")
            handle.write(f"  Recall:    {values['recall']:.6f}\n")
            handle.write(f"  F1-score:  {values['f1']:.6f}\n")
            handle.write(f"  Support:   {values['support']:,}\n\n")

        handle.write(
            "Protocol: threshold selected using validation only; "
            "test partition evaluated once. Padding label X was excluded.\n"
        )

    print("\n" + "=" * 54)
    print("INDEPENDENT TEST EVALUATION COMPLETED")
    print("=" * 54)
    print(f"Validation-selected threshold: {best_threshold:.2f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    for class_name in CLASS_NAMES:
        values = test_metrics[class_name]
        print(
            f"{class_name}: Precision={values['precision']:.4f}, "
            f"Recall={values['recall']:.4f}, F1={values['f1']:.4f}"
        )
    print(f"\nSaved: {threshold_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {confusion_path}")
    print(f"Saved: {report_path}")
    print("Test partition was not used for threshold selection.")
    print("=" * 54)


if __name__ == "__main__":
    main()
