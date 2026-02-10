# backend/models/evaluate_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your model class
from train_detector import DeepfakeDetector, SafeImageFolder


def evaluate_model(model_path: str, test_data_dir: str, device: str = "cuda"):
    """
    Comprehensive evaluation of trained deepfake detector

    Args:
        model_path: Path to trained model checkpoint
        test_data_dir: Path to test dataset (e.g., 'data/real_vs_fake/test')
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with all metrics
    """

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    model = DeepfakeDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Classes: {checkpoint['class_names']}")

    # Prepare test data
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = SafeImageFolder(root=test_data_dir, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Test dataset: {len(test_dataset)} images")

    # Collect predictions
    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)

            # Get model outputs
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # Probability of "fake" class (class 0 due to alphabetical order)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_scores = np.array(all_probabilities)

    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_scores)

    return metrics, y_true, y_pred, y_scores


def calculate_all_metrics(y_true, y_pred, y_scores):
    """Calculate all standard deepfake detection metrics"""

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="binary", pos_label=0)  # 0=fake
    metrics["recall"] = recall_score(y_true, y_pred, average="binary", pos_label=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="binary", pos_label=0)

    # AUC (Area Under ROC Curve)
    metrics["auc_roc"] = roc_auc_score(y_true, y_scores)

    # Equal Error Rate (EER)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=0)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    metrics["eer"] = fpr[eer_idx]
    metrics["eer_threshold"] = thresholds[eer_idx]

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    # Specificity (True Negative Rate)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # False Positive Rate
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # False Negative Rate
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics


def print_metrics(metrics: dict):
    """Pretty print all metrics"""

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Classification Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")

    print("\n📈 Performance Metrics:")
    print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"   EER:       {metrics['eer']:.4f} (threshold: {metrics['eer_threshold']:.4f})")
    print(f"   Specificity: {metrics['specificity']:.4f}")

    print("\n🔍 Confusion Matrix:")
    print(f"   True Negatives:  {metrics['true_negatives']:>6} (correctly identified real)")
    print(f"   False Positives: {metrics['false_positives']:>6} (real flagged as fake)")
    print(f"   False Negatives: {metrics['false_negatives']:>6} (fake missed)")
    print(f"   True Positives:  {metrics['true_positives']:>6} (correctly identified fake)")

    print("\n❌ Error Rates:")
    print(f"   False Positive Rate: {metrics['fpr']:.4f} ({metrics['fpr'] * 100:.2f}%)")
    print(f"   False Negative Rate: {metrics['fnr']:.4f} ({metrics['fnr'] * 100:.2f}%)")

    print("\n" + "=" * 60)


def plot_roc_curve(y_true, y_scores, save_path: str = "results/roc_curve.png"):
    """Plot and save ROC curve"""

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=0)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 ROC curve saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str = "results/confusion_matrix.png"):
    """Plot and save confusion matrix"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()

    classes = ["Fake", "Real"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()


def save_metrics_json(metrics: dict, save_path: str = "results/evaluation_metrics.json"):
    """Save metrics to JSON file"""

    Path(save_path).parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Metrics saved to: {save_path}")


def generate_latex_table(metrics: dict):
    """Generate LaTeX table for thesis"""

    latex = (
        r"""
\begin{table}[h]
\centering
\caption{EfficientNet-B4 Performance on 140k Real-Fake Faces Test Set}
\label{tab:model_performance}
\begin{tabular}{lr}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Accuracy & """
        + f"{metrics['accuracy']:.4f}"
        + r""" \\
Precision & """
        + f"{metrics['precision']:.4f}"
        + r""" \\
Recall & """
        + f"{metrics['recall']:.4f}"
        + r""" \\
F1-Score & """
        + f"{metrics['f1_score']:.4f}"
        + r""" \\
AUC-ROC & """
        + f"{metrics['auc_roc']:.4f}"
        + r""" \\
EER & """
        + f"{metrics['eer']:.4f}"
        + r""" \\
\hline
\end{tabular}
\end{table}
"""
    )

    print("\n📝 LaTeX Table for Thesis:")
    print(latex)

    with open("results/metrics_table.tex", "w") as f:
        f.write(latex)
    print("💾 LaTeX table saved to: results/metrics_table.tex")


if __name__ == "__main__":
    # Evaluate your trained model
    metrics, y_true, y_pred, y_scores = evaluate_model(
        model_path="checkpoints/best_model.pt",
        test_data_dir="data/raw/real_vs_fake/test",  # ← Change this line
        device="cuda",
    )

    # Print results
    print_metrics(metrics)

    # Generate visualizations
    plot_roc_curve(y_true, y_scores)
    plot_confusion_matrix(y_true, y_pred)

    # Save results
    save_metrics_json(metrics)
    generate_latex_table(metrics)
