import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from train_detector import DeepfakeDetector, SafeImageFolder

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(model_path: str, device: torch.device) -> DeepfakeDetector:
    checkpoint = torch.load(model_path, map_location=device)
    model = DeepfakeDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model — trained for {checkpoint['epoch']} epochs")
    print(f"Training classes: {checkpoint['class_names']}")
    return model


def run_inference(model, data_dir: str, device: torch.device) -> tuple:
    dataset = SafeImageFolder(root=data_dir, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images | Classes: {dataset.classes}")

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running inference"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        dataset.classes,
    )


def compute_metrics(y_true, y_pred, y_probs, dataset_name: str) -> dict:
    # Determine which class index corresponds to "fake"
    # SafeImageFolder uses alphabetical order: fake=0, real=1
    fake_class_idx = 0

    return {
        "dataset": dataset_name,
        "n_samples": int(len(y_true)),
        "n_real": int(np.sum(y_true == 1)),
        "n_fake": int(np.sum(y_true == 0)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, pos_label=fake_class_idx, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, pos_label=fake_class_idx, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_true, y_pred, pos_label=fake_class_idx, zero_division=0)), 4
        ),
        "auc_roc": round(float(roc_auc_score(y_true, y_probs)), 4),
    }


def print_results(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print(f"CIPLAB EVALUATION — {metrics['dataset']}")
    print("=" * 50)
    print(
        f"  Samples:   {metrics['n_samples']} ({metrics['n_real']} real, {metrics['n_fake']} fake)"
    )
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="backend/checkpoints/best_model.pt")
    parser.add_argument("--data", default="backend/data/ciplab")
    parser.add_argument("--output", default="backend/results/ciplab_evaluation.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model, device)
    y_true, y_pred, y_probs, classes = run_inference(model, args.data, device)
    metrics = compute_metrics(y_true, y_pred, y_probs, dataset_name="CIPLAB")

    print_results(metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {args.output}")
