"""
Fine-tune the deepfake detector on the FFHQ+SG3 dataset.

Loads ``best_model_140k_ciplab_ffpp.pt`` (the latest checkpoint from the
progressive cross-dataset runs) and continues training at a low learning
rate on the FFHQ+SG3 train/val split produced by ``prepare_ffhq_sg3.py``.

The goal is to bring detector accuracy on the FFHQ+SG3 held-out test set up
to a level where the 12 user-study images can be selected as all-correctly
classified — so the user study evaluates VLM explanation quality rather
than detector failures.

The fine-tune uses separate learning rates for the backbone (very low, so
we don't destroy the existing feature representations) and the classifier
head (slightly higher, so it can adapt to the new distribution).

Usage:
    python -m backend.scripts.fine_tune_ffhq_sg3
    python -m backend.scripts.fine_tune_ffhq_sg3 --epochs 5 --lr-classifier 2e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend" / "models"))
from train_detector import DeepfakeDetector, SafeImageFolder  # noqa: E402

DEFAULT_RESUME = REPO_ROOT / "backend" / "checkpoints" / "best_model_140k_ciplab_ffpp.pt"
DEFAULT_DATA = REPO_ROOT / "backend" / "data" / "ffhq_sg3"
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "checkpoints" / "best_model_140k_ciplab_ffpp_ffhq_sg3.pt"
DEFAULT_HISTORY = REPO_ROOT / "backend" / "results" / "ffhq_sg3_finetune_history.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Training transforms use mild augmentation; val uses deterministic resize."""
    train_t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_t, val_t


def load_resume_checkpoint(model: nn.Module, resume_path: Path) -> int:
    """Load weights from ``resume_path`` into ``model``.

    Returns the epoch index stored in the checkpoint, for logging only.
    """
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    prior_epochs = int(checkpoint.get("epoch", -1)) + 1
    classes = checkpoint.get("class_names", ["?", "?"])
    print(f"Loaded {resume_path.name} (trained {prior_epochs} epochs, classes={classes})")
    return prior_epochs


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    """One pass over ``loader``. Pass ``optimizer=None`` for eval."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--val-subdir", default="val")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-classifier", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: running on CPU — fine-tune will be very slow.")

    train_t, val_t = build_transforms()
    train_dir = args.data / args.train_subdir
    val_dir = args.data / args.val_subdir
    train_ds = SafeImageFolder(root=str(train_dir), transform=train_t)
    val_ds = SafeImageFolder(root=str(val_dir), transform=val_t)
    print(f"Train: {len(train_ds)} images from {train_dir} (classes={train_ds.classes})")
    print(f"Val:   {len(val_ds)} images from {val_dir}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DeepfakeDetector().to(device)
    prior_epochs = load_resume_checkpoint(model, args.resume)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.features.parameters(), "lr": args.lr_backbone},
            {"params": model.model.classifier.parameters(), "lr": args.lr_classifier},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    print(
        f"Fine-tune: {args.epochs} epochs, batch={args.batch_size}, "
        f"lr_classifier={args.lr_classifier:.0e}, lr_backbone={args.lr_backbone:.0e}"
    )

    best_val_acc = 0.0
    patience_counter = 0
    history: dict = {
        "resumed_from": str(args.resume),
        "prior_epochs": prior_epochs,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_classifier": args.lr_classifier,
            "lr_backbone": args.lr_backbone,
            "weight_decay": args.weight_decay,
        },
        "epochs": [],
        "started_at": datetime.now().isoformat(),
    }

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, desc="train"
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, desc="val")
        scheduler.step(val_loss)
        print(
            f"  train: loss={train_loss:.4f} acc={train_acc:.2f}%  |  "
            f"val: loss={val_loss:.4f} acc={val_acc:.2f}%"
        )

        history["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": prior_epochs + epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "class_names": train_ds.classes,
                    "resumed_from": args.resume.name,
                },
                args.output,
            )
            print(f"  saved (val_acc={val_acc:.2f}%) -> {args.output.name}")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"  early stopping after epoch {epoch + 1}")
                break

    history["best_val_acc"] = round(best_val_acc, 2)
    history["finished_at"] = datetime.now().isoformat()
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps(history, indent=2))
    print(f"\nDone. best_val_acc={best_val_acc:.2f}% -> {args.output}\nhistory -> {args.history}")


if __name__ == "__main__":
    main()
