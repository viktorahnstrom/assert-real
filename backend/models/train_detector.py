# backend/models/train_detector.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

# Custom dataset class that skips corrupted images
class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupted images"""
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            print(f"\nWarning: Skipping corrupted image at index {index}: {e}")
            # Return next valid image
            return self.__getitem__((index + 1) % len(self))

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load pre-trained EfficientNet-B4
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        self.model = efficientnet_b4(weights=weights)

        # Freeze backbone initially
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(data_dir="data/real_vs_fake", epochs=5, batch_size=32, lr=0.001, max_train_samples=15000):
    """Train deepfake detector with limited samples for faster training"""

    # Device selection - Apple Silicon support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Training with max {max_train_samples} samples for faster iteration")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full datasets
    full_train_dataset = SafeImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )
    full_val_dataset = SafeImageFolder(
        root=f"{data_dir}/valid",
        transform=val_transform
    )

    # Limit dataset size for faster training
    train_size = min(max_train_samples, len(full_train_dataset))
    val_size = min(max_train_samples // 5, len(full_val_dataset))  # 20% of train size

    print(f"Full dataset: {len(full_train_dataset)} train, {len(full_val_dataset)} val")
    print(f"Using subset: {train_size} train, {val_size} val")

    # Create random subsets
    train_indices = torch.randperm(len(full_train_dataset))[:train_size]
    val_indices = torch.randperm(len(full_val_dataset))[:val_size]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")
    print(f"Classes: {full_train_dataset.classes}")

    # Model, loss, optimizer
    model = DeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_samples': train_size,
                'class_names': full_train_dataset.classes
            }, "checkpoints/best_model.pt")
            print(f"   ✓ Saved best model (val_acc: {val_acc:.2f}%)")

    # Save final results
    results = {
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "epochs": epochs,
        "train_samples": train_size,
        "val_samples": val_size,
        "history": history,
        "timestamp": datetime.now().isoformat()
    }

    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: checkpoints/best_model.pt")
    print(f"   Results saved to: checkpoints/training_results.json")
    print(f"{'='*60}")

    return model, history

if __name__ == "__main__":
    # Train with smaller dataset for faster development
    model, history = train_model(
        data_dir="data/real_vs_fake",
        epochs=5,                  # Reduced from 10
        batch_size=32,
        lr=0.001,
        max_train_samples=15000    # Use 15k instead of 100k
    )
