import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets

from common import TrafficSignCNN, train_transform, eval_transform, gan_transform, get_device, CLASS_NAMES, NUM_CLASSES
from gan import ConditionalGenerator, generate_traffic_signs


class SyntheticSignDataset(Dataset):
    """PyTorch Dataset holding GAN-generated synthetic traffic sign images."""
    def __init__(self, images, labels):
        self.images = images  # Tensor of shape (N, 3, 32, 32)
        self.labels = [int(lbl.item()) if torch.is_tensor(lbl) else int(lbl) for lbl in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def evaluate_per_class(model, loader, device, num_classes=NUM_CLASSES):
    """Evaluates overall accuracy and per-class recall/accuracy."""
    model.eval()
    correct_per_class = Counter()
    total_per_class = Counter()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

            for p, label in zip(preds.cpu().tolist(), y.cpu().tolist()):
                total_per_class[label] += 1
                if p == label:
                    correct_per_class[label] += 1

    per_class_recalls = [
        (correct_per_class[c] / total_per_class[c]) if total_per_class[c] > 0 else 0.0
        for c in range(num_classes)
    ]
    overall_acc = total_correct / total_samples
    return overall_acc, per_class_recalls


def main():
    parser = argparse.ArgumentParser(description="Evaluate & Fine-tune TrafficSignCNN with GAN Data Augmentation")
    parser.add_argument("--gan_checkpoint", type=str, default="checkpoints/cgan_generator.pth",
                        help="Path to trained GAN generator")
    parser.add_argument("--samples_per_minority_class", type=int, default=250,
                        help="Number of synthetic samples to synthesize for each rare class")
    parser.add_argument("--minority_threshold", type=int, default=300,
                        help="Class sample threshold below which a class is considered underrepresented")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of fine-tuning epochs with GAN augmentation")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for fine-tuning")
    args = parser.parse_args()

    device = get_device()
    print("=== Traffic Sign Recognition: GAN Data Augmentation Pipeline ===")
    print(f"Device: {device}")

    # 1. Load real datasets
    print("Loading GTSRB dataset...")
    train_dataset = datasets.GTSRB("data", split="train", transform=train_transform, download=True)
    test_dataset = datasets.GTSRB("data", split="test", transform=eval_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. Analyze class distribution and identify minority classes
    train_labels = [label for _, label in train_dataset]
    class_counts = Counter(train_labels)
    minority_classes = [c for c in range(NUM_CLASSES) if class_counts[c] < args.minority_threshold]

    print(f"\nTotal training images: {len(train_dataset)}")
    print(f"Minority classes (< {args.minority_threshold} images): {minority_classes}")
    for mc in minority_classes:
        print(f"  Class {mc:02d} ({CLASS_NAMES[mc]}): {class_counts[mc]} real images")

    # 3. Evaluate Baseline Model
    baseline_model = TrafficSignCNN().to(device)
    baseline_path = "checkpoints/traffic_sign_cnn.pth"
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found. Please train baseline first.")
        return

    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    base_acc, base_recalls = evaluate_per_class(baseline_model, test_loader, device)
    print(f"\n[Baseline Model] Overall Test Accuracy: {base_acc * 100:.2f}%")
    base_minority_avg = np.mean([base_recalls[c] for c in minority_classes])
    print(f"[Baseline Model] Average Recall on Minority Classes: {base_minority_avg * 100:.2f}%")

    # 4. Generate Synthetic Data with cGAN
    if not os.path.exists(args.gan_checkpoint):
        print(f"\nError: GAN Generator checkpoint '{args.gan_checkpoint}' not found.")
        print("Please train the GAN first with: python3 train_gan.py")
        return

    print(f"\nSynthesizing {args.samples_per_minority_class} images per minority class using cGAN...")
    netG = ConditionalGenerator().to(device)
    netG.load_state_dict(torch.load(args.gan_checkpoint, map_location=device))
    netG.eval()

    synth_images = []
    synth_labels = []

    for mc in minority_classes:
        fakes = generate_traffic_signs(netG, mc, num_samples=args.samples_per_minority_class, device=device)
        synth_images.append(fakes.cpu())
        synth_labels.extend([mc] * args.samples_per_minority_class)

    synth_images = torch.cat(synth_images, dim=0)
    synth_labels = torch.tensor(synth_labels, dtype=torch.long)
    synthetic_dataset = SyntheticSignDataset(synth_images, synth_labels)
    print(f"Generated {len(synthetic_dataset)} synthetic traffic signs.")

    # 5. Create Augmented Dataset
    # Concat real training set + synthetic dataset
    augmented_train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    augmented_loader = DataLoader(augmented_train_dataset, batch_size=64, shuffle=True)
    print(f"Combined Training Size: {len(augmented_train_dataset)} "
          f"({len(train_dataset)} real + {len(synthetic_dataset)} GAN synthetic)")

    # 6. Fine-tune CNN on Augmented Dataset
    aug_model = TrafficSignCNN().to(device)
    aug_model.load_state_dict(torch.load(baseline_path, map_location=device))
    optimizer = torch.optim.Adam(aug_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning model on GAN-augmented distribution for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        aug_model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in augmented_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = aug_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {run_loss / total:.4f} - Train Acc: {correct / total * 100:.2f}%")

    # 7. Evaluate Augmented Model
    aug_acc, aug_recalls = evaluate_per_class(aug_model, test_loader, device)
    aug_minority_avg = np.mean([aug_recalls[c] for c in minority_classes])
    print(f"\n[GAN-Augmented Model] Overall Test Accuracy: {aug_acc * 100:.2f}%")
    print(f"[GAN-Augmented Model] Average Recall on Minority Classes: {aug_minority_avg * 100:.2f}%")

    torch.save(aug_model.state_dict(), "checkpoints/traffic_sign_cnn_gan_augmented.pth")
    print("Saved GAN-augmented model to: checkpoints/traffic_sign_cnn_gan_augmented.pth")

    # 8. Generate Side-by-Side Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x_indices = np.arange(len(minority_classes))
    width = 0.35

    base_vals = [base_recalls[c] * 100 for c in minority_classes]
    aug_vals = [aug_recalls[c] * 100 for c in minority_classes]

    ax.bar(x_indices - width / 2, base_vals, width, label="Baseline CNN (Real Data Only)", color="#4A90E2")
    ax.bar(x_indices + width / 2, aug_vals, width, label="GAN-Augmented CNN (+ Synthetic Data)", color="#2ECC71")

    class_short_labels = [f"C{c}: {CLASS_NAMES[c][:15]}..." for c in minority_classes]
    ax.set_ylabel("Test Set Recall (%)", fontsize=12)
    ax.set_title("Impact of Conditional GAN Synthetic Data Augmentation on Rare Traffic Signs", fontsize=14, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(class_short_labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("gan_augmentation_impact.png", dpi=200)
    plt.close()

    # 9. Write Comprehensive Report
    with open("gan_impact_report.txt", "w") as f:
        f.write("=== IMPACT REPORT: CONDITIONAL GAN DATA AUGMENTATION ===\n")
        f.write(f"Baseline Overall Accuracy: {base_acc * 100:.2f}%\n")
        f.write(f"GAN-Augmented Overall Accuracy: {aug_acc * 100:.2f}%\n")
        f.write(f"Baseline Minority Class Recall: {base_minority_avg * 100:.2f}%\n")
        f.write(f"GAN-Augmented Minority Class Recall: {aug_minority_avg * 100:.2f}%\n\n")
        f.write("Breakdown of Rare Classes:\n")
        for c in minority_classes:
            f.write(f"  Class {c:02d} ({CLASS_NAMES[c]}):\n")
            f.write(f"    Baseline Recall: {base_recalls[c] * 100:.2f}%\n")
            f.write(f"    GAN-Augmented Recall: {aug_recalls[c] * 100:.2f}%\n")
            diff = (aug_recalls[c] - base_recalls[c]) * 100
            sign = "+" if diff >= 0 else ""
            f.write(f"    Change: {sign}{diff:.2f}%\n")

    print("\nSaved comparison plot: gan_augmentation_impact.png")
    print("Saved text report: gan_impact_report.txt")


if __name__ == "__main__":
    main()
