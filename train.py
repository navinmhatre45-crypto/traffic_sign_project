import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from common import TrafficSignCNN, train_transform, eval_transform


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train traffic sign classifier on GTSRB")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # Download GTSRB (first run only, about 300 MB)
    print("Loading dataset (downloads on first run)...")
    train_full = datasets.GTSRB("data", split="train", transform=train_transform, download=True)
    val_full = datasets.GTSRB("data", split="train", transform=eval_transform, download=True)
    test_set = datasets.GTSRB("data", split="test", transform=eval_transform, download=True)

    # 90% train / 10% validation (same indices for both versions of the dataset)
    n = len(train_full)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(0.1 * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=256, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=0)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_set)}")

    model = TrafficSignCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        tr_loss, tr_acc = run_loss / total, correct / total
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "checkpoints/traffic_sign_cnn.pth")

    # Final test on the best saved model
    model.load_state_dict(torch.load("checkpoints/traffic_sign_cnn.pth", map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest validation accuracy: {best_val:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="Train")
    ax[0].plot(history["val_loss"], label="Validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="Train")
    ax[1].plot(history["val_acc"], label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training_curves.png and checkpoints/traffic_sign_cnn.pth")


if __name__ == "__main__":
    main()