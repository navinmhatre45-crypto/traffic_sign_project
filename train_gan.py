import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils

from common import gan_transform, get_device, denormalize, CLASS_NAMES
from gan import ConditionalGenerator, ConditionalDiscriminator, weights_init


def save_sample_grid(generator, fixed_z, fixed_labels, epoch, output_dir="samples", device="cpu"):
    """Saves a grid of generated samples across representative classes."""
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_z, fixed_labels)
        fake_imgs = denormalize(fake_imgs)
        grid = vutils.make_grid(fake_imgs, nrow=8, padding=2, normalize=False)
        filepath = os.path.join(output_dir, f"cgan_epoch_{epoch:02d}.png")
        vutils.save_image(grid, filepath)
    generator.train()
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Train Conditional GAN on GTSRB Traffic Signs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam (default: 0.0002)")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent noise vector dimension (default: 100)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'mps', 'cuda', or 'cpu'")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"=== Starting Conditional GAN Training ===")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | Latent Dim: {args.latent_dim}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Load dataset
    print("Loading GTSRB dataset...")
    train_dataset = datasets.GTSRB("data", split="train", transform=gan_transform, download=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    print(f"Dataset loaded: {len(train_dataset)} images in {len(loader)} batches.")

    # Initialize Generator and Discriminator
    netG = ConditionalGenerator(latent_dim=args.latent_dim).to(device)
    netD = ConditionalDiscriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function and Optimizers (Radford et al. DCGAN recommendations)
    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Fixed evaluation batch: 8 classes x 8 samples each = 64 total images
    # Classes: 0 (20km/h), 14 (Stop), 17 (No Entry), 19 (Dangerous Curve),
    #          28 (Children crossing), 33 (Turn right), 38 (Keep right), 41 (End of passing)
    eval_classes = [0, 14, 17, 19, 28, 33, 38, 41]
    fixed_labels = torch.tensor([c for c in eval_classes for _ in range(8)], dtype=torch.long, device=device)
    fixed_z = torch.randn(64, args.latent_dim, device=device)

    history = {"loss_g": [], "loss_d": []}

    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()
        running_loss_d, running_loss_g = 0.0, 0.0

        for i, (real_imgs, real_labels) in enumerate(loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Targets with one-sided label smoothing (0.9 instead of 1.0)
            real_targets = torch.full((batch_size, 1), 0.9, device=device)
            fake_targets = torch.zeros((batch_size, 1), device=device)

            # ---------------------
            #  1. Train Discriminator
            # ---------------------
            optD.zero_grad()

            # Real images loss
            d_real_out = netD(real_imgs, real_labels)
            loss_d_real = criterion(d_real_out, real_targets)

            # Fake images loss
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_labels = torch.randint(0, 43, (batch_size,), device=device)
            fake_imgs = netG(z, gen_labels)

            d_fake_out = netD(fake_imgs.detach(), gen_labels)
            loss_d_fake = criterion(d_fake_out, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # ---------------------
            #  2. Train Generator
            # ---------------------
            optG.zero_grad()

            # Generator aims to make Discriminator predict real (1.0)
            d_gen_out = netD(fake_imgs, gen_labels)
            loss_g = criterion(d_gen_out, real_targets)
            loss_g.backward()
            optG.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()

            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                      f"Batch [{i+1:03d}/{len(loader):03d}] "
                      f"Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")

        epoch_loss_d = running_loss_d / len(loader)
        epoch_loss_g = running_loss_g / len(loader)
        history["loss_d"].append(epoch_loss_d)
        history["loss_g"].append(epoch_loss_g)

        # Save visual samples
        sample_path = save_sample_grid(netG, fixed_z, fixed_labels, epoch, device=device)
        print(f"--> Epoch {epoch:02d} Complete. Avg Loss_D: {epoch_loss_d:.4f}, Loss_G: {epoch_loss_g:.4f}")
        print(f"    Saved visual sample grid to: {sample_path}")

        # Save model checkpoints
        torch.save(netG.state_dict(), "checkpoints/cgan_generator.pth")
        torch.save(netD.state_dict(), "checkpoints/cgan_discriminator.pth")

    # Plot and save GAN training curves
    plt.figure(figsize=(9, 4))
    plt.plot(history["loss_d"], label="Discriminator Loss (D)", color="crimson")
    plt.plot(history["loss_g"], label="Generator Loss (G)", color="royalblue")
    plt.title("Conditional GAN Training Loss Curves (GTSRB)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("gan_training_curves.png", dpi=200)
    plt.close()

    print("\n=== Training Completed Successfully! ===")
    print("Checkpoints saved:")
    print(" - checkpoints/cgan_generator.pth")
    print(" - checkpoints/cgan_discriminator.pth")
    print(" - gan_training_curves.png")


if __name__ == "__main__":
    main()

