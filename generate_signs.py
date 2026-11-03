import argparse
import os
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from common import CLASS_NAMES, NUM_CLASSES, get_device, denormalize, TrafficSignCNN
from gan import ConditionalGenerator, generate_traffic_signs, interpolate_latent


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic signs using trained cGAN")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cgan_generator.pth",
                        help="Path to generator checkpoint")
    parser.add_argument("--class_id", type=int, default=14,
                        help="Class ID to generate (0 to 42, default: 14 = Stop)")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--output", type=str, default="generated_signs.png",
                        help="Output image path")
    parser.add_argument("--gallery", action="store_true",
                        help="Generate a comprehensive gallery of all 43 classes")
    parser.add_argument("--interpolate", action="store_true",
                        help="Perform latent space interpolation between two classes")
    parser.add_argument("--class_b", type=int, default=17,
                        help="Second class ID for interpolation (default: 17 = No Entry)")
    parser.add_argument("--classify", action="store_true",
                        help="Verify generated signs using the TrafficSignCNN classifier")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading Conditional Generator from: {args.checkpoint} on {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please train the GAN first with: python3 train_gan.py")
        return

    netG = ConditionalGenerator().to(device)
    netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
    netG.eval()

    if args.gallery:
        print("Generating 1 sample for each of the 43 classes...")
        all_samples = []
        with torch.no_grad():
            for c in range(NUM_CLASSES):
                z = torch.randn(1, 100, device=device)
                lbl = torch.tensor([c], device=device)
                fake = netG(z, lbl)
                all_samples.append(fake)
        tensor_all = torch.cat(all_samples, dim=0)
        grid = vutils.make_grid(denormalize(tensor_all), nrow=8, padding=2)
        vutils.save_image(grid, args.output)
        print(f"Saved 43-class synthetic gallery to: {args.output}")

    elif args.interpolate:
        print(f"Interpolating latent space: Class {args.class_id} ({CLASS_NAMES[args.class_id]}) "
              f"--> Class {args.class_b} ({CLASS_NAMES[args.class_b]})")
        interpolated = interpolate_latent(netG, args.class_id, args.class_b, steps=12, device=device)
        grid = vutils.make_grid(denormalize(interpolated), nrow=12, padding=2)
        vutils.save_image(grid, args.output)
        print(f"Saved latent interpolation strip to: {args.output}")

    else:
        print(f"Generating {args.num_samples} samples of Class {args.class_id}: '{CLASS_NAMES[args.class_id]}'")
        fakes = generate_traffic_signs(netG, args.class_id, num_samples=args.num_samples, device=device)

        if args.classify:
            cnn_path = "checkpoints/traffic_sign_cnn.pth"
            if os.path.exists(cnn_path):
                cnn = TrafficSignCNN().to(device)
                cnn.load_state_dict(torch.load(cnn_path, map_location=device))
                cnn.eval()
                with torch.no_grad():
                    logits = cnn(fakes)
                    preds = logits.argmax(1).cpu().tolist()
                print("\nClassifier verification on generated images:")
                for idx, p in enumerate(preds):
                    match = "MATCH" if p == args.class_id else "MISMATCH"
                    print(f"  Sample {idx+1:02d}: Predicted = '{CLASS_NAMES[p]}' [{match}]")
                acc = sum(1 for p in preds if p == args.class_id) / len(preds)
                print(f"Classifier Recognition Rate on Synthetic Signs: {acc * 100:.1f}%\n")

        grid = vutils.make_grid(denormalize(fakes), nrow=4, padding=2)
        vutils.save_image(grid, args.output)
        print(f"Saved synthetic sign grid to: {args.output}")


if __name__ == "__main__":
    main()

