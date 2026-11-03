import torch
import torch.nn as nn

from common import IMG_SIZE, NUM_CLASSES


def weights_init(m):
    """
    Standard DCGAN weight initialization:
    Convolutional layers initialized from N(0.0, 0.02),
    BatchNorm initialized from N(1.0, 0.02) with bias 0.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN Generator for 32x32 Traffic Sign Images.
    Takes latent noise vector z and class label c, generating an RGB traffic sign.
    """
    def __init__(self, latent_dim=100, num_classes=NUM_CLASSES, embed_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Learnable embedding for traffic sign classes
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Initial linear projection: (latent_dim + embed_dim) -> 256 * 4 * 4
        self.init_linear = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )

        # Transposed convolution upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Output pixel range [-1, 1]
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.init_linear(x)
        x = x.view(-1, 256, 4, 4)
        return self.conv_blocks(x)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator for 32x32 Traffic Sign Images.
    Evaluates whether a given image is authentic and matches the specified class label.
    """
    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Class embedding mapped to a 1-channel spatial mask (1 x img_size x img_size)
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        # Input: 3 image channels + 1 conditioned class channel = 4 channels
        self.features = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Reshape label embedding to spatial mask and concatenate with image
        emb = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, emb], dim=1)  # (Batch, 4, 32, 32)
        validity = self.features(x)
        return validity.view(-1, 1)


def generate_traffic_signs(generator, class_id, num_samples=8, latent_dim=100, device="cpu"):
    """
    Generate synthetic traffic signs for a specific class ID.
    Returns tensor of shape (num_samples, 3, 32, 32) in range [-1, 1].
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        fake_imgs = generator(z, labels)
    return fake_imgs


def interpolate_latent(generator, class_a, class_b=None, steps=10, latent_dim=100, device="cpu"):
    """
    Perform linear interpolation in latent space between two random seeds or two classes.
    Demonstrates continuous manifold representation learned by the GAN.
    """
    generator.eval()
    if class_b is None:
        class_b = class_a

    with torch.no_grad():
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)

        alphas = torch.linspace(0, 1, steps, device=device)
        interpolated = []
        for alpha in alphas:
            z_interp = (1.0 - alpha) * z1 + alpha * z2
            # Interpolate class label if distinct, or keep class_a
            lbl = class_a if alpha < 0.5 else class_b
            lbl_tensor = torch.tensor([lbl], dtype=torch.long, device=device)
            fake = generator(z_interp, lbl_tensor)
            interpolated.append(fake)

        return torch.cat(interpolated, dim=0)

