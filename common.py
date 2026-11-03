import torch
import torch.nn as nn
from torchvision import transforms

IMG_SIZE = 32
NUM_CLASSES = 43

# Class names in GTSRB label order (0 to 42)
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 tons",
]

_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Used for validation, testing and prediction
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _norm,
])

# Used for training only. No horizontal flip, because left/right signs differ.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    _norm,
])

# GAN specific transform (clean resize and [-1, 1] normalization)
gan_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _norm,
])


def get_device():
    """Detect available accelerator: CUDA, Apple Silicon MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(tensor):
    """Converts a [-1, 1] normalized tensor to [0, 1] for visualization."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 32), nn.MaxPool2d(2), nn.Dropout(0.2),     # 16x16
            block(32, 64), block(64, 64), nn.MaxPool2d(2), nn.Dropout(0.3),    # 8x8
            block(64, 128), nn.MaxPool2d(2), nn.Dropout(0.4),                  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
