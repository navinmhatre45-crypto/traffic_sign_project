import argparse

import torch
from PIL import Image

from common import TrafficSignCNN, CLASS_NAMES, eval_transform

parser = argparse.ArgumentParser(description="Predict a traffic sign from an image")
parser.add_argument("image", help="Path to the image file")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN().to(device)
model.load_state_dict(torch.load("checkpoints/traffic_sign_cnn.pth", map_location=device))
model.eval()

img = Image.open(args.image).convert("RGB")
x = eval_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

top_p, top_i = probs.topk(3)
print(f"\nImage: {args.image}")
for rank, (p, i) in enumerate(zip(top_p, top_i), start=1):
    print(f"  {rank}. {CLASS_NAMES[i.item()]}  ({p.item() * 100:.1f}%)")