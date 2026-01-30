import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from model import UNet

# Load model
model = UNet(num_classes=3)
model.load_state_dict(torch.load("parking_model.pth", map_location="cpu"))
model.eval()

# Load image
img = Image.open("dataset/images/4.jpg").convert("RGB")
img = F.resize(img, (256, 256))
x = F.to_tensor(img).unsqueeze(0)

# Predict
with torch.no_grad():
    pred = model(x).argmax(1).squeeze().numpy()

# Convert class → color
out = np.zeros((256, 256, 3), dtype=np.uint8)
out[pred == 1] = [0, 255, 0]   # parkable
out[pred == 2] = [0, 0, 255]   # path
out[pred == 0] = [0, 0, 0]     # obstacle

Image.fromarray(out).save("prediction.png")
print("Saved prediction.png")
