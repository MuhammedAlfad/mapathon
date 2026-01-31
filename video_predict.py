import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from model import UNet

# =========================================
# LOAD MODEL
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(num_classes=3)
model.load_state_dict(torch.load("parking_model.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded on", device)

# =========================================
# VIDEO SOURCE
# =========================================

# Option 1: video file
VIDEO_PATH = r"C:\Users\USER\Desktop\parking_area_model\CCTV Camera Based Parking Guidance for Rooftop Parking Lot - TIS APGS (1080p, h264).mp4"

# Option 2: webcam (uncomment below)
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

# =========================================
# PROCESS VIDEO
# =========================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Resize for model
    img = cv2.resize(frame, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(x).argmax(1).squeeze().cpu().numpy().astype("uint8")

    # =========================================
    # COLORIZE SEGMENTATION
    # =========================================

    seg = np.zeros((256, 256, 3), dtype=np.uint8)

    seg[pred == 0] = [0, 0, 0]       # obstacle
    seg[pred == 1] = [0, 255, 0]     # parkable
    seg[pred == 2] = [0, 0, 255]     # path

    # Resize back to original size
    seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # =========================================
    # OVERLAY
    # =========================================

    overlay = frame.copy()
    alpha = 0.5

    mask = seg.sum(axis=2) > 0
    overlay[mask] = (
        overlay[mask] * (1 - alpha) +
        seg[mask] * alpha
    ).astype("uint8")

    # =========================================
    # DISPLAY
    # =========================================

    cv2.imshow("Parking Segmentation (Video)", overlay)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
