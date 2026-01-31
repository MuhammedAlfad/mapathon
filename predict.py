import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from model import UNet


# -------- Load model ONCE --------
model = UNet(num_classes=3)
model.load_state_dict(torch.load("parking_model.pth", map_location="cpu"))
model.eval()


def predict_image(frame_bgr):
    """
    Takes a BGR OpenCV frame and returns segmentation mask (numpy array)
    """
    # OpenCV → PIL
    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB
    img = F.resize(img, (256, 256))
    x = F.to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).argmax(1).squeeze().cpu().numpy()

    return pred


def save_prediction_image(pred, filename="prediction.png"):
    """
    Optional: save colored segmentation image
    """
    out = np.zeros((256, 256, 3), dtype=np.uint8)
    out[pred == 1] = [0, 255, 0]   # parkable
    out[pred == 2] = [0, 0, 255]   # path
    out[pred == 0] = [0, 0, 0]     # obstacle

    Image.fromarray(out).save(filename)


# -------- Script mode (keeps old behavior) --------
if __name__ == "__main__":
    img = Image.open("dataset/images/4.jpg").convert("RGB")
    img = F.resize(img, (256, 256))
    x = F.to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).argmax(1).squeeze().numpy()

    save_prediction_image(pred)
    print("Saved prediction.png")
