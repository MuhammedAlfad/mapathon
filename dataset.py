import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class ParkingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[idx].replace(".jpg", ".png")
        )

        # Load image & mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Convert mask colors → class labels
        mask = self.rgb_to_class(mask)

        # Resize image
        image = F.resize(image, (256, 256))

        # Resize mask (nearest only)
        mask = torch.tensor(mask, dtype=torch.long)
        mask = F.resize(
            mask.unsqueeze(0),
            (256, 256),
            interpolation=F.InterpolationMode.NEAREST
        ).squeeze(0)

        # Convert image to tensor
        image = F.to_tensor(image)

        return image, mask

    def rgb_to_class(self, mask):
        """
        Converts colored mask to class IDs:
        0 = obstacle
        1 = parkable (green-ish)
        2 = path (blue-ish)
        """
        mask = np.array(mask)
        label = np.zeros(mask.shape[:2], dtype=np.uint8)

        r = mask[:, :, 0]
        g = mask[:, :, 1]
        b = mask[:, :, 2]

        # 🟩 Parkable (green dominant)
        green = (g > 140) & (g > r + 30) & (g > b + 30)
        label[green] = 1

        # 🟦 Path (blue dominant)
        blue = (b > 140) & (b > r + 30) & (b > g + 30)
        label[blue] = 2

        # Everything else = obstacle (0)
        return label
