import os
import numpy as np
from PIL import Image

MASK_DIR = "dataset/masks"
OUT_DIR = "dataset/masks_clean"

os.makedirs(OUT_DIR, exist_ok=True)

for name in os.listdir(MASK_DIR):
    if not name.lower().endswith(".png"):
        continue

    mask = np.array(Image.open(os.path.join(MASK_DIR, name)))

    clean = np.zeros_like(mask)

    # Blue channel dominant → path
    blue = mask[:,:,2] > 128
    clean[blue] = [0, 0, 255]

    # Green channel dominant → parkable
    green = mask[:,:,1] > 128
    clean[green] = [0, 255, 0]

    # Everything else → black
    Image.fromarray(clean).save(os.path.join(OUT_DIR, name))

print("Mask cleaning complete")
