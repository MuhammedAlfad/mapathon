from PIL import Image
import numpy as np
from dataset import ParkingDataset

ds = ParkingDataset("dataset/images", "dataset/masks")

mask = Image.open("dataset/masks/1.png")  # change name if needed
class_mask = ds.rgb_to_class(mask)

print(np.unique(class_mask))
