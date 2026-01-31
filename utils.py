import cv2
import numpy as np
from shapely.geometry import Polygon

def mask_to_polygon(mask, parking_class=1):
    """
    Convert segmentation class mask to Shapely polygon
    parking_class = class index representing parking area
    """

    # mask is class-index array (0,1,2)
    binary_mask = (mask == parking_class).astype(np.uint8) * 255

    if binary_mask.sum() == 0:
        return None

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()

    if len(pts) < 3:
        return None

    return Polygon(pts)
