import cv2
import numpy as np
from shapely.geometry import Polygon

def mask_to_polygon(mask, parking_class=1, frame_shape=None):
    binary_mask = (mask == parking_class).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()

    if frame_shape is not None:
        h, w = frame_shape[:2]
        pts[:, 0] = pts[:, 0] * (w / mask.shape[1])
        pts[:, 1] = pts[:, 1] * (h / mask.shape[0])

    return Polygon(pts)

