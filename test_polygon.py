import cv2
from utils import mask_to_polygon

mask = cv2.imread("parking_decision.png", 0)
poly = mask_to_polygon(mask)

print("Polygon created:", poly is not None)
print("Polygon area:", poly.area if poly else None)
