import cv2
import numpy as np
from shapely.geometry import Polygon

BEV_SIZE = 600

def compute_homography(image_shape, src_pts):
    h, w = image_shape[:2]
    dst_pts = np.float32([
        [0, 0],
        [BEV_SIZE, 0],
        [BEV_SIZE, BEV_SIZE],
        [0, BEV_SIZE]
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def bbox_to_bev_polygon(bbox, H):
    x1, y1, x2, y2 = bbox
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32).reshape(-1,1,2)

    bev_pts = cv2.perspectiveTransform(corners, H)
    return Polygon(bev_pts.reshape(-1,2))

def meters_per_pixel(real_width_m, bev_size=BEV_SIZE):
    return real_width_m / bev_size
