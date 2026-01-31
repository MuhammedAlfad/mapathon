import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from predict import predict_image
from utils import mask_to_polygon
from geometry.bev_mapping import meters_per_pixel
from pipeline.process_frame import process_frame


# -------- CONFIG --------
VIDEO_PATH = "CCTV Camera Based Parking Guidance for Rooftop Parking Lot - TIS APGS (1080p, h264).mp4"

REAL_PARKING_WIDTH_M = 20.0
ENTRANCE_POINT = (300, 10)
# ------------------------


cap = cv2.VideoCapture(VIDEO_PATH)
print("Video opened:", cap.isOpened())

# ---- TEMP homography (identity) ----
H = np.eye(3)

# ---- scaling ----
METERS_PER_PIXEL = meters_per_pixel(REAL_PARKING_WIDTH_M)
CAR_LENGTH_PX = 4.5 / METERS_PER_PIXEL
CAR_WIDTH_PX  = 2.0 / METERS_PER_PIXEL


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Segmentation
    mask = predict_image(frame)

    # 2️⃣ Mask → polygon
    parking_area_polygon = mask_to_polygon(mask)
    if parking_area_polygon is None:
        continue

    # 3️⃣ TEMP: no vehicle detection yet
    vehicle_bboxes = []

    # 4️⃣ Parking logic
    slots, path = process_frame(
        vehicle_bboxes,
        parking_area_polygon,
        H,
        CAR_LENGTH_PX,
        CAR_WIDTH_PX,
        entrance=ENTRANCE_POINT
    )

    print("Slots:", len(slots), "Path length:", len(path))

    # 5️⃣ Display raw frame for now
    cv2.imshow("Smart Parking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
