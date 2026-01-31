import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from predict import predict_image
from utils import mask_to_polygon
from geometry.bev_mapping import meters_per_pixel
from pipeline.process_frame import process_frame


def detect_vehicles(frame):
    """
    Simple vehicle detection using contour analysis.
    Adjust thresholds based on video.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 50000:  # Adjust min/max area for vehicles
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, x + w, y + h))
    return bboxes


# -------- CONFIG --------
VIDEO_PATH = r"C:\Users\USER\Desktop\parking_area_model\CCTV Camera Based Parking Guidance for Rooftop Parking Lot - TIS APGS (1080p, h264).mp4"

REAL_PARKING_WIDTH_M = 20.0
GRID_SIZE = 300
ENTRANCE_POINT = (150, 150)   # center of BEV grid
FRAME_SKIP = 5                # run segmentation every N frames
# ------------------------


cap = cv2.VideoCapture(VIDEO_PATH)
print("Video opened:", cap.isOpened())

# ---- TEMP homography (identity) ----
H = np.eye(3)

# ---- scaling ----
METERS_PER_PIXEL = meters_per_pixel(REAL_PARKING_WIDTH_M)
CAR_LENGTH_PX = int(4.5 / METERS_PER_PIXEL)
CAR_WIDTH_PX  = int(2.0 / METERS_PER_PIXEL)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    # 1️⃣ Segmentation
    mask = predict_image(frame)

    # Visualize mask
    mask_vis = (mask == 1).astype(np.uint8) * 255
    cv2.imshow("Parking Mask", mask_vis)

    # 2️⃣ Mask → polygon (scaled to frame)
    parking_area_polygon = mask_to_polygon(
        mask,
        frame_shape=frame.shape
    )

    if parking_area_polygon is None:
        continue

    # Draw parking area polygon
    x, y = parking_area_polygon.exterior.xy
    pts = np.array(list(zip(x, y)), np.int32)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)  # Blue for parking area

    # 3️⃣ Vehicle detection
    vehicle_bboxes = detect_vehicles(frame)

    # Draw detected vehicles
    for bbox in vehicle_bboxes:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)  # Red for vehicles

    # 4️⃣ Parking logic
    slots, path = process_frame(
        vehicle_bboxes,
        parking_area_polygon,
        H,
        CAR_LENGTH_PX,
        CAR_WIDTH_PX,
        entrance=ENTRANCE_POINT,
        grid_size=GRID_SIZE
    )

    print(f"Frame {frame_id}: Slots: {len(slots)}  Vehicles: {len(vehicle_bboxes)}  Path length: {len(path)}")

    # 5️⃣ Visualization (slots in green)
    for slot in slots[:10]:  # draw first 10 slots
        x, y = slot.exterior.xy
        pts = np.array(list(zip(x, y)), np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # Draw path if exists
    if path:
        for i in range(len(path) - 1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i+1][0]), int(path[i+1][1]))
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # Yellow for path

    cv2.imshow("Smart Parking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
