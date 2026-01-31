import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
from pipeline.process_frame import process_frame
from geometry.bev_mapping import meters_per_pixel

cap = cv2.VideoCapture("test_video.mp4")

METERS_PER_PIXEL = meters_per_pixel(20.0)
CAR_LENGTH_PX = 4.5 / METERS_PER_PIXEL
CAR_WIDTH_PX  = 2.0 / METERS_PER_PIXEL

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Existing segmentation logic
    parking_area_polygon = get_parking_polygon(frame)

    # Existing vehicle detection
    vehicle_bboxes = detect_vehicles(frame)

    slots, path = process_frame(
        vehicle_bboxes,
        parking_area_polygon,
        H,
        CAR_LENGTH_PX,
        CAR_WIDTH_PX,
        entrance=(300, 10)
    )

    visualize(frame, slots, path)
    cv2.imshow("Smart Parking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
