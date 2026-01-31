import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from geometry.bev_mapping import bbox_to_bev_polygon
from geometry.slot_generation import compute_free_space, generate_slots
from geometry.path_planning import build_grid, astar

def process_frame(
    vehicle_bboxes,
    parking_area_polygon,
    H,
    car_len_px,
    car_wid_px,
    entrance
):
    vehicle_polygons = [
        bbox_to_bev_polygon(bbox, H)
        for bbox in vehicle_bboxes
    ]

    free_space = compute_free_space(
        parking_area_polygon,
        vehicle_polygons
    )

    slots = generate_slots(
        free_space,
        car_len_px,
        car_wid_px
    )

    grid = build_grid(600, vehicle_polygons)
    target = slots[0].centroid
    path = astar(
        grid,
        entrance,
        (int(target.x), int(target.y))
    )

    return slots, path
if __name__ == "__main__":
    print("process_frame module loaded successfully")
if __name__ == "__main__":
    from shapely.geometry import Polygon
    import numpy as np

    # Dummy parking area
    parking_area = Polygon([
        (50, 50), (550, 50), (550, 550), (50, 550)
    ])

    # Dummy vehicle bounding box (image coords)
    vehicle_bboxes = [
        (200, 200, 260, 260)
    ]

    # Dummy homography (identity for test)
    H = np.eye(3)

    # Dummy car size
    car_len_px = 60
    car_wid_px = 30

    slots, path = process_frame(
        vehicle_bboxes,
        parking_area,
        H,
        car_len_px,
        car_wid_px,
        entrance=(300, 10)
    )

    print("Slots generated:", len(slots))
    print("Path length:", len(path))
