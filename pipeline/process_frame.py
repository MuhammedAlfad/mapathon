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
    entrance,
    grid_size=300
):
    """
    Process a single frame:
    - vehicle bounding boxes -> BEV polygons
    - compute free space
    - generate parking slots
    - optionally run path planning
    """

    # 1️⃣ Convert vehicle boxes → BEV polygons
    vehicle_polygons = [
        bbox_to_bev_polygon(bbox, H)
        for bbox in vehicle_bboxes
    ]

    # 2️⃣ Compute free space
    free_space = compute_free_space(
        parking_area_polygon,
        vehicle_polygons
    )

    # 3️⃣ Generate slots
    slots = generate_slots(
        free_space,
        car_len_px,
        car_wid_px
    )

    # Safety: no slots
    if not slots:
        return [], []

    # 🚑 IMPORTANT: if no vehicles, skip path planning
    if not vehicle_polygons:
        return slots, []

    # 4️⃣ Build grid + path planning
    grid = build_grid(grid_size, vehicle_polygons)

    target = slots[0].centroid
    goal = (int(target.x), int(target.y))

    path = astar(
        grid,
        entrance,
        goal,
        max_steps=15000
    )

    return slots, path


# ---------------- TEST BLOCK ----------------
if __name__ == "__main__":
    from shapely.geometry import Polygon
    import numpy as np

    print("Testing process_frame()")

    parking_area = Polygon([
        (50, 50), (250, 50), (250, 250), (50, 250)
    ])

    vehicle_bboxes = []  # no vehicles
    H = np.eye(3)

    slots, path = process_frame(
        vehicle_bboxes,
        parking_area,
        H,
        car_len_px=40,
        car_wid_px=20,
        entrance=(150, 150)
    )

    print("Slots generated:", len(slots))
    print("Path length:", len(path))
