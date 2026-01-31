from shapely.geometry import box

def compute_free_space(parking_area, vehicle_polygons):
    occupied = vehicle_polygons[0]
    for v in vehicle_polygons[1:]:
        occupied = occupied.union(v)
    return parking_area.difference(occupied)

def generate_slots(free_space, car_len_px, car_wid_px, gap=5):
    slots = []
    minx, miny, maxx, maxy = free_space.bounds

    y = miny
    while y + car_wid_px < maxy:
        x = minx
        while x + car_len_px < maxx:
            slot = box(x, y, x + car_len_px, y + car_wid_px)
            if free_space.contains(slot):
                slots.append(slot)
            x += car_len_px + gap
        y += car_wid_px + gap

    return slots
