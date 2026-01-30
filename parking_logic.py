import cv2
import numpy as np
from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================

MIN_CAR_AREA_PX = 6000     # pixel-based threshold
MIN_CAR_WIDTH_PX = 60     # approx car width
MIN_CAR_LENGTH_PX = 120   # approx car length
KERNEL_SIZE = 7

print("Minimum car area threshold (pixels):", MIN_CAR_AREA_PX)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def orientation_check(mask):
    """
    Check if a rotated rectangle of car size fits.
    Returns list of valid rectangles.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_rects = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CAR_AREA_PX:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), angle = rect

        w, h = max(w, h), min(w, h)

        if w >= MIN_CAR_LENGTH_PX and h >= MIN_CAR_WIDTH_PX:
            valid_rects.append((rect, cv2.contourArea(cnt)))

    return valid_rects


def path_check(green_mask, blue_mask):
    """
    Returns True if green region touches blue path
    """
    dilated = cv2.dilate(green_mask, np.ones((5, 5), np.uint8))
    overlap = cv2.bitwise_and(dilated, blue_mask)
    return np.any(overlap > 0)


def score_spot(area, centroid_y, angle, img_h):
    """
    Rank candidate spots
    """
    size_score = area / 1000
    entry_score = (img_h - centroid_y) / img_h
    angle_penalty = abs(angle) / 90
    return size_score + entry_score - angle_penalty


# =========================================================
# LOAD SEGMENTATION RESULT
# =========================================================

pred_img = Image.open("prediction.png").convert("RGB")
pred = np.array(pred_img)
h, w, _ = pred.shape

# =========================================================
# EXTRACT MASKS
# =========================================================

green_mask = (
    (pred[:, :, 1] > 200) &
    (pred[:, :, 0] < 50) &
    (pred[:, :, 2] < 50)
).astype("uint8") * 255

blue_mask = (
    (pred[:, :, 2] > 200) &
    (pred[:, :, 0] < 50) &
    (pred[:, :, 1] < 50)
).astype("uint8") * 255

# =========================================================
# CLEAN MASKS
# =========================================================

kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# =========================================================
# ORIENTATION + PATH CHECK
# =========================================================

candidates = orientation_check(green_mask)
output = pred.copy()
best_score = -1
best_rect = None

for rect, area in candidates:
    ((cx, cy), (rw, rh), angle) = rect

    if not path_check(green_mask, blue_mask):
        continue

    score = score_spot(area, cy, angle, h)

    if score > best_score:
        best_score = score
        best_rect = rect

# =========================================================
# VISUALIZATION & DECISION
# =========================================================

if best_rect is not None:
    box = cv2.boxPoints(best_rect)
    box = box.astype(int)


    cv2.drawContours(output, [box], 0, (0, 255, 0), 3)
    cv2.putText(
        output,
        "BEST PARKING SPOT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    print("\n🚗 RESULT: A CAR CAN FIT (BEST SPOT SELECTED).")

else:
    print("\n🚫 RESULT: NO SUITABLE PARKING SPACE FOUND.")

Image.fromarray(output).save("parking_decision.png")
print("Saved parking_decision.png")

# =========================================================
# OPTIONAL: CALIBRATION (RUN SEPARATELY IF NEEDED)
# =========================================================

"""
UNCOMMENT BELOW TO CALIBRATE PIXELS → METERS

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(points)

img = cv2.imread("dataset/images/44.jpg")
cv2.imshow("Click two points 1 meter apart", img)
cv2.setMouseCallback("Click two points 1 meter apart", click)

cv2.waitKey(0)
cv2.destroyAllWindows()

(x1, y1), (x2, y2) = points
dist_px = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
print("Pixels per meter:", dist_px)
"""
