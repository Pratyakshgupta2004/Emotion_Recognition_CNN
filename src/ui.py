import cv2
import numpy as np
from datetime import datetime

# ==========================
# Colors (BGR)
# ==========================
BG = (28, 28, 28)
PANEL = (42, 42, 42)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 170, 0)
RED = (0, 0, 255)
GRAY = (80, 80, 80)
CYAN = (255, 255, 0)


# ==========================
# Draw Rounded Rectangle
# ==========================
def rounded_rect(img, pt1, pt2, color, radius=15):

    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(img, (x1 + radius, y1),
                  (x2 - radius, y2), color, -1)

    cv2.rectangle(img, (x1, y1 + radius),
                  (x2, y2 - radius), color, -1)

    cv2.circle(img, (x1 + radius, y1 + radius),
               radius, color, -1)

    cv2.circle(img, (x2 - radius, y1 + radius),
               radius, color, -1)

    cv2.circle(img, (x1 + radius, y2 - radius),
               radius, color, -1)

    cv2.circle(img, (x2 - radius, y2 - radius),
               radius, color, -1)


# ==========================
# Progress Bar
# ==========================
def progress_bar(img, x, y, w, h, value, label):

    value = max(0, min(100, value))

    cv2.putText(img,
                label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2)

    cv2.rectangle(img,
                  (x, y),
                  (x + w, y + h),
                  GRAY,
                  -1)

    fill = int(w * value / 100)

    color = (0, 220, 0)

    if value < 30:
        color = (0, 0, 255)

    elif value < 60:
        color = (0, 255, 255)

    cv2.rectangle(img,
                  (x, y),
                  (x + fill, y + h),
                  color,
                  -1)

    cv2.putText(img,
                f"{value:.1f}%",
                (x + w + 15, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1)


# ==========================
# Dashboard
# ==========================
def draw_dashboard(frame,
                   probs,
                   labels,
                   fps):

    h, w = frame.shape[:2]

    dashboard = np.zeros((h, 400, 3),
                         dtype=np.uint8)

    dashboard[:] = BG

    rounded_rect(
        dashboard,
        (15, 15),
        (385, h - 15),
        PANEL
    )

    cv2.putText(
        dashboard,
        "EMOTION ANALYSIS",
        (55, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        CYAN,
        2
    )

    y = 90

    for label, p in zip(labels, probs):

        progress_bar(
            dashboard,
            30,
            y,
            220,
            20,
            p * 100,
            label
        )

        y += 50

    cv2.putText(
        dashboard,
        f"FPS : {fps:.1f}",
        (30, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN,
        2
    )

    current = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cv2.putText(
        dashboard,
        current,
        (30, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    return np.hstack((frame, dashboard))