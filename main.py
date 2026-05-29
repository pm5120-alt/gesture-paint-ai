import cv2
import numpy as np
import time
from hand_tracker import HandTracker

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

tracker = HandTracker()

success, frame = cap.read()

frame = cv2.flip(frame, 1)

h, w, c = frame.shape

# =========================
# CANVAS
# =========================

canvas = np.zeros((h, w, 3), dtype=np.uint8)

# =========================
# DRAW SETTINGS
# =========================

draw_color = (0, 0, 255)

brush_thickness = 5

# DRAW HISTORY
prev_draw_x = 0
prev_draw_y = 0

# CURSOR SMOOTHING
prev_cursor_x = 0
prev_cursor_y = 0

smoothening = 4

# FPS
prev_time = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # =========================
    # UI BAR
    # =========================

    cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)

    # RED
    cv2.rectangle(frame, (10, 15), (110, 75), (0, 0, 255), -1)
    cv2.putText(frame, "RED", (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # BLUE
    cv2.rectangle(frame, (130, 15), (230, 75), (255, 0, 0), -1)
    cv2.putText(frame, "BLUE", (140, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # GREEN
    cv2.rectangle(frame, (250, 15), (350, 75), (0, 255, 0), -1)
    cv2.putText(frame, "GREEN", (255, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # ERASER
    cv2.rectangle(frame, (370, 15), (500, 75), (0, 0, 0), -1)
    cv2.putText(frame, "ERASER", (385, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # CLEAR
    cv2.rectangle(frame, (520, 15), (650, 75), (100, 100, 100), -1)
    cv2.putText(frame, "CLEAR", (545, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # =========================
    # HAND DETECTION
    # =========================

    results = tracker.detect_hands(frame)

    if results.hand_landmarks:

        for hand in results.hand_landmarks:

            # LANDMARKS

            index_tip = hand[8]
            index_joint = hand[6]

            middle_tip = hand[12]
            middle_joint = hand[10]

            ring_tip = hand[16]
            ring_joint = hand[14]

            pinky_tip = hand[20]
            pinky_joint = hand[18]

            # =========================
            # STRICT POINTING GESTURE
            # =========================

            index_up = index_tip.y < index_joint.y

            middle_down = middle_tip.y > middle_joint.y
            ring_down = ring_tip.y > ring_joint.y
            pinky_down = pinky_tip.y > pinky_joint.y

            pointing = (
                index_up and
                middle_down and
                ring_down and
                pinky_down
            )

            # =========================
            # RAW COORDINATES
            # =========================

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # =========================
            # SMOOTH CURSOR
            # =========================

            cursor_x = prev_cursor_x + (x - prev_cursor_x) // smoothening
            cursor_y = prev_cursor_y + (y - prev_cursor_y) // smoothening

            prev_cursor_x = cursor_x
            prev_cursor_y = cursor_y

            # POINTER
            cv2.circle(
                frame,
                (cursor_x, cursor_y),
                12,
                (0, 255, 255),
                -1
            )

            # =========================
            # BUTTON INTERACTION
            # =========================

            if cursor_y < 90:

                if 10 < cursor_x < 110:
                    draw_color = (0, 0, 255)

                elif 130 < cursor_x < 230:
                    draw_color = (255, 0, 0)

                elif 250 < cursor_x < 350:
                    draw_color = (0, 255, 0)

                elif 370 < cursor_x < 500:
                    draw_color = (0, 0, 0)

                elif 520 < cursor_x < 650:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # =========================
            # DRAWING
            # =========================

            elif pointing:

                if prev_draw_x == 0 and prev_draw_y == 0:
                    prev_draw_x, prev_draw_y = cursor_x, cursor_y

                cv2.line(
                    canvas,
                    (prev_draw_x, prev_draw_y),
                    (cursor_x, cursor_y),
                    draw_color,
                    brush_thickness
                )

                prev_draw_x, prev_draw_y = cursor_x, cursor_y

            else:

                prev_draw_x, prev_draw_y = 0, 0

    else:

        prev_draw_x, prev_draw_y = 0, 0

    # =========================
    # MERGE
    # =========================

    combined = cv2.add(frame, canvas)

    # =========================
    # FPS
    # =========================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        combined,
        f"FPS: {int(fps)}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # SHOW
    cv2.imshow("AI Virtual Paint", combined)

    # ESC EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

# RELEASE
cap.release()

cv2.destroyAllWindows()