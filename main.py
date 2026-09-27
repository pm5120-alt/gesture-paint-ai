import time

import cv2
import numpy as np

from hand_tracker import HandTracker


cap = cv2.VideoCapture(0)
tracker = HandTracker()

success, frame = cap.read()
if not success:
    cap.release()
    raise SystemExit("Could not open the camera.")

frame = cv2.flip(frame, 1)
height, width, _ = frame.shape

canvas = np.zeros((height, width, 3), dtype=np.uint8)

draw_color = (0, 0, 255)
brush_size = 5

previous_draw_x = 0
previous_draw_y = 0

previous_cursor_x = 0
previous_cursor_y = 0
smoothening = 4

previous_time = time.time()


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (0, 0), (width, 90), (40, 40, 40), -1)

    # Color buttons
    cv2.rectangle(frame, (10, 15), (110, 75), (0, 0, 255), -1)
    cv2.putText(frame, "RED", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (130, 15), (230, 75), (255, 0, 0), -1)
    cv2.putText(frame, "BLUE", (140, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (250, 15), (350, 75), (0, 255, 0), -1)
    cv2.putText(frame, "GREEN", (255, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (370, 15), (500, 75), (0, 0, 0), -1)
    cv2.putText(frame, "ERASER", (385, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (520, 15), (650, 75), (100, 100, 100), -1)
    cv2.putText(frame, "CLEAR", (545, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    results = tracker.detect_hands(frame)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            index_tip = hand[8]
            index_joint = hand[6]
            middle_tip = hand[12]
            middle_joint = hand[10]
            ring_tip = hand[16]
            ring_joint = hand[14]
            pinky_tip = hand[20]
            pinky_joint = hand[18]

            index_up = index_tip.y < index_joint.y
            middle_down = middle_tip.y > middle_joint.y
            ring_down = ring_tip.y > ring_joint.y
            pinky_down = pinky_tip.y > pinky_joint.y

            pointing = (
                index_up
                and middle_down
                and ring_down
                and pinky_down
            )

            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            cursor_x = previous_cursor_x + (x - previous_cursor_x) // smoothening
            cursor_y = previous_cursor_y + (y - previous_cursor_y) // smoothening

            previous_cursor_x = cursor_x
            previous_cursor_y = cursor_y

            cv2.circle(frame, (cursor_x, cursor_y), 12, (0, 255, 255), -1)

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
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

            elif pointing:
                if previous_draw_x == 0 and previous_draw_y == 0:
                    previous_draw_x = cursor_x
                    previous_draw_y = cursor_y

                cv2.line(
                    canvas,
                    (previous_draw_x, previous_draw_y),
                    (cursor_x, cursor_y),
                    draw_color,
                    brush_size
                )

                previous_draw_x = cursor_x
                previous_draw_y = cursor_y
            else:
                previous_draw_x = 0
                previous_draw_y = 0

    else:
        previous_draw_x = 0
        previous_draw_y = 0

    combined = cv2.add(frame, canvas)

    current_time = time.time()
    fps = 1 / max(current_time - previous_time, 0.0001)
    previous_time = current_time

    cv2.putText(
        combined,
        f"FPS: {int(fps)}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Virtual Paint", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
