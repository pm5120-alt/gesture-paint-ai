import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from hand_tracker import HandTracker


st.title("AI Virtual Paint")

tracker = HandTracker()


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.previous_x = 0
        self.previous_y = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        height, width, _ = image.shape

        if self.canvas is None:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        results = tracker.detect_hands(image)

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

                cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

                if pointing:
                    if self.previous_x == 0 and self.previous_y == 0:
                        self.previous_x = x
                        self.previous_y = y

                    cv2.line(
                        self.canvas,
                        (self.previous_x, self.previous_y),
                        (x, y),
                        (0, 0, 255),
                        5
                    )

                    self.previous_x = x
                    self.previous_y = y
                else:
                    self.previous_x = 0
                    self.previous_y = 0

        combined = cv2.add(image, self.canvas)

        return av.VideoFrame.from_ndarray(
            combined,
            format="bgr24"
        )


webrtc_streamer(
    key="paint-app",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
