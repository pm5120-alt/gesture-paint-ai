import cv2
import av
import numpy as np
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from hand_tracker import HandTracker

st.title("🎨 AI Virtual Paint")

tracker = HandTracker()

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        h, w, c = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        results = tracker.detect_hands(img)

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

                # STRICT POINTING DETECTION
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

                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                # POINTER
                cv2.circle(img, (x, y), 10, (0,255,255), -1)

                # DRAWING
                if pointing:

                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = x, y

                    cv2.line(
                        self.canvas,
                        (self.prev_x, self.prev_y),
                        (x, y),
                        (0,0,255),
                        5
                    )

                    self.prev_x, self.prev_y = x, y

                else:

                    self.prev_x, self.prev_y = 0, 0

        combined = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(
            combined,
            format="bgr24"
        )

webrtc_streamer(
    key="paint-app",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)