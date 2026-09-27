import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_hands(self, frame):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )
        return self.detector.detect(image)
