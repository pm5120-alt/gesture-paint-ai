import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(self):

        model_path = "hand_landmarker.task"

        BaseOptions = python.BaseOptions

        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            running_mode=VisionRunningMode.IMAGE
        )

        self.detector = HandLandmarker.create_from_options(
            options
        )

    def detect_hands(self, frame):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        results = self.detector.detect(mp_image)

        return results