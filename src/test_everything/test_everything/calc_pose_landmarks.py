import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

class PoseLandmarkerSolver():
    def __init__(self, callback, model_path):
        BaseOptions = mp.tasks.BaseOptions
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=callback,
            num_poses=1
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._context = self.landmarker.__enter__()

    def solve_async(self, image):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self):
        if self.landmarker:
            self.landmarker.__exit__(None, None, None)