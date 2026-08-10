import time
import cv2
import mediapipe as mp
import numpy as np


class MediaPipeHandTracker:

  def __init__(self, model_path: str, num_hands: int = 1):
    self.model_path = model_path
    self.num_hands = num_hands
    self.latest_result = None

    # Setup MediaPipe Tasks options
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self._result_callback,
        num_hands=self.num_hands,
    )
    self.landmarker = HandLandmarker.create_from_options(options)

  def _result_callback(
      self, result: mp.tasks.vision.HandLandmarkerResult, output_image, ts: int
  ):
    self.latest_result = result

  def process_frame(self, frame_bgr: np.ndarray):
    """Feeds OpenCV frame into MediaPipe pipeline asynchronously."""
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    ts_ms = int(time.time() * 1000)
    self.landmarker.detect_async(mp_image, ts_ms)

  def get_landmarks_numpy(self) -> list[np.ndarray]:
    """Returns a list of (21, 3) NumPy arrays for all detected hands."""
    if not self.latest_result or not self.latest_result.hand_landmarks:
      return []

    hands_np = []
    for hand in self.latest_result.hand_landmarks:
      coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
      hands_np.append(coords)
    return hands_np

  def close(self):
    self.landmarker.close()