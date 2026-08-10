import time

import cv2

from src.trackers.hand_track import MediaPipeHandTracker
from src.utils.guestures import (
    GestureEvent,
    HandGestureEngine,
    HandGestureResult,
    PinchPhase,
)

MODEL_PATH = "assets/weights/hand_landmarker.task"


def draw_hud(frame, result: HandGestureResult):
  pinch = result.pinch
  wave = result.wave

  # Default UI Colors (BGR)
  color = (255, 255, 255)
  status_str = "OPEN / IDLE"

  if pinch.phase == PinchPhase.HOLDING:
    color = (0, 255, 255)  # Yellow for passive holding
    status_str = "HOLDING PINCH (NO SNAP)"
  elif pinch.phase == PinchPhase.CONTACT:
    color = (0, 0, 255)  # Flash RED on contact/snap
    status_str = "SNAP / PINCH CONTACT!"
  elif pinch.phase in (PinchPhase.APPROACHING, PinchPhase.ARMED):
    color = (255, 165, 0)
    status_str = "CLOSING FAST..."

  # Render Text Overlays
  cv2.putText(
      frame,
      f"Context: {status_str}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      color,
      2,
  )
  cv2.putText(
      frame,
      f"Motion: {wave.state.name}",
      (30, 90),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      frame,
      f"Pinch Dist: {pinch.distance:.2f}",
      (30, 130),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (200, 200, 200),
      1,
  )


def main():
  tracker = MediaPipeHandTracker(model_path=MODEL_PATH, num_hands=1)
  engine = HandGestureEngine()
  cap = cv2.VideoCapture(0)

  had_hand = False

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.flip(frame, 1)
    tracker.process_frame(frame)

    hands = tracker.get_landmarks_numpy()
    if hands:
      had_hand = True

      # Process primary hand
      result = engine.process(hands[0], timestamp=time.monotonic())
      draw_hud(frame, result)

      if GestureEvent.PINCH_SNAP in result.events:
        print("Snap!")
      if GestureEvent.HORIZONTAL_WAVE in result.events:
        print("Horizontal wave")
      if GestureEvent.VERTICAL_WAVE in result.events:
        print("Vertical wave")

      # Render keypoints on frame using OpenCV
      h, w, _ = frame.shape
      for pt in hands[0]:
        cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 3, (0, 255, 0), -1)
    elif had_hand:
      # Hand dropped out of frame; clear temporal state so the next
      # detection doesn't inherit stale velocity/phase.
      engine.reset()
      had_hand = False

    cv2.imshow("Contextual Gesture Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  tracker.close()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()