import os
from dataclasses import dataclass

@dataclass
class Config:
    # Camera
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Gesture Thresholds
    SWIPE_HISTORY_FRAMES: int = 12
    SWIPE_MIN_DISTANCE: float = 0.25  # Normalized screen distance
    PINCH_THRESHOLD: float = 0.05
    CLAP_THRESHOLD: float = 0.12
    CLAP_COOLDOWN_FRAMES: int = 30    # Prevent rapid flickering
    
    # Assets
    MODEL_DIR: str = os.path.join("assets", "weights")
    STYLE_MODELS: tuple = ("mosaic.onnx", "candy.onnx", "rain_princess.onnx", "udnie.onnx")