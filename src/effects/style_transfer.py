import os
import cv2
import numpy as np
import onnxruntime as ort

class StyleTransferEngine:
    def __init__(self, model_dir, model_files):
        self.model_dir = model_dir
        self.model_files = model_files
        self.current_idx = 0
        self.session = None
        self._load_current_model()

    def _load_current_model(self):
        model_path = os.path.join(self.model_dir, self.model_files[self.current_idx])
        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.session = None

    def next_style(self):
        self.current_idx = (self.current_idx + 1) % len(self.model_files)
        self._load_current_model()

    def apply_style(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            return image_bgr

        h, w = image_bgr.shape[:2]
        
        # Pre-process: Resize to network size (e.g., 512x512 or 224x224), BGR -> RGB, NCHW conversion
        input_blob = cv2.resize(image_bgr, (512, 512))
        input_blob = cv2.cvtColor(input_blob, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]  # Shape: (1, 3, 512, 512)

        # Run ONNX inference
        output = self.session.run(None, {self.input_name: input_blob})[0]

        # Post-process: Reshape, clip RGB values, convert back to BGR
        output_img = output[0].clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        return cv2.resize(output_img, (w, h))