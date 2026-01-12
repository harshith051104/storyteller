import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class EmotionEngine:
    def __init__(self, model_path="face_landmarker.task"):
        """
        Initializes the MediaPipe FaceLandmarker with Blendshapes enabled.
        """
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model file '{model_path}' not found. Emotion detection disabled.")
                self.detector = None
                return

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("EmotionEngine (FaceLandmarker) initialized successfully.")
        except Exception as e:
            print(f"Error initializing EmotionEngine: {e}")
            self.detector = None

    def detect_emotion(self, image):
        """
        Detects emotion from a numpy image (RGB) using Blendshapes.
        Returns: { "emotion": label, "confidence": float }
        """
        if self.detector is None or image is None:
            return {"emotion": "neutral", "confidence": 0.0}

        try:
            # Create MP Image from Numpy
            # Gradio provides RGB numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Detect
            detection_result = self.detector.detect(mp_image)

            if not detection_result.face_blendshapes:
                return {"emotion": "neutral", "confidence": 0.0}

            # Extract Blendshapes (list of categories)
            # There is 1 face, so index 0
            blendshapes = detection_result.face_blendshapes[0]
            
            # Map blendshapes to dict for easy access
            bs = {b.category_name: b.score for b in blendshapes}

            # Heuristics for Basic Emotions based on ARKit Blendshapes
            # Scores are 0.0 to 1.0
            
            scores = {
                "happy": (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2,
                "surprise": (bs.get('browInnerUp', 0) + bs.get('jawOpen', 0)) / 2,
                "angry": (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2,
                "sad": (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0) + bs.get('browInnerUp', 0)) / 3,
                "fear": (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0) + bs.get('mouthStretchLeft', 0)) / 3
            }

            # Find max score
            best_emotion = max(scores, key=scores.get)
            best_score = scores[best_emotion]

            # Thresholding
            if best_score < 0.3: # If signals are weak, default to neutral
                return {"emotion": "neutral", "confidence": round(1.0 - best_score, 2)}
            
            print(f"[EmotionEngine] Detected emotion: {best_emotion} (confidence={best_score:.2f})")
            
            return {
                "emotion": best_emotion,
                "confidence": round(best_score, 2)
            }

        except Exception as e:
            print(f"Emotion detection runtime error: {e}")
            return {"emotion": "neutral", "confidence": 0.0}
