# app/services/face_detector_id.py

"""
YuNet Face Detector - Optimized for KYC verification
Handles face detection from ID documents and selfies
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="test_yunet.log")





class FaceDetectionResult:
    """Encapsulates face detection result"""
    
    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        landmarks: np.ndarray,
        face_crop: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.landmarks = landmarks  # 5 points: [right_eye, left_eye, nose, right_mouth, left_mouth]
        self.face_crop = face_crop
    
    def to_dict(self) -> dict:
        """Convert to serializable dict"""
        return {
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "landmarks": self.landmarks.tolist(),
            "has_crop": self.face_crop is not None
        }


class YuNetFaceDetector:
    """
    YuNet face detector using OpenCV's FaceDetectorYN.
    Optimized for:
    - ID document face detection (single face, may be small/rotated)
    - Selfie face detection (single face, centered, good quality)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3
    ):
        """
        Initialize YuNet detector.

        Args:
            model_path: Path to yunet.onnx. If None, uses config.
            conf_threshold: Confidence threshold (0.0-1.0). Lower = more lenient
            nms_threshold: Non-maximum suppression threshold
        """
        if model_path is None:
            models_dir = Path(config.get("paths", "models_dir", default="models"))
            model_file = config.get("models", "face_detection", "local_file", default="yunet.onnx")
            model_path = str(models_dir / model_file)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. "
                f"Download from: {config.get('models', 'face_detection', 'url')}"
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._last_size = (0, 0)

        logger.info(f"YuNet initialized: {model_path} (conf={conf_threshold})")

    def _ensure_detector(self, width: int, height: int) -> None:
        """
        Initialize or reinitialize detector if image size changed.
        YuNet requires input size at model creation.
        """
        if self.detector is None or self._last_size != (width, height):
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(width, height),
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000
            )
            self._last_size = (width, height)
            logger.debug(f"Detector initialized for size: {width}x{height}")

    def detect(
        self,
        image: np.ndarray,
        return_largest: bool = True
    ) -> Optional[FaceDetectionResult]:
        """
        Detect face(s) in image.

        Args:
            image: Input image (BGR format)
            return_largest: If True, return only largest face. For KYC, should always be True.

        Returns:
            FaceDetectionResult or None if no face found
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return None

        h, w = image.shape[:2]
        self._ensure_detector(w, h)

        # Detect faces
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            logger.debug("No faces detected")
            return None

        # Parse detections - YuNet format: [x, y, w, h, 5 landmarks (x,y pairs), confidence]
        detections = []
        for face in faces:
            bbox = face[:4].astype(np.int32)
            landmarks = face[4:14].reshape(5, 2).astype(np.int32)
            confidence = float(face[14])
            
            detections.append(FaceDetectionResult(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks
            ))

        if return_largest:
            # Return face with largest area
            largest = max(detections, key=lambda x: x.bbox[2] * x.bbox[3])
            logger.info(f"Face detected: confidence={largest.confidence:.3f}, bbox={largest.bbox.tolist()}")
            return largest
        
        # Sort by confidence (not used in KYC but available)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections[0] if detections else None

    def extract_face(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.2,
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Crop and resize face from image.

        Args:
            image: Source image
            bbox: Bounding box [x, y, w, h]
            padding: Padding ratio around face (0.2 = 20% extra space)
            target_size: Output size (width, height). 112x112 is standard for InsightFace

        Returns:
            Cropped and resized face image
        """
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        # Ensure bounds
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Crop
        face_crop = image[y1:y2, x1:x2]

        # Resize to target size for embedding extraction
        if target_size is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

        return face_crop

    def detect_and_extract(
        self,
        image: np.ndarray,
        padding: float = 0.2,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[FaceDetectionResult]:
        """
        One-shot: detect largest face and extract crop.

        Args:
            image: Input image
            padding: Padding around detected face
            target_size: Resize extracted face to this size

        Returns:
            FaceDetectionResult with face_crop populated, or None
        """
        result = self.detect(image, return_largest=True)
        
        if result is None:
            return None

        # Extract face crop
        result.face_crop = self.extract_face(
            image,
            result.bbox,
            padding=padding,
            target_size=target_size
        )

        return result


# Singleton instance for FastAPI dependency injection
import threading

_detector_instance: Optional[YuNetFaceDetector] = None
_detector_lock = threading.Lock()

def get_face_detector() -> YuNetFaceDetector:
    """Thread-safe singleton getter."""
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = YuNetFaceDetector()
                logger.info("Face detector singleton created")
    return _detector_instance


def reset_detector() -> None:
    """Reset singleton (useful for testing)"""
    global _detector_instance
    _detector_instance = None