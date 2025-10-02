# app/services/face_detector.py

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from configs.config import config

logger = logging.getLogger(__name__)


class YuNetFaceDetector:
    """
    YuNet face detector for extracting faces from ID documents and selfies.
    Uses OpenCV's FaceDetectorYN with ONNX model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        """
        Initialize YuNet face detector.

        Args:
            model_path: Path to yunet.onnx model. If None, uses config.
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            nms_threshold: NMS IoU threshold for filtering overlapping boxes
            top_k: Keep top K detections before NMS
        """
        if model_path is None:
            models_dir = Path(config.get("paths", "models_dir", default="models"))
            model_file = config.get("models", "face_detection", "local_file", default="yunet.onnx")
            model_path = str(models_dir / model_file)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"YuNet model not found at {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None

        logger.info(f"Initialized YuNetFaceDetector with model: {model_path}")

    def _initialize_detector(self, img_width: int, img_height: int) -> None:
        """
        Initialize or reinitialize detector with image dimensions.
        YuNet requires input size at initialization.
        """
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(img_width, img_height),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect(
        self,
        image: np.ndarray,
        return_largest: bool = True,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR format, numpy array)
            return_largest: If True, return only the largest face. If False, return all faces.

        Returns:
            If return_largest=True:
                Tuple of (face_bbox, confidence, landmarks) or None if no face found
                - face_bbox: [x, y, w, h]
                - confidence: detection confidence score
                - landmarks: 5 facial landmarks [right_eye, left_eye, nose, right_mouth, left_mouth]
                  each landmark is (x, y)
            If return_largest=False:
                List of tuples, each containing (face_bbox, confidence, landmarks)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detector")
            return None if return_largest else []

        h, w = image.shape[:2]

        # Initialize or reinitialize if size changed
        if self.detector is None:
            self._initialize_detector(w, h)
        else:
            # Check if we need to reinitialize due to size change
            current_size = self.detector.getInputSize()
            if current_size[0] != w or current_size[1] != h:
                self._initialize_detector(w, h)

        # Detect faces
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            logger.debug("No faces detected")
            return None if return_largest else []

        logger.info(f"Detected {len(faces)} face(s)")

        # Parse detections
        results = []
        for face in faces:
            # YuNet output format: [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, conf]
            bbox = face[:4].astype(np.int32)  # [x, y, w, h]
            landmarks = face[4:14].reshape(5, 2).astype(np.int32)  # 5 landmarks
            confidence = float(face[14])

            results.append((bbox, confidence, landmarks))

        if return_largest:
            # Return face with largest bounding box area
            largest = max(results, key=lambda x: x[0][2] * x[0][3])  # w * h
            return largest
        else:
            # Sort by confidence descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def extract_face(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.2,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract and crop face from image using bounding box.

        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x, y, w, h]
            padding: Padding ratio to add around bbox (0.2 = 20% padding)
            target_size: Optional (width, height) to resize extracted face

        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Crop face
        face_crop = image[y1:y2, x1:x2]

        # Resize if target size specified
        if target_size is not None:
            face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

        return face_crop

    def detect_and_extract(
        self,
        image: np.ndarray,
        padding: float = 0.2,
        target_size: Optional[Tuple[int, int]] = (112, 112),
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Convenience method: detect largest face and extract it.

        Args:
            image: Input image (BGR format)
            padding: Padding around detected face
            target_size: Resize extracted face to this size (for embeddings)

        Returns:
            Tuple of (face_image, confidence, landmarks) or None if no face found
        """
        detection = self.detect(image, return_largest=True)

        if detection is None:
            return None

        bbox, confidence, landmarks = detection
        face_img = self.extract_face(image, bbox, padding=padding, target_size=target_size)

        return face_img, confidence, landmarks

    def visualize_detection(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        confidence: float,
        landmarks: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding box and landmarks on image for visualization.

        Args:
            image: Input image
            bbox: [x, y, w, h]
            confidence: Detection confidence
            landmarks: Optional 5x2 array of facial landmarks
            color: BGR color for drawing
            thickness: Line thickness

        Returns:
            Image with annotations
        """
        img_vis = image.copy()
        x, y, w, h = bbox

        # Draw bounding box
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), color, thickness)

        # Draw confidence
        label = f"{confidence:.2f}"
        cv2.putText(
            img_vis,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

        # Draw landmarks
        if landmarks is not None:
            for lm in landmarks:
                cv2.circle(img_vis, tuple(lm), 2, (0, 0, 255), -1)

        return img_vis


# Singleton instance for reuse across requests
_detector_instance: Optional[YuNetFaceDetector] = None


def get_face_detector() -> YuNetFaceDetector:
    """
    Get or create singleton YuNet detector instance.
    Useful for FastAPI dependency injection.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YuNetFaceDetector()
    return _detector_instance