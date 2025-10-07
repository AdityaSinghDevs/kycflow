# app/services/face_matcher.py

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import insightface
from insightface.app import FaceAnalysis

from configs.config import config

logger = logging.getLogger(__name__)


class InsightFaceMatcher:
    """
    InsightFace-based face verification and matching.
    Compares selfie with ID document face using embeddings and cosine similarity.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ctx_id: int = 0,  # 0 for CPU, GPU id for CUDA
        det_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize InsightFace face analysis model.

        Args:
            model_name: Model pack name (e.g., 'buffalo_l', 'buffalo_s'). If None, uses config.
            ctx_id: Context ID - 0 for CPU, GPU id for CUDA (e.g., 0, 1)
            det_size: Detection size for face analysis
        """
        if model_name is None:
            model_name = config.get("models", "face_recognition", "model_name", default="buffalo_l")

        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size

        logger.info(f"Initializing InsightFace with model: {model_name}")

        try:
            # Initialize FaceAnalysis
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
            )
            
            # Prepare model with detection size
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            
            logger.info(f"InsightFace initialized successfully on {'GPU' if ctx_id >= 0 else 'CPU'}")

        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

        # Thresholds for verification
        self.similarity_threshold = 0.4  # Cosine similarity threshold (adjustable)
        self.distance_threshold = 1.0    # Euclidean distance threshold (adjustable)

    def get_embedding(
        self,
        image: np.ndarray,
        return_face_info: bool = False,
    ) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Extract face embedding from image.

        Args:
            image: Input image (BGR format, numpy array)
            return_face_info: If True, also return face detection info (bbox, landmarks, age, gender)

        Returns:
            If return_face_info=False: embedding array (512-d for buffalo_l) or None
            If return_face_info=True: (embedding, face_info_dict) or (None, None)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for embedding extraction")
            return (None, None) if return_face_info else None

        # Convert RGB to BGR if needed (InsightFace expects BGR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        try:
            # Detect and analyze faces
            faces = self.app.get(image)

            if not faces or len(faces) == 0:
                logger.warning("No face detected in image")
                return (None, None) if return_face_info else None

            # Use the largest face (by bounding box area)
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

            # Get embedding and normalize it (L2 normalization)
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            if return_face_info:
                face_info = {
                    "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                    "landmarks": face.kps.tolist() if hasattr(face, 'kps') else None,
                    "det_score": float(face.det_score),
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": "Male" if (hasattr(face, 'gender') and face.gender == 1) else "Female" if hasattr(face, 'gender') else None,
                }
                return embedding, face_info

            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return (None, None) if return_face_info else None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Dictionary with similarity metrics:
            {
                "cosine_similarity": float (0-1, higher is more similar),
                "euclidean_distance": float (lower is more similar),
                "normalized_score": float (0-1, higher is more similar)
            }
        """
        # Cosine similarity
        cosine_sim = float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(embedding1 - embedding2))

        # Normalized score (0-1 range, combining both metrics)
        # Higher cosine similarity and lower euclidean distance = higher score
        normalized_score = (cosine_sim + (1 - min(euclidean_dist / 2, 1))) / 2

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "euclidean_distance": round(euclidean_dist, 4),
            "normalized_score": round(normalized_score, 4),
        }

    def verify_faces(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify if two images contain the same person.

        Args:
            image1: First image (selfie)
            image2: Second image (ID document face)
            threshold: Custom similarity threshold. If None, uses self.similarity_threshold

        Returns:
            Dictionary with verification result:
            {
                "verified": bool,
                "confidence": float (0-1),
                "similarity_metrics": {
                    "cosine_similarity": float,
                    "euclidean_distance": float,
                    "normalized_score": float
                },
                "threshold_used": float,
                "face1_detected": bool,
                "face2_detected": bool,
                "message": str
            }
        """
        logger.info("Starting face verification...")

        if threshold is None:
            threshold = self.similarity_threshold

        result = {
            "verified": False,
            "confidence": 0.0,
            "similarity_metrics": {},
            "threshold_used": threshold,
            "face1_detected": False,
            "face2_detected": False,
            "message": "",
        }

        # Extract embeddings
        embedding1, face1_info = self.get_embedding(image1, return_face_info=True)
        embedding2, face2_info = self.get_embedding(image2, return_face_info=True)

        # Check if faces detected
        result["face1_detected"] = embedding1 is not None
        result["face2_detected"] = embedding2 is not None

        if embedding1 is None:
            result["message"] = "No face detected in selfie image"
            logger.warning(result["message"])
            return result

        if embedding2 is None:
            result["message"] = "No face detected in ID document image"
            logger.warning(result["message"])
            return result

        # Compute similarity
        similarity_metrics = self.compute_similarity(embedding1, embedding2)
        result["similarity_metrics"] = similarity_metrics

        # Determine verification status
        cosine_sim = similarity_metrics["cosine_similarity"]
        result["confidence"] = similarity_metrics["normalized_score"]

        if cosine_sim >= threshold:
            result["verified"] = True
            result["message"] = f"Faces match with {cosine_sim:.2%} similarity"
            logger.info(f"✓ Verification PASSED - Similarity: {cosine_sim:.4f}")
        else:
            result["verified"] = False
            result["message"] = f"Faces do not match - {cosine_sim:.2%} similarity (threshold: {threshold:.2%})"
            logger.info(f"✗ Verification FAILED - Similarity: {cosine_sim:.4f}")

        return result

    def apply_augmentation(
        self,
        image: np.ndarray,
        augmentation_type: str = "basic",
    ) -> np.ndarray:
        """
        Apply augmentation to improve face matching robustness.
        Note: InsightFace is already robust, but this can help with poor quality images.

        Args:
            image: Input image
            augmentation_type: Type of augmentation
                - "basic": Histogram equalization
                - "clahe": CLAHE for better contrast
                - "denoise": Denoise for noisy images
                - "sharpen": Sharpen blurry images

        Returns:
            Augmented image
        """
        if augmentation_type == "basic":
            # Histogram equalization on Y channel (YCrCb)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        elif augmentation_type == "clahe":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif augmentation_type == "denoise":
            # Denoise for noisy images
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        elif augmentation_type == "sharpen":
            # Sharpen kernel
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)

        return image

    def verify_with_augmentation(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: Optional[float] = None,
        augment_types: list = ["basic", "clahe"],
    ) -> Dict[str, Any]:
        """
        Verify faces with multiple augmentation attempts for robustness.

        Args:
            image1: First image (selfie)
            image2: Second image (ID document face)
            threshold: Custom similarity threshold
            augment_types: List of augmentation types to try

        Returns:
            Best verification result across all augmentations
        """
        logger.info("Starting verification with augmentation...")

        # Try without augmentation first
        best_result = self.verify_faces(image1, image2, threshold)
        best_score = best_result["similarity_metrics"].get("cosine_similarity", 0.0)

        # If already verified, return
        if best_result["verified"]:
            logger.info("Verification passed without augmentation")
            return best_result

        # Try with augmentations
        for aug_type in augment_types:
            logger.info(f"Trying augmentation: {aug_type}")

            # Augment both images
            aug_img1 = self.apply_augmentation(image1, aug_type)
            aug_img2 = self.apply_augmentation(image2, aug_type)

            # Verify
            result = self.verify_faces(aug_img1, aug_img2, threshold)
            score = result["similarity_metrics"].get("cosine_similarity", 0.0)

            # Keep best result
            if score > best_score:
                best_score = score
                best_result = result
                best_result["augmentation_used"] = aug_type
                logger.info(f"Improved score with {aug_type}: {score:.4f}")

            # Early exit if verified
            if result["verified"]:
                logger.info(f"Verification passed with augmentation: {aug_type}")
                return result

        logger.info(f"Best score achieved: {best_score:.4f}")
        return best_result

    def batch_verify(
        self,
        selfie: np.ndarray,
        id_faces: list[np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify selfie against multiple extracted faces from ID (if multiple faces detected).

        Args:
            selfie: Selfie image
            id_faces: List of face images from ID document
            threshold: Custom similarity threshold

        Returns:
            Best match result
        """
        logger.info(f"Batch verification: 1 selfie vs {len(id_faces)} ID faces")

        best_result = None
        best_score = 0.0

        for idx, id_face in enumerate(id_faces):
            result = self.verify_faces(selfie, id_face, threshold)
            score = result["similarity_metrics"].get("cosine_similarity", 0.0)

            if score > best_score:
                best_score = score
                best_result = result
                best_result["matched_face_index"] = idx

            if result["verified"]:
                logger.info(f"Match found with face {idx}")
                return result

        logger.info(f"Best match: face {best_result.get('matched_face_index', -1)} with score {best_score:.4f}")
        return best_result


# Singleton instance
_face_matcher_instance: Optional[InsightFaceMatcher] = None


def get_face_matcher() -> InsightFaceMatcher:
    """
    Get or create singleton InsightFace matcher instance.
    Useful for FastAPI dependency injection.
    """
    global _face_matcher_instance
    if _face_matcher_instance is None:
        _face_matcher_instance = InsightFaceMatcher()
    return _face_matcher_instance