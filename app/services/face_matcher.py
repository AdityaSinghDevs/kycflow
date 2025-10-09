# app/services/face_matcher.py

"""
InsightFace Matcher - Face verification via embeddings
Compares ID document face with selfie face using deep learning embeddings
"""

import numpy as np
from typing import Optional, Dict, Any
import cv2


import insightface
from insightface.app import FaceAnalysis

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="test_yunet.log")



class FaceMatchResult:
    """Encapsulates face matching result"""
    
    def __init__(
        self,
        verified: bool,
        confidence: float,
        cosine_similarity: float,
        euclidean_distance: float,
        threshold_used: float,
        message: str = ""
    ):
        self.verified = verified
        self.confidence = confidence
        self.cosine_similarity = cosine_similarity
        self.euclidean_distance = euclidean_distance
        self.threshold_used = threshold_used
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "verified": self.verified,
            "confidence": round(self.confidence, 4),
            "similarity_metrics": {
                "cosine_similarity": round(self.cosine_similarity, 4),
                "euclidean_distance": round(self.euclidean_distance, 4)
            },
            "threshold_used": self.threshold_used,
            "message": self.message
        }


class InsightFaceMatcher:
    """
    Face verification using InsightFace embeddings.
    
    Workflow:
    1. Extract 512-dim embeddings from both face images
    2. Compute cosine similarity between embeddings
    3. Compare against threshold to determine match
    
    Note: Face detection is handled by YuNet, this class only does matching.
    """

    def __init__(
    self,
    model_name: Optional[str] = None,
    use_gpu: bool = False,
    similarity_threshold: float = 0.4
):
    """
                 Initialize InsightFace matcher.
                
                 Args:
                 model_name: Model pack ('buffalo_l', 'buffalo_s'). If None, uses config.
                 use_gpu: Use CUDA if available. Set to False for CPU-only.
                     similarity_threshold: Cosine similarity threshold for verification.
                         - 0.3: Very lenient (high false positives)
                         - 0.4: Balanced (recommended)
                         - 0.5: Strict (may reject valid matches)
    """
    if model_name is None:
        model_name = config.get("models", "face_recognition", "model_name", default="buffalo_l")

    self.model_name = model_name
    self.similarity_threshold = similarity_threshold

    # Initialize FaceAnalysis
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    try:
        self.app = FaceAnalysis(name=model_name, providers=providers)
        
        # Try to prepare with requested device
        try:
            ctx_id = 0 if use_gpu else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(f"InsightFace prepared with ctx_id={ctx_id}")
        except Exception as gpu_error:
            # Fallback to CPU only if GPU was requested
            if use_gpu:
                logger.warning(f"GPU initialization failed, falling back to CPU: {gpu_error}")
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
            else:
                # CPU failed - this is fatal
                raise
        
        device = "GPU" if use_gpu else "CPU"
        logger.info(f"InsightFace initialized: {model_name} on {device}, threshold={similarity_threshold}")
        
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        raise

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized embedding from face image.

        Args:
            face_image: Face image (BGR format). Should be already cropped to face region.

        Returns:
            512-dim normalized embedding, or None if face not detected
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None

        # Convert grayscale to BGR if needed
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        try:
            # InsightFace re-detects faces in image
            faces = self.app.get(face_image)

            if not faces or len(faces) == 0:
                logger.warning("InsightFace could not detect face in cropped image")
                return None

            # Use first face (should only be one in a cropped face image)
            face = faces[0]
            
            # Get embedding and normalize (L2 norm)
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            logger.debug(f"Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute similarity metrics between two embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Dict with cosine_similarity, euclidean_distance, normalized_score
        """
        # Cosine similarity (primary metric)
        cosine_sim = float(np.dot(embedding1, embedding2))

        # Euclidean distance (secondary metric)
        euclidean_dist = float(np.linalg.norm(embedding1 - embedding2))

        # Normalized score (0-1 range combining both)
        normalized_score = (cosine_sim + (1 - min(euclidean_dist / 2, 1))) / 2

        return {
            "cosine_similarity": cosine_sim,
            "euclidean_distance": euclidean_dist,
            "normalized_score": normalized_score
        }

    def verify(
        self,
        face1: np.ndarray,
        face2: np.ndarray,
        threshold: Optional[float] = None
    ) -> FaceMatchResult:
        """
        Verify if two face images belong to the same person.

        Args:
            face1: First face image (typically ID document face)
            face2: Second face image (typically selfie face)
            threshold: Custom threshold. If None, uses self.similarity_threshold

        Returns:
            FaceMatchResult with verification decision and metrics
        """
        if threshold is None:
            threshold = self.similarity_threshold

        logger.info("Starting face verification...")

        # Extract embeddings
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)

        # Check if embeddings extracted successfully
        if emb1 is None:
            return FaceMatchResult(
                verified=False,
                confidence=0.0,
                cosine_similarity=0.0,
                euclidean_distance=999.0,
                threshold_used=threshold,
                message="Failed to extract embedding from first image"
            )

        if emb2 is None:
            return FaceMatchResult(
                verified=False,
                confidence=0.0,
                cosine_similarity=0.0,
                euclidean_distance=999.0,
                threshold_used=threshold,
                message="Failed to extract embedding from second image"
            )

        # Compute similarity
        metrics = self.compute_similarity(emb1, emb2)
        cosine_sim = metrics["cosine_similarity"]
        confidence = metrics["normalized_score"]

        # Verify
        verified = cosine_sim >= threshold

        if verified:
            message = f"Faces match ({cosine_sim:.1%} similarity)"
            logger.info(f"✓ MATCH: {cosine_sim:.4f} >= {threshold:.4f}")
        else:
            message = f"Faces do not match ({cosine_sim:.1%} similarity, threshold: {threshold:.1%})"
            logger.info(f"✗ NO MATCH: {cosine_sim:.4f} < {threshold:.4f}")

        return FaceMatchResult(
            verified=verified,
            confidence=confidence,
            cosine_similarity=cosine_sim,
            euclidean_distance=metrics["euclidean_distance"],
            threshold_used=threshold,
            message=message
        )


# Singleton instance
import threading

_matcher_instance: Optional[InsightFaceMatcher] = None
_matcher_lock = threading.Lock()

def get_face_matcher() -> InsightFaceMatcher:
    """Thread-safe singleton getter."""
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = InsightFaceMatcher()
                logger.info("Face matcher singleton created")
    return _matcher_instance


def reset_matcher() -> None:
    """Reset singleton (useful for testing)"""
    global _matcher_instance
    _matcher_instance = None