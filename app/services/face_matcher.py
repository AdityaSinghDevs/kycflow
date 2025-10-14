# app/services/face_matcher.py

"""
InsightFace Matcher - Face verification via embeddings
FIXED: Direct embedding extraction without re-detection + shape fix
"""

import numpy as np
from typing import Optional, Dict, Any
import cv2
import threading

import insightface
from insightface.app import FaceAnalysis

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="face_matcher.log")


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
    Uses recognition model directly without re-detection.
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
        """
        if model_name is None:
            model_name = config.get("models", "face_recognition", "model_name", default="buffalo_l")
        
        if similarity_threshold is None:
            similarity_threshold = config.get("models", "face_recognition", "similarity_threshold", default=0.4)

        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

        # Initialize FaceAnalysis
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        try:
            logger.info(f"Initializing InsightFace with model: {model_name}")
            self.app = FaceAnalysis(name=model_name, providers=providers)
            
            # Prepare with device
            try:
                ctx_id = 0 if use_gpu else -1
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                logger.info(f"InsightFace prepared with ctx_id={ctx_id}")
            except Exception as gpu_error:
                if use_gpu:
                    logger.warning(f"GPU failed, falling back to CPU: {gpu_error}")
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                else:
                    raise
            
            # ✅ FIX: Get recognition model directly (bypass detection)
            self.rec_model = self.app.models.get('recognition')
            if self.rec_model is None:
                # Fallback: find first recognition model
                for model in self.app.models.values():
                    if hasattr(model, 'get_feat'):
                        self.rec_model = model
                        break
            
            if self.rec_model is None:
                raise RuntimeError("No recognition model found in InsightFace app")
            
            device = "GPU" if use_gpu else "CPU"
            logger.info(f"InsightFace ready: {model_name} on {device}, threshold={similarity_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized embedding from face image.
        Uses recognition model directly - NO re-detection.

        Args:
            face_image: Face image (BGR format). Already cropped to 112x112 by YuNet.

        Returns:
            512-dim normalized embedding, or None if extraction fails
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None

        # Convert grayscale to BGR if needed
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        
        # Ensure 112x112 size (InsightFace standard)
        if face_image.shape[:2] != (112, 112):
            logger.debug(f"Resizing face from {face_image.shape[:2]} to (112, 112)")
            face_image = cv2.resize(face_image, (112, 112))

        try:
            # ✅ FIX: Direct embedding extraction (no detection)
            # Convert BGR to RGB (InsightFace expects RGB)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get embedding directly from recognition model
            embedding = self.rec_model.get_feat(face_rgb)
            
            # ✅ FIX: Flatten to 1D array if needed (1,512) -> (512,)
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # Normalize (L2 norm)
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

        # Normalized score (0-1 range)
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
            face1: First face image (ID document face, 112x112)
            face2: Second face image (selfie face, 112x112)
            threshold: Custom threshold. If None, uses self.similarity_threshold

        Returns:
            FaceMatchResult with verification decision and metrics
        """
        if threshold is None:
            threshold = self.similarity_threshold

        logger.info(f"Starting face verification (threshold={threshold})...")

        # Extract embeddings
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)

        # Check if embeddings extracted successfully
        if emb1 is None:
            logger.warning("Failed to extract embedding from ID face")
            return FaceMatchResult(
                verified=False,
                confidence=0.0,
                cosine_similarity=0.0,
                euclidean_distance=999.0,
                threshold_used=threshold,
                message="Failed to extract embedding from ID document face"
            )

        if emb2 is None:
            logger.warning("Failed to extract embedding from selfie face")
            return FaceMatchResult(
                verified=False,
                confidence=0.0,
                cosine_similarity=0.0,
                euclidean_distance=999.0,
                threshold_used=threshold,
                message="Failed to extract embedding from selfie face"
            )

        # Compute similarity
        metrics = self.compute_similarity(emb1, emb2)
        cosine_sim = metrics["cosine_similarity"]
        confidence = metrics["normalized_score"]

        # Verify
        verified = cosine_sim >= threshold

        if verified:
            message = f"Faces match ({cosine_sim:.1%} similarity)"
            logger.info(f"✓ MATCH: cosine={cosine_sim:.4f} >= threshold={threshold:.4f}")
        else:
            message = f"Faces do not match ({cosine_sim:.1%} similarity, threshold: {threshold:.1%})"
            logger.info(f"✗ NO MATCH: cosine={cosine_sim:.4f} < threshold={threshold:.4f}")

        return FaceMatchResult(
            verified=verified,
            confidence=confidence,
            cosine_similarity=cosine_sim,
            euclidean_distance=metrics["euclidean_distance"],
            threshold_used=threshold,
            message=message
        )


# ============================================================================
# Singleton Pattern - Thread-safe
# ============================================================================

_matcher_instance: Optional[InsightFaceMatcher] = None
_matcher_lock = threading.Lock()


def get_face_matcher() -> InsightFaceMatcher:
    """Thread-safe singleton getter."""
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = InsightFaceMatcher(
                    use_gpu=config.use_gpu,
                    similarity_threshold=config.get("models", "face_recognition", "similarity_threshold", default=0.4)
                )
                logger.info("Face matcher singleton created")
    return _matcher_instance


def reset_matcher() -> None:
    """Reset singleton (for testing)"""
    global _matcher_instance
    _matcher_instance = None
    logger.info("Face matcher singleton reset")