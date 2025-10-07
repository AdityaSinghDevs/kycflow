#!/usr/bin/env python3
"""
Test script for face detection (YuNet) and face matching (InsightFace).
Tests the complete pipeline: ID face detection -> Face matching with selfie.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.services.face_detector_id import YuNetFaceDetector
from app.services.face_matcher import InsightFaceMatcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "data/id",
        "data/realtime_photo",
        "tests/outputs"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {d}")


def load_images(id_path: str, selfie_path: str):
    """Load ID and selfie images."""
    logger.info(f"Loading ID image from: {id_path}")
    id_img = cv2.imread(id_path)
    
    logger.info(f"Loading selfie image from: {selfie_path}")
    selfie_img = cv2.imread(selfie_path)
    
    if id_img is None:
        raise FileNotFoundError(f"Could not load ID image: {id_path}")
    if selfie_img is None:
        raise FileNotFoundError(f"Could not load selfie image: {selfie_path}")
    
    logger.info(f"ID image shape: {id_img.shape}")
    logger.info(f"Selfie image shape: {selfie_img.shape}")
    
    return id_img, selfie_img


def visualize_results(
    id_img: np.ndarray,
    selfie_img: np.ndarray,
    id_detection,
    selfie_detection,
    verification_result: dict,
    output_path: str
):
    """
    Create a visualization combining all results.
    
    Args:
        id_img: Original ID image
        selfie_img: Original selfie image
        id_detection: YuNet detection result for ID
        selfie_detection: YuNet detection result for selfie
        verification_result: InsightFace verification result
        output_path: Path to save visualization
    """
    logger.info("Creating visualization...")
    
    # Create canvas for side-by-side comparison
    h1, w1 = id_img.shape[:2]
    h2, w2 = selfie_img.shape[:2]
    
    # Make both images same height for side-by-side display
    target_height = 600
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    id_resized = cv2.resize(id_img, (int(w1 * scale1), target_height))
    selfie_resized = cv2.resize(selfie_img, (int(w2 * scale2), target_height))
    
    # Draw detections on resized images
    if id_detection is not None:
        bbox, conf, landmarks = id_detection
        # Scale bbox and landmarks
        scaled_bbox = (bbox * scale1).astype(np.int32)
        scaled_landmarks = (landmarks * scale1).astype(np.int32)
        
        x, y, w, h = scaled_bbox
        cv2.rectangle(id_resized, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(id_resized, f"Conf: {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in scaled_landmarks:
            cv2.circle(id_resized, tuple(lm), 3, (0, 0, 255), -1)
    
    if selfie_detection is not None:
        bbox, conf, landmarks = selfie_detection
        # Scale bbox and landmarks
        scaled_bbox = (bbox * scale2).astype(np.int32)
        scaled_landmarks = (landmarks * scale2).astype(np.int32)
        
        x, y, w, h = scaled_bbox
        cv2.rectangle(selfie_resized, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(selfie_resized, f"Conf: {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in scaled_landmarks:
            cv2.circle(selfie_resized, tuple(lm), 3, (0, 0, 255), -1)
    
    # Add labels
    cv2.putText(id_resized, "ID DOCUMENT", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(selfie_resized, "SELFIE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Combine images horizontally
    combined = np.hstack([id_resized, selfie_resized])
    
    # Add verification result panel at bottom
    result_height = 200
    result_panel = np.zeros((result_height, combined.shape[1], 3), dtype=np.uint8)
    result_panel[:] = (40, 40, 40)  # Dark gray background
    
    # Verification status
    verified = verification_result.get("verified", False)
    confidence = verification_result.get("confidence", 0.0)
    
    status_color = (0, 255, 0) if verified else (0, 0, 255)
    status_text = "✓ VERIFIED" if verified else "✗ NOT VERIFIED"
    
    cv2.putText(result_panel, status_text, (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
    
    # Confidence score
    cv2.putText(result_panel, f"Confidence: {confidence:.2%}", (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Similarity metrics
    metrics = verification_result.get("similarity_metrics", {})
    cosine_sim = metrics.get("cosine_similarity", 0.0)
    euclidean_dist = metrics.get("euclidean_distance", 0.0)
    
    cv2.putText(result_panel, f"Cosine Similarity: {cosine_sim:.4f}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(result_panel, f"Euclidean Distance: {euclidean_dist:.4f}", (500, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Stack vertically
    final_output = np.vstack([combined, result_panel])
    
    # Save
    cv2.imwrite(output_path, final_output)
    logger.info(f"Visualization saved to: {output_path}")
    
    return final_output


def test_face_verification_pipeline(id_image_path: str, selfie_image_path: str):
    """
    Complete test pipeline:
    1. Load images
    2. Detect faces with YuNet
    3. Extract face crops
    4. Match faces with InsightFace
    5. Visualize results
    
    Args:
        id_image_path: Path to ID document image
        selfie_image_path: Path to selfie image
    """
    logger.info("="*80)
    logger.info("FACE VERIFICATION PIPELINE TEST")
    logger.info("="*80)
    
    try:
        # Setup directories
        setup_directories()
        
        # Load images
        id_img, selfie_img = load_images(id_image_path, selfie_image_path)
        
        # Initialize models
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Initialize Models")
        logger.info("="*80)
        
        face_detector = YuNetFaceDetector()
        face_matcher = InsightFaceMatcher()
        
        # Detect faces with YuNet
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Face Detection (YuNet)")
        logger.info("="*80)
        
        logger.info("Detecting face in ID document...")
        id_detection = face_detector.detect(id_img, return_largest=True)
        
        if id_detection is None:
            logger.error("❌ No face detected in ID document!")
            return
        
        id_bbox, id_conf, id_landmarks = id_detection
        logger.info(f"✓ Face detected in ID - Confidence: {id_conf:.3f}")
        logger.info(f"  BBox: {id_bbox}")
        
        logger.info("\nDetecting face in selfie...")
        selfie_detection = face_detector.detect(selfie_img, return_largest=True)
        
        if selfie_detection is None:
            logger.error("❌ No face detected in selfie!")
            return
        
        selfie_bbox, selfie_conf, selfie_landmarks = selfie_detection
        logger.info(f"✓ Face detected in selfie - Confidence: {selfie_conf:.3f}")
        logger.info(f"  BBox: {selfie_bbox}")
        
        # Extract face crops
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Extract Face Crops")
        logger.info("="*80)
        
        id_face_crop = face_detector.extract_face(
            id_img, id_bbox, padding=0.2, target_size=(112, 112)
        )
        logger.info(f"ID face crop shape: {id_face_crop.shape}")
        
        selfie_face_crop = face_detector.extract_face(
            selfie_img, selfie_bbox, padding=0.2, target_size=(112, 112)
        )
        logger.info(f"Selfie face crop shape: {selfie_face_crop.shape}")
        
        # Save face crops
        cv2.imwrite("tests/outputs/id_face_crop.jpg", id_face_crop)
        cv2.imwrite("tests/outputs/selfie_face_crop.jpg", selfie_face_crop)
        logger.info("Face crops saved to tests/outputs/")
        
        # Face matching with InsightFace
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Face Matching (InsightFace)")
        logger.info("="*80)
        
        # Method 1: Direct verification
        logger.info("\n--- Method 1: Direct Verification ---")
        result = face_matcher.verify_faces(selfie_img, id_img)
        
        logger.info(f"\nVerification Result:")
        logger.info(f"  Verified: {result['verified']}")
        logger.info(f"  Confidence: {result['confidence']:.2%}")
        logger.info(f"  Cosine Similarity: {result['similarity_metrics']['cosine_similarity']:.4f}")
        logger.info(f"  Euclidean Distance: {result['similarity_metrics']['euclidean_distance']:.4f}")
        logger.info(f"  Message: {result['message']}")
        
        # Method 2: With augmentation (if not verified)
        if not result['verified']:
            logger.info("\n--- Method 2: Verification with Augmentation ---")
            result_aug = face_matcher.verify_with_augmentation(
                selfie_img, id_img,
                augment_types=["basic", "clahe"]
            )
            
            logger.info(f"\nAugmented Verification Result:")
            logger.info(f"  Verified: {result_aug['verified']}")
            logger.info(f"  Confidence: {result_aug['confidence']:.2%}")
            logger.info(f"  Cosine Similarity: {result_aug['similarity_metrics']['cosine_similarity']:.4f}")
            logger.info(f"  Augmentation Used: {result_aug.get('augmentation_used', 'None')}")
            
            # Use augmented result if better
            if result_aug['confidence'] > result['confidence']:
                result = result_aug
                logger.info("  → Using augmented result (better confidence)")
        
        # Visualize results
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Visualize Results")
        logger.info("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tests/outputs/verification_result_{timestamp}.jpg"
        
        visualize_results(
            id_img, selfie_img,
            id_detection, selfie_detection,
            result, output_path
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"ID Face Detected: ✓")
        logger.info(f"Selfie Face Detected: ✓")
        logger.info(f"Verification Status: {'✓ VERIFIED' if result['verified'] else '✗ NOT VERIFIED'}")
        logger.info(f"Final Confidence: {result['confidence']:.2%}")
        logger.info(f"Output saved to: {output_path}")
        logger.info("="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise


def main():
    """Main entry point for test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test face verification pipeline")
    parser.add_argument(
        "--id",
        type=str,
        default="data/id/sample_id.jpg",
        help="Path to ID document image"
    )
    parser.add_argument(
        "--selfie",
        type=str,
        default="data/realtime_photo/sample_selfie4.png",
        help="Path to selfie image"
    )
    
    args = parser.parse_args()
    
    # Run test
    try:
        result = test_face_verification_pipeline(args.id, args.selfie)
        
        # Exit with appropriate code
        sys.exit(0 if result['verified'] else 1)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("\nPlease ensure images exist at:")
        logger.error(f"  ID: {args.id}")
        logger.error(f"  Selfie: {args.selfie}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()