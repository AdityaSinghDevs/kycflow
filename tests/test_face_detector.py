# tests/test_yunet.py

import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.face_detector_id import get_face_detector
from utils.logger import get_logger

logger = get_logger(__name__, log_file="test_yunet.log")


def test_yunet_detection(image_path: str):
    """
    Test YuNet face detection on an ID card image.
    
    Args:
        image_path: Path to ID card image
    """
    logger.info(f"Testing YuNet detection on: {image_path}")
    
    # Check if image exists
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return False
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    logger.info(f"Image loaded successfully. Shape: {img.shape}")
    
    # Get detector
    try:
        detector = get_face_detector()
        logger.info("YuNet detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return False
    
    # Detect faces
    try:
        detection = detector.detect(img, return_largest=True)
        
        if detection is None:
            logger.warning("No faces detected in the image")
            return False
        
        bbox, confidence, landmarks = detection
        logger.info(f"✓ Face detected!")
        logger.info(f"  - Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        logger.info(f"  - Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        logger.info(f"  - Landmarks: {landmarks.shape[0]} points detected")
        
        # Extract face
        face_img = detector.extract_face(img, bbox, padding=0.2, target_size=(112, 112))
        logger.info(f"  - Extracted face shape: {face_img.shape}")
        
        # Visualize detection
        img_vis = detector.visualize_detection(img, bbox, confidence, landmarks)
        
        # Save outputs
        output_dir = Path("tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_path = output_dir / "detection_result.jpg"
        face_path = output_dir / "extracted_face.jpg"
        
        cv2.imwrite(str(vis_path), img_vis)
        cv2.imwrite(str(face_path), face_img)
        
        logger.info(f"✓ Saved visualization to: {vis_path}")
        logger.info(f"✓ Saved extracted face to: {face_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return False


def test_multiple_faces(image_path: str):
    """
    Test detection of all faces in image (if ID has multiple photos).
    """
    logger.info(f"\n--- Testing multiple face detection ---")
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    detector = get_face_detector()
    detections = detector.detect(img, return_largest=False)
    
    if not detections:
        logger.warning("No faces detected")
        return False
    
    logger.info(f"Found {len(detections)} face(s)")
    
    for i, (bbox, conf, landmarks) in enumerate(detections, 1):
        logger.info(f"  Face {i}: confidence={conf:.4f}, bbox={bbox}")
    
    return True


if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default path - UPDATE THIS to your ID image location
        image_path = "data/sample_id.jpg"
        print(f"\nUsage: python tests/test_yunet.py <path_to_id_image>")
        print(f"Using default: {image_path}\n")
    
    logger.info("="*60)
    logger.info("YuNet Face Detection Test")
    logger.info("="*60)
    
    # Run single face test
    success = test_yunet_detection(image_path)
    
    if success:
        logger.info("\n✓✓✓ TEST PASSED ✓✓✓")
        
        # Run multiple face test
        test_multiple_faces(image_path)
    else:
        logger.error("\n✗✗✗ TEST FAILED ✗✗✗")
        sys.exit(1)
    
    logger.info("="*60)