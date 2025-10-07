"""
KYC Processor - Main Orchestration Layer
Coordinates face detection, face matching, and OCR extraction
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.face_detector_id import get_face_detector
from app.services.face_matcher import get_face_matcher
from app.services.ocr_extractor import get_ocr_extractor
from configs.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KYCProcessor:
    """
    Main KYC verification processor that orchestrates:
    1. Face detection on ID and selfie
    2. Face matching between ID and selfie
    3. OCR extraction from ID document
    """
    
    def __init__(self):
        """Initialize all service instances"""
        logger.info("Initializing KYC Processor...")
        try:
            self.face_detector = get_face_detector()
            self.face_matcher = get_face_matcher()
            self.ocr_extractor = get_ocr_extractor()
            logger.info("✓ All services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            NumPy array in BGR format or None if failed
        """
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {image_path}")
                return None
            
            image = cv2.imread(str(path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            logger.info(f"✓ Loaded image: {path.name} (shape: {image.shape})")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _detect_face_from_id(self, id_image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Detect and extract face from ID document
        
        Args:
            id_image: ID document image (BGR)
            
        Returns:
            (face_crop, detection_info)
        """
        try:
            result = self.face_detector.detect_and_extract(id_image)
            
            if result is None:
                logger.warning("No face detected on ID document")
                return None, {
                    "detected": False,
                    "confidence": 0.0,
                    "message": "No face found on ID document"
                }
            
            face_crop, confidence, landmarks = result
            logger.info(f"✓ ID face detected (confidence: {confidence:.3f})")
            
            return face_crop, {
                "detected": True,
                "confidence": float(confidence),
                "landmarks": landmarks.tolist() if landmarks is not None else None,
                "message": "Face detected successfully"
            }
            
        except Exception as e:
            logger.error(f"Error detecting face on ID: {e}")
            return None, {
                "detected": False,
                "confidence": 0.0,
                "message": f"Face detection error: {str(e)}"
            }
    
    def _detect_face_from_selfie(self, selfie_image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Detect and extract face from selfie
        
        Args:
            selfie_image: Selfie image (BGR)
            
        Returns:
            (face_crop, detection_info)
        """
        try:
            result = self.face_detector.detect_and_extract(selfie_image)
            
            if result is None:
                logger.warning("No face detected in selfie")
                return None, {
                    "detected": False,
                    "confidence": 0.0,
                    "message": "No face found in selfie"
                }
            
            face_crop, confidence, landmarks = result
            logger.info(f"✓ Selfie face detected (confidence: {confidence:.3f})")
            
            return face_crop, {
                "detected": True,
                "confidence": float(confidence),
                "landmarks": landmarks.tolist() if landmarks is not None else None,
                "message": "Face detected successfully"
            }
            
        except Exception as e:
            logger.error(f"Error detecting face in selfie: {e}")
            return None, {
                "detected": False,
                "confidence": 0.0,
                "message": f"Face detection error: {str(e)}"
            }
    
    def _verify_faces(self, id_face: np.ndarray, selfie_face: np.ndarray) -> Dict[str, Any]:
        """
        Verify if ID face matches selfie face
        
        Args:
            id_face: Cropped face from ID (BGR)
            selfie_face: Cropped face from selfie (BGR)
            
        Returns:
            Verification result dictionary
        """
        try:
            threshold = config.get("verification", "similarity_threshold", 0.4)
            result = self.face_matcher.verify_faces(id_face, selfie_face, threshold=threshold)
            
            if result["verified"]:
                logger.info(f"✓ Faces MATCH (confidence: {result['confidence']:.2%})")
            else:
                logger.warning(f"✗ Faces DO NOT MATCH (confidence: {result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying faces: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "message": f"Verification error: {str(e)}",
                "similarity_metrics": None
            }
    
    def _extract_ocr(self, id_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text from ID document using OCR
        
        Args:
            id_image: ID document image (BGR)
            
        Returns:
            OCR extraction result
        """
        try:
            confidence_threshold = config.get("ocr", "confidence_threshold", 0.3)
            result = self.ocr_extractor.extract_structured(
                id_image,
                confidence_threshold=confidence_threshold
            )
            
            doc_type = result.get("document_type", "unknown")
            total_regions = result.get("metadata", {}).get("total_text_regions", 0)
            logger.info(f"✓ OCR extracted: {doc_type} ({total_regions} text regions)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting OCR: {e}")
            return {
                "raw_text": [],
                "document_type": "unknown",
                "fields": {},
                "metadata": {
                    "total_text_regions": 0,
                    "avg_confidence": 0.0,
                    "error": str(e)
                }
            }
    
    def process_kyc(
        self,
        id_document_path: str,
        selfie_path: str,
        run_ocr: bool = True
    ) -> Dict[str, Any]:
        """
        Complete KYC verification pipeline
        
        Args:
            id_document_path: Path to ID document image
            selfie_path: Path to selfie image
            run_ocr: Whether to run OCR extraction (default: True)
            
        Returns:
            Complete KYC result dictionary formatted for Ballerine frontend:
            - verification_status: "approved" | "rejected" | "pending"
            - confidence_score: Overall confidence (0-1)
            - face_match_score: Face matching confidence (0-1)
            - ocr_data: Extracted document information
            - processing_time_ms: Processing time in milliseconds
            - timestamp: ISO 8601 timestamp
        """
        start_time = time.time()
        logger.info(f"Starting KYC processing...")
        logger.info(f"ID Document: {id_document_path}")
        logger.info(f"Selfie: {selfie_path}")
        
        # Initialize result structure (Frontend-compatible format)
        result = {
            "verification_status": "pending",
            "confidence_score": 0.0,
            "face_match_score": 0.0,
            "ocr_data": {
                "extracted_text": "",
                "fields": {}
            },
            "processing_time_ms": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "details": {
                "face_detection": {
                    "id_face_found": False,
                    "selfie_face_found": False,
                    "id_detection_confidence": 0.0,
                    "selfie_detection_confidence": 0.0
                },
                "errors": []
            }
        }
        
        # Load images
        id_image = self._load_image(id_document_path)
        selfie_image = self._load_image(selfie_path)
        
        if id_image is None:
            result["details"]["errors"].append("Failed to load ID document")
            result["verification_status"] = "rejected"
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        if selfie_image is None:
            result["details"]["errors"].append("Failed to load selfie")
            result["verification_status"] = "rejected"
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Parallel processing: Face detection + OCR
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._detect_face_from_id, id_image): "id_face",
                executor.submit(self._detect_face_from_selfie, selfie_image): "selfie_face"
            }
            
            # Add OCR task if requested
            if run_ocr:
                futures[executor.submit(self._extract_ocr, id_image)] = "ocr"
            
            # Collect results
            id_face_crop = None
            selfie_face_crop = None
            ocr_result = None
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    if task_name == "id_face":
                        id_face_crop, id_detection_info = future.result()
                        result["details"]["face_detection"]["id_face_found"] = id_detection_info["detected"]
                        result["details"]["face_detection"]["id_detection_confidence"] = id_detection_info["confidence"]
                    
                    elif task_name == "selfie_face":
                        selfie_face_crop, selfie_detection_info = future.result()
                        result["details"]["face_detection"]["selfie_face_found"] = selfie_detection_info["detected"]
                        result["details"]["face_detection"]["selfie_detection_confidence"] = selfie_detection_info["confidence"]
                    
                    elif task_name == "ocr":
                        ocr_result = future.result()
                
                except Exception as e:
                    logger.error(f"Error in parallel task {task_name}: {e}")
                    result["details"]["errors"].append(f"{task_name} failed: {str(e)}")
        
        # Process OCR data if available
        if ocr_result:
            # Format extracted text
            raw_texts = [text for _, text, _ in ocr_result.get("raw_text", [])]
            result["ocr_data"]["extracted_text"] = "\n".join(raw_texts)
            result["ocr_data"]["fields"] = ocr_result.get("fields", {})
            
            # Add document type and metadata to details
            result["details"]["document_type"] = ocr_result.get("document_type", "unknown")
            result["details"]["ocr_metadata"] = ocr_result.get("metadata", {})
        
        # Face verification (only if both faces detected)
        if id_face_crop is not None and selfie_face_crop is not None:
            verification_result = self._verify_faces(id_face_crop, selfie_face_crop)
            
            # Extract face match score
            result["face_match_score"] = verification_result.get("confidence", 0.0)
            
            # Determine verification status
            if verification_result.get("verified", False):
                result["verification_status"] = "approved"
                result["confidence_score"] = verification_result.get("confidence", 0.0)
            else:
                result["verification_status"] = "rejected"
                result["confidence_score"] = verification_result.get("confidence", 0.0)
            
            # Store detailed similarity metrics
            result["details"]["similarity_metrics"] = verification_result.get("similarity_metrics")
            result["details"]["verification_message"] = verification_result.get("message", "")
        else:
            # Face detection failed
            result["verification_status"] = "rejected"
            
            # Provide specific message
            if id_face_crop is None and selfie_face_crop is None:
                result["details"]["verification_message"] = "No faces detected in either image"
            elif id_face_crop is None:
                result["details"]["verification_message"] = "No face detected on ID document"
            else:
                result["details"]["verification_message"] = "No face detected in selfie"
        
        # Calculate overall confidence score (average of detection + matching)
        detection_scores = []
        if result["details"]["face_detection"]["id_face_found"]:
            detection_scores.append(result["details"]["face_detection"]["id_detection_confidence"])
        if result["details"]["face_detection"]["selfie_face_found"]:
            detection_scores.append(result["details"]["face_detection"]["selfie_detection_confidence"])
        
        if detection_scores:
            avg_detection_score = sum(detection_scores) / len(detection_scores)
            # Combine detection and matching scores
            if result["face_match_score"] > 0:
                result["confidence_score"] = (avg_detection_score + result["face_match_score"]) / 2
            else:
                result["confidence_score"] = avg_detection_score * 0.5  # Penalize if no match
        
        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time_ms"] = int(processing_time * 1000)
        
        logger.info(f"✓ KYC processing completed in {processing_time:.3f}s")
        logger.info(f"Verification Status: {result['verification_status'].upper()}")
        logger.info(f"Confidence Score: {result['confidence_score']:.2%}")
        
        return result


# Singleton instance
_processor_instance: Optional[KYCProcessor] = None


def get_kyc_processor() -> KYCProcessor:
    """
    Get or create singleton KYC processor instance
    
    Returns:
        KYCProcessor instance
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = KYCProcessor()
    return _processor_instance


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="KYC Verification Processor")
    parser.add_argument("--id", required=True, help="Path to ID document image")
    parser.add_argument("--selfie", required=True, help="Path to selfie image")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR extraction")
    parser.add_argument("--output", help="Save result to JSON file")
    
    args = parser.parse_args()
    
    # Process KYC
    processor = get_kyc_processor()
    result = processor.process_kyc(
        id_document_path=args.id,
        selfie_path=args.selfie,
        run_ocr=not args.no_ocr
    )
    
    # Print result
    print("\n" + "="*60)
    print("KYC VERIFICATION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to: {args.output}")