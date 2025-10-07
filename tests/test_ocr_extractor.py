#!/usr/bin/env python3
"""
Test script for OCR extraction from ID documents.
Tests the enhanced OCR extractor with preprocessing and structured output.
"""

import cv2
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ocr_extractor import ImprovedOCRExtractor, get_ocr_extractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["data/id", "tests/outputs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {d}")


def visualize_ocr_results(image, text_regions, output_path: str):
    """
    Visualize OCR detection with bounding boxes and text overlays.
    
    Args:
        image: Original image
        text_regions: List of extracted text regions with bbox, text, confidence
        output_path: Path to save visualization
    """
    import numpy as np
    
    logger.info("Creating OCR visualization...")
    
    vis_img = image.copy()
    
    for region in text_regions:
        bbox = region["bbox"]
        text = region["text"]
        conf = region["confidence"]
        lang = region["language"]
        
        # Convert bbox to integer points
        pts = np.array(bbox, dtype=np.int32)
        
        # Choose color based on language
        color = (0, 255, 0) if lang == "en" else (255, 165, 0)  # Green for EN, Orange for HI
        
        # Draw bounding box
        cv2.polylines(vis_img, [pts], True, color, 2)
        
        # Draw text background
        text_size = cv2.getTextSize(text[:30], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis_img, 
                     (pts[0][0], pts[0][1] - text_size[1] - 5),
                     (pts[0][0] + text_size[0], pts[0][1]),
                     color, -1)
        
        # Draw text
        cv2.putText(vis_img, 
                   text[:30],  # Truncate long text
                   (pts[0][0], pts[0][1] - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save visualization
    cv2.imwrite(output_path, vis_img)
    logger.info(f"Visualization saved to: {output_path}")
    
    return vis_img


def create_results_panel(ocr_result, image_width: int) -> any:
    """
    Create a results panel showing extracted fields.
    
    Args:
        ocr_result: OCRResult dataclass
        image_width: Width to match original image
        
    Returns:
        Results panel image
    """
    import numpy as np
    
    panel_height = 800
    panel = np.ones((panel_height, image_width, 3), dtype=np.uint8) * 240  # Light gray
    
    y_offset = 40
    line_height = 35
    
    # Title
    cv2.putText(panel, "EXTRACTED OCR DATA", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # Increased thickness for bold effect
    y_offset += 50
    
    # Document type
    doc_type = ocr_result.document_type.upper()
    color = (0, 150, 0) if doc_type != "UNKNOWN" else (0, 0, 200)
    cv2.putText(panel, f"Document Type: {doc_type}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_offset += line_height
    
    # Confidence
    conf_color = (0, 200, 0) if ocr_result.confidence > 0.7 else (0, 100, 200)
    cv2.putText(panel, f"Avg Confidence: {ocr_result.confidence:.1%}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
    y_offset += line_height + 10
    
    # Divider
    cv2.line(panel, (20, y_offset), (image_width - 20, y_offset), (150, 150, 150), 2)
    y_offset += 20
    
    # Extracted fields
    fields = [
        ("Document Number", ocr_result.document_number),
        ("Name", ocr_result.name),
        ("Father's Name", ocr_result.father_name),
        ("Mother's Name", ocr_result.mother_name),
        ("Date of Birth", ocr_result.date_of_birth),
        ("Gender", ocr_result.gender),
        ("Blood Group", ocr_result.blood_group),
        ("Mobile", ocr_result.mobile),
        ("Pincode", ocr_result.pincode),
        ("Date of Issue", ocr_result.date_of_issue),
        ("Date of Expiry", ocr_result.date_of_expiry),
    ]
    
    for label, value in fields:
        if value:
            # Truncate long values
            display_value = value if len(str(value)) < 40 else str(value)[:37] + "..."
            
            cv2.putText(panel, f"{label}:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
            cv2.putText(panel, str(display_value), (250, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += line_height
    
    # Address (multi-line)
    if ocr_result.address:
        cv2.putText(panel, "Address:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
        y_offset += line_height
        
        # Wrap address text
        address = ocr_result.address
        max_chars = 50
        words = address.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        for line in lines[:3]:  # Max 3 lines
            cv2.putText(panel, line, (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 30
    
    y_offset += 10
    
    # Languages detected
    if ocr_result.language_detected:
        lang_str = ", ".join(ocr_result.language_detected).upper()
        cv2.putText(panel, f"Languages: {lang_str}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    return panel


def test_ocr_extraction(image_path: str):
    """
    Complete OCR extraction test pipeline.
    
    Args:
        image_path: Path to ID document image
    """
    logger.info("=" * 80)
    logger.info("OCR EXTRACTION TEST")
    logger.info("=" * 80)
    
    try:
        # Setup directories
        setup_directories()
        
        # Load image
        logger.info(f"\nLoading image from: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        logger.info(f"Image shape: {image.shape}")
        
        # Initialize OCR extractor
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Initialize OCR Extractor")
        logger.info("=" * 80)
        
        ocr_extractor = get_ocr_extractor()
        
        # Extract text regions (with visualization)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extract Text Regions")
        logger.info("=" * 80)
        
        text_regions = ocr_extractor.extract_text_regions(
        image, 
        confidence_threshold=0.2,
        preprocess="auto"  # Try original first, then fallback to preprocessing
        )
        
        logger.info(f"\nExtracted {len(text_regions)} text regions")
        
        # Show sample extractions
        logger.info("\nSample text regions:")
        for i, region in enumerate(text_regions[:10]):  # Show first 10
            logger.info(f"  {i+1}. [{region['language'].upper()}] "
                       f"{region['text'][:50]:50s} (conf: {region['confidence']:.3f})")
        
        if len(text_regions) > 10:
            logger.info(f"  ... and {len(text_regions) - 10} more")
        
        # Extract structured data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Extract Structured Data")
        logger.info("=" * 80)
        
        ocr_result = ocr_extractor.extract_structured(
            image,
            confidence_threshold=0.2  # Lower threshold
        )
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("EXTRACTED FIELDS")
        logger.info("=" * 80)
        
        result_dict = ocr_result.to_dict()
        
        # Pretty print results (excluding raw_text and language_detected)
        for key, value in result_dict.items():
            if key not in ["raw_text", "language_detected"]:
                formatted_key = key.replace("_", " ").title()
                logger.info(f"{formatted_key:20s}: {value}")
        
        # Languages
        if ocr_result.language_detected:
            logger.info(f"{'Languages':20s}: {', '.join(ocr_result.language_detected).upper()}")
        
        # Create visualizations
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Create Visualizations")
        logger.info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Visualization 1: OCR bounding boxes
        vis_path = f"tests/outputs/ocr_visualization_{timestamp}.jpg"
        visualize_ocr_results(image, text_regions, vis_path)
        
        # Visualization 2: Combined output (original + results panel)
        results_panel = create_results_panel(ocr_result, image.shape[1])
        combined_output = cv2.vconcat([image, results_panel])
        combined_path = f"tests/outputs/ocr_results_{timestamp}.jpg"
        cv2.imwrite(combined_path, combined_output)
        logger.info(f"Combined results saved to: {combined_path}")
        
        # Save JSON output
        json_path = f"tests/outputs/ocr_data_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON data saved to: {json_path}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Document Type: {ocr_result.document_type.upper()}")
        logger.info(f"Text Regions Extracted: {len(text_regions)}")
        logger.info(f"Average Confidence: {ocr_result.confidence:.1%}")
        logger.info(f"Fields Extracted: {sum(1 for k, v in result_dict.items() if v is not None and k not in ['raw_text', 'language_detected'])}")
        logger.info(f"Languages Detected: {', '.join(ocr_result.language_detected).upper()}")
        logger.info("\nOutputs:")
        logger.info(f"  - OCR Visualization: {vis_path}")
        logger.info(f"  - Combined Results: {combined_path}")
        logger.info(f"  - JSON Data: {json_path}")
        logger.info("=" * 80)
        
        return ocr_result
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise


def main():
    """Main entry point for test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OCR extraction from ID documents")
    parser.add_argument(
        "--id",
        type=str,
        default="data/id/sample_id.jpg",
        help="Path to ID document image"
    )
    
    args = parser.parse_args()
    
    # Run test
    try:
        result = test_ocr_extraction(args.id)
        
        # Exit with success
        sys.exit(0)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(f"\nPlease ensure image exists at: {args.id}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()