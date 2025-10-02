# tests/test_ocr_extractor.py
import cv2
import os
from app.services.ocr_extractor import OCRExtractor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_ocr_extraction():
    """
    Test the OCRExtractor with a sample ID image.
    """
    # Path to the sample image
    image_path = "data/sample_id.jpg"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
    
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Initialize OCRExtractor
        ocr = OCRExtractor(languages=['en'], gpu=False)  # Set gpu=False for testing
        
        # Extract structured data
        structured_data = ocr.extract_structured(image, confidence_threshold=0.3)
        
        # Format and print the output
        formatted_output = ocr.format_output(structured_data)
        print(formatted_output)
        
        logger.info("OCR extraction test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during OCR extraction test: {e}")
        raise

if __name__ == "__main__":
    test_ocr_extraction()