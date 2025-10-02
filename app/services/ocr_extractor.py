import easyocr
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import cv2
from configs.config import config

logger = logging.getLogger(__name__)

class OCRExtractor:
    """
    EasyOCR-based text extraction from ID documents.
    Extracts raw text and parses into structured key-value pairs.
    """
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = True,
    ):
        """
        Initialize EasyOCR reader.
        Args:
            languages: List of language codes (e.g., ['en', 'hi']). If None, uses config.
            gpu: Use GPU acceleration if available
        """
        if languages is None:
            languages = config.get("models", "ocr", "languages", default=["en"])
        logger.info(f"Initializing EasyOCR with languages: {languages}")
       
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False,
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
        # Regex patterns for common ID fields
        self.patterns = {
            "date": [
                r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b", # DD-MM-YYYY or DD/MM/YYYY
                r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b", # YYYY-MM-DD
                r"\b(\d{2}\s+[A-Za-z]{3,9}\s+\d{4})\b", # DD Month YYYY
            ],
            "aadhaar": [
                r"\b(\d{4}\s+\d{4}\s+\d{4})\b", # Aadhaar: 1234 5678 9012
                r"\b(\d{12})\b", # Aadhaar without spaces
            ],
            "pan": [
                r"\b([A-Z]{5}\d{4}[A-Z])\b", # PAN: ABCDE1234F
            ],
            "driving_license": [
                r"\b([A-Z]{2}\d{13})\b", # DL: HR0619850034761
                r"\b([A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7})\b",
            ],
            "passport": [
                r"\b([A-Z]\d{7})\b", # Passport: A1234567
            ],
            "voter_id": [
                r"\b([A-Z]{3}\d{7})\b", # Voter ID: ABC1234567
            ],
            "pincode": [
                r"\b(\d{6})\b", # Indian pincode
            ],
            "mobile": [
                r"\b([6-9]\d{9})\b", # Indian mobile: 9876543210
                r"\b(\+91[\s-]?[6-9]\d{9})\b", # With +91
            ],
        }
        # Common field keywords for heuristic matching
        self.field_keywords = {
            "name": ["name", "naam", "holder"],
            "father_name": ["father", "s/o", "son of", "pita"],
            "mother_name": ["mother", "d/o", "daughter of", "mata"],
            "dob": ["date of birth", "dob", "birth", "janm"],
            "doi": ["date of issue", "issue", "issued"],
            "doe": ["date of expiry", "expiry", "valid", "doe"],
            "address": ["address", "pata"],
            "gender": ["sex", "gender", "male", "female", "ling"],
        }

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better OCR accuracy.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding to improve contrast
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Convert back to BGR for OCR compatibility
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced

    def extract_text(
        self,
        image: np.ndarray,
        detail: int = 1,
    ) -> List[Tuple[List[List[int]], str, float]]:
        """
        Extract raw text from image using EasyOCR.
        Args:
            image: Input image (BGR or RGB format)
            detail: 0 = text only, 1 = text + bbox + confidence
        Returns:
            List of (bounding_box, text, confidence) tuples
            bounding_box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        logger.info("Extracting text from image...")
        # Enhance image before OCR
        image = self._enhance_image(image)
        try:
            results = self.reader.readtext(image, detail=detail)
            logger.info(f"Extracted {len(results)} text regions")
            return results
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    def extract_text_simple(self, image: np.ndarray) -> List[str]:
        """
        Extract only text strings (no bounding boxes).
        Args:
            image: Input image
        Returns:
            List of text strings
        """
        results = self.extract_text(image, detail=1)
        return [text for _, text, _ in results]

    def extract_structured(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,  # Increased default
    ) -> Dict[str, Any]:
        """
        Extract and parse text into structured key-value pairs.
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for text extraction
        Returns:
            Dictionary with structured data
        """
        logger.info("Extracting structured data from document...")
        # Extract raw text
        results = self.extract_text(image, detail=1)
       
        # Filter by confidence
        filtered_results = [
            (bbox, text, conf)
            for bbox, text, conf in results
            if conf >= confidence_threshold
        ]
        if not filtered_results:
            logger.warning("No text extracted above confidence threshold")
            return {
                "raw_text": [],
                "document_type": "unknown",
                "fields": {},
                "metadata": {
                    "total_text_regions": 0,
                    "avg_confidence": 0.0,
                }
            }
        # Extract text and confidence
        texts = [text for _, text, _ in filtered_results]
        confidences = [conf for _, _, conf in filtered_results]
       
        avg_confidence = sum(confidences) / len(confidences)
        logger.info(f"Average OCR confidence: {avg_confidence:.3f}")
        # Detect document type
        doc_type = self._detect_document_type(texts)
        logger.info(f"Detected document type: {doc_type}")
        # Parse fields
        fields = self._parse_fields(texts)
       
        # Add document number if detected
        doc_number = self._extract_document_number(texts, doc_type)
        if doc_number:
            fields["document_number"] = doc_number
            fields["document_type"] = doc_type
        return {
            "raw_text": texts,
            "document_type": doc_type,
            "fields": fields,
            "metadata": {
                "total_text_regions": len(filtered_results),
                "avg_confidence": round(avg_confidence, 3),
            }
        }

    def _detect_document_type(self, texts: List[str]) -> str:
        """
        Detect document type based on extracted text patterns.
        """
        combined_text = " ".join(texts).upper()
        if re.search(self.patterns["aadhaar"][0], combined_text) or "AADHAAR" in combined_text:
            return "aadhaar"
        elif re.search(self.patterns["pan"][0], combined_text) or "INCOME TAX" in combined_text:
            return "pan"
        elif re.search(self.patterns["driving_license"][0], combined_text) or "DRIVING" in combined_text:
            return "driving_license"
        elif re.search(self.patterns["passport"][0], combined_text) or "PASSPORT" in combined_text:
            return "passport"
        elif re.search(self.patterns["voter_id"][0], combined_text) or "VOTER" in combined_text or "ELECTION" in combined_text:
            return "voter_id"
        else:
            return "unknown"

    def _extract_document_number(self, texts: List[str], doc_type: str) -> Optional[str]:
        """
        Extract document number based on document type.
        """
        combined_text = " ".join(texts)
        if doc_type in self.patterns:
            for pattern in self.patterns[doc_type]:
                match = re.search(pattern, combined_text)
                if match:
                    return match.group(1)
        return None

    def _parse_fields(self, texts: List[str]) -> Dict[str, str]:
        """
        Parse common fields from text using heuristics and patterns.
        """
        fields = {}
        combined_text = " ".join(texts)
        dates = self._extract_all_dates(combined_text)
        if dates:
            fields.update(self._classify_dates(texts, dates))
        name = self._extract_name(texts)
        if name:
            fields["name"] = name
        parent_names = self._extract_parent_names(texts)
        fields.update(parent_names)
        gender = self._extract_gender(combined_text)
        if gender:
            fields["gender"] = gender
        address = self._extract_address(texts)
        if address:
            fields["address"] = address
        mobile = self._extract_mobile(combined_text)
        if mobile:
            fields["mobile"] = mobile
        pincode = self._extract_pincode(combined_text)
        if pincode:
            fields["pincode"] = pincode
        return fields

    def _extract_all_dates(self, text: str) -> List[str]:
        """Extract all dates from text."""
        dates = []
        for pattern in self.patterns["date"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        return dates

    def _classify_dates(self, texts: List[str], dates: List[str]) -> Dict[str, str]:
        """
        Classify dates as DOB, DOI, DOE based on context keywords.
        """
        date_fields = {}
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if any(kw in text_lower for kw in self.field_keywords["dob"]):
                for date in dates:
                    if date in text or (i + 1 < len(texts) and date in texts[i + 1]):
                        date_fields["dob"] = date
                        break
            elif any(kw in text_lower for kw in self.field_keywords["doi"]):
                for date in dates:
                    if date in text or (i + 1 < len(texts) and date in texts[i + 1]):
                        date_fields["date_of_issue"] = date
                        break
            elif any(kw in text_lower for kw in self.field_keywords["doe"]):
                for date in dates:
                    if date in text or (i + 1 < len(texts) and date in texts[i + 1]):
                        date_fields["date_of_expiry"] = date
                        break
        if not date_fields and dates:
            date_fields["dob"] = dates[0]
        return date_fields

    def _extract_name(self, texts: List[str]) -> Optional[str]:
        """
        Extract name - usually the first significant capitalized text line.
        """
        for text in texts:
            if len(text) < 3:
                continue
            text_upper = text.upper()
            skip_keywords = ["GOVERNMENT", "INDIA", "CARD", "AADHAAR", "PAN", "LICENSE", "PASSPORT"]
            if any(kw in text_upper for kw in skip_keywords):
                continue
            if text.isupper() and len(text.split()) >= 2:
                return text.strip()
            for keyword in self.field_keywords["name"]:
                if keyword in text.lower():
                    parts = re.split(keyword, text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        name = parts[1].strip().strip(":").strip()
                        if name:
                            return name
        return None

    def _extract_parent_names(self, texts: List[str]) -> Dict[str, str]:
        """Extract father's and mother's name with validation."""
        parent_fields = {}
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for keyword in self.field_keywords["father_name"]:
                if keyword in text_lower:
                    # Look in current or next lines for a valid name
                    for j in range(i, min(i + 3, len(texts))):  # Check up to 2 lines after
                        candidate = texts[j].strip()
                        # Validate: name should have at least 2 words and not contain keywords like "gender"
                        if (len(candidate.split()) >= 2 and 
                            not any(kw in candidate.lower() for kw in self.field_keywords["gender"]) and
                            not candidate.isdigit()):
                            parent_fields["father_name"] = candidate
                            break
                    break
            for keyword in self.field_keywords["mother_name"]:
                if keyword in text_lower:
                    for j in range(i, min(i + 3, len(texts))):
                        candidate = texts[j].strip()
                        if (len(candidate.split()) >= 2 and 
                            not any(kw in candidate.lower() for kw in self.field_keywords["gender"]) and
                            not candidate.isdigit()):
                            parent_fields["mother_name"] = candidate
                            break
                    break
        return parent_fields

    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from text."""
        text_lower = text.lower()
        if "male" in text_lower and "female" not in text_lower:
            return "Male"
        elif "female" in text_lower:
            return "Female"
        elif re.search(r"\bm\b", text_lower):
            return "Male"
        elif re.search(r"\bf\b", text_lower):
            return "Female"
        return None

    def _extract_address(self, texts: List[str]) -> Optional[str]:
        """
        Extract address - usually multiple lines near end of document.
        """
        address_lines = []
        found_address_keyword = False
        for text in texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in self.field_keywords["address"]):
                found_address_keyword = True
                parts = re.split("|".join(self.field_keywords["address"]), text, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    address_lines.append(parts[1].strip().strip(":").strip())
                continue
            if found_address_keyword and len(text) > 10:
                address_lines.append(text.strip())
                if len(address_lines) >= 4:
                    break
        if address_lines:
            return ", ".join(address_lines)
        return None

    def _extract_mobile(self, text: str) -> Optional[str]:
        """Extract mobile number."""
        for pattern in self.patterns["mobile"]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_pincode(self, text: str) -> Optional[str]:
        """Extract pincode."""
        match = re.search(self.patterns["pincode"][0], text)
        if match:
            return match.group(1)
        return None

    def format_output(self, structured_data: Dict[str, Any]) -> str:
        """
        Format structured data into human-readable string.
        """
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("EXTRACTED DOCUMENT DATA")
        output_lines.append("=" * 60)
        doc_type = structured_data.get("document_type", "unknown").upper()
        output_lines.append(f"\nDocument Type: {doc_type}")
        fields = structured_data.get("fields", {})
        if fields:
            output_lines.append("\nExtracted Fields:")
            output_lines.append("-" * 60)
            for key, value in fields.items():
                formatted_key = key.replace("_", " ").title()
                output_lines.append(f"{formatted_key:20s}: {value}")
        metadata = structured_data.get("metadata", {})
        if metadata:
            output_lines.append("\nMetadata:")
            output_lines.append("-" * 60)
            output_lines.append(f"{'Text Regions':20s}: {metadata.get('total_text_regions', 0)}")
            output_lines.append(f"{'Avg Confidence':20s}: {metadata.get('avg_confidence', 0):.2%}")
        raw_text = structured_data.get("raw_text", [])
        if raw_text:
            output_lines.append("\nRaw Extracted Text:")
            output_lines.append("-" * 60)
            for i, text in enumerate(raw_text, 1):
                output_lines.append(f"{i:2d}. {text}")
        output_lines.append("=" * 60)
        return "\n".join(output_lines)

# Singleton instance
_ocr_instance: Optional[OCRExtractor] = None
def get_ocr_extractor() -> OCRExtractor:
    """
    Get or create singleton OCR extractor instance.
    """
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRExtractor()
    return _ocr_instance