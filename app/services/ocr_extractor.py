# app/services/ocr_extractor.py

import easyocr
import re
from typing import Dict, List, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from configs.config import config

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types."""
    AADHAAR = "aadhaar"
    PAN = "pan"
    DRIVING_LICENSE = "driving_license"
    PASSPORT = "passport"
    VOTER_ID = "voter_id"
    UNKNOWN = "unknown"


@dataclass
class OCRResult:
    """Structured OCR extraction result."""
    document_type: str
    confidence: float
    
    # Personal details
    name: Optional[str] = None
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    
    # Dates
    date_of_birth: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None
    
    # Document identifiers
    document_number: Optional[str] = None
    
    # Contact & Address
    address: Optional[str] = None
    pincode: Optional[str] = None
    mobile: Optional[str] = None
    
    # Additional fields
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    
    # Metadata
    raw_text: List[str] = None
    language_detected: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        data = asdict(self)
        # Remove None values for cleaner API response
        return {k: v for k, v in data.items() if v is not None}


class ImprovedOCRExtractor:
    """
    Enhanced OCR extractor with better Hindi support and field extraction.
    Optimized for Indian ID documents.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = True,
    ):
        """
        Initialize EasyOCR with multi-language support.

        Args:
            languages: List of language codes. Default: ['en', 'hi'] for English + Hindi
            gpu: Use GPU acceleration if available
        """
        if languages is None:
            languages = config.get("models", "ocr", "languages", default=["en", "hi"])

        self.languages = languages
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

        # Enhanced regex patterns
        self.patterns = {
            "aadhaar": [
                r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b",  # 1234 5678 9012 or 1234-5678-9012
            ],
            "pan": [
                r"\b([A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z])\b",  # More strict PAN pattern
            ],
            "driving_license": [
                r"\b([A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{11})\b",  # HR-06-19850034761
                r"\b([A-Z]{2}\d{13})\b",  # HR0619850034761
            ],
            "passport": [
                r"\b([A-Z]\d{7})\b",
            ],
            "voter_id": [
                r"\b([A-Z]{3}\d{7})\b",
            ],
            "date": [
                r"\b(\d{2}[\-/\.]\d{2}[\-/\.]\d{4})\b",  # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
                r"\b(\d{4}[\-/\.]\d{2}[\-/\.]\d{2})\b",  # YYYY-MM-DD
                r"\b(\d{1,2}[\-/\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\-/\s]+\d{4})\b",  # 25 Jun 2003
            ],
            "pincode": [
                r"\b(\d{6})\b",
            ],
            "mobile": [
                r"\b(\+91[\s\-]?)?([6-9]\d{9})\b",
            ],
            "blood_group": [
                r"\b([ABO][\+\-]|AB[\+\-])\b",
            ],
        }

        # Enhanced keywords with Hindi transliterations
        self.keywords = {
            "name": [
                "name", "naam", "नाम", "holder", "card holder",
            ],
            "father_name": [
                "father", "father's", "s/o", "son of", "pita", "पिता",
            ],
            "mother_name": [
                "mother", "mother's", "d/o", "daughter of", "w/o", "wife of", "mata", "माता",
            ],
            "dob": [
                "date of birth", "dob", "birth", "d.o.b", "janm", "जन्म",
            ],
            "doi": [
                "date of issue", "issue", "issued", "doi", "d.o.i",
            ],
            "doe": [
                "date of expiry", "expiry", "valid", "doe", "d.o.e", "valid till", "valid upto",
            ],
            "address": [
                "address", "pata", "पता", "house", "residence",
            ],
            "gender": [
                "sex", "gender", "male", "female", "ling", "लिंग", "m/f",
            ],
            "document_number": [
                "number", "no", "id", "card no", "document no",
            ],
        }

        # Document-specific header keywords for better detection
        self.doc_headers = {
            DocumentType.AADHAAR: [
                "aadhaar", "आधार", "uidai", "government of india", "भारत सरकार",
            ],
            DocumentType.PAN: [
                "income tax", "pan", "permanent account", "आयकर",
            ],
            DocumentType.DRIVING_LICENSE: [
                "driving licence", "driving license", "ड्राइविंग", "transport",
            ],
            DocumentType.PASSPORT: [
                "passport", "republic of india", "पासपोर्ट",
            ],
            DocumentType.VOTER_ID: [
                "election", "voter", "electoral", "मतदाता", "चुनाव",
            ],
        }

    def preprocess_image(self, image: np.ndarray, mode: str = "light") -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image
            mode: "light", "heavy", "denoise", or "watermark"
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if mode == "light":
            # Minimal preprocessing - just enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        elif mode == "heavy":
            # Aggressive preprocessing for poor quality images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        elif mode == "denoise":
            # Strong denoising for watermarked/noisy images
            denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
            
            # Morphological operations to remove fine patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(morphed)
            
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        elif mode == "watermark":
            # Specialized for watermarked documents (like your voter ID)
            # Bilateral filter preserves edges while removing texture
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Strong CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(bilateral)
            
            # Morphological opening to remove fine patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        
        else:
            # No preprocessing
            return image

    def extract_text_regions(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.3,
        preprocess: str = "light",
    ) -> List[Dict[str, Any]]:
        """
        Extract text with bounding boxes and confidence scores.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for extraction
            preprocess: Preprocessing mode - "none", "light", or "heavy"

        Returns:
            List of dicts with {bbox, text, confidence, language}
        """
        try:
            # Try multiple preprocessing strategies
            results_by_mode = {}
            
            # Try with original image first
            logger.info("Extracting text from original image...")
            results_original = self.reader.readtext(image, detail=1)
            results_by_mode["original"] = len(results_original)
            
            # Try with preprocessing if requested
            if preprocess != "none":
                logger.info(f"Extracting text with '{preprocess}' preprocessing...")
                processed_img = self.preprocess_image(image, mode=preprocess)
                results_preprocessed = self.reader.readtext(processed_img, detail=1)
                results_by_mode[preprocess] = len(results_preprocessed)
                
                # Use whichever gives more results
                if len(results_preprocessed) > len(results_original):
                    logger.info(f"Using preprocessed results ({len(results_preprocessed)} vs {len(results_original)} regions)")
                    results = results_preprocessed
                else:
                    logger.info(f"Using original results ({len(results_original)} vs {len(results_preprocessed)} regions)")
                    results = results_original
            else:
                results = results_original

            # Filter and structure results
            text_regions = []
            for bbox, text, conf in results:
                if conf >= confidence_threshold:
                    # Detect language of text region
                    lang = self._detect_text_language(text)
                    
                    text_regions.append({
                        "bbox": bbox,
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "language": lang,
                    })

            logger.info(f"Extracted {len(text_regions)} text regions (threshold: {confidence_threshold})")
            return text_regions

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise

    def _detect_text_language(self, text: str) -> str:
        """Detect if text is Hindi or English."""
        # Check for Devanagari Unicode range
        if any('\u0900' <= char <= '\u097F' for char in text):
            return "hi"
        return "en"

    def extract_structured(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> OCRResult:
        """
        Extract and parse structured data from ID document.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for text extraction

        Returns:
            OCRResult dataclass with all extracted fields
        """
        logger.info("Starting structured data extraction...")

        # Extract text regions
        text_regions = self.extract_text_regions(image, confidence_threshold)

        if not text_regions:
            logger.warning("No text extracted above confidence threshold")
            return OCRResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                raw_text=[],
                language_detected=[],
            )

        # Extract texts and compute average confidence
        texts = [region["text"] for region in text_regions]
        confidences = [region["confidence"] for region in text_regions]
        languages = list(set([region["language"] for region in text_regions]))
        avg_confidence = sum(confidences) / len(confidences)

        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"Languages detected: {languages}")

        # Detect document type
        doc_type = self._detect_document_type(texts)
        logger.info(f"Detected document type: {doc_type}")

        # Initialize result
        result = OCRResult(
            document_type=doc_type,
            confidence=round(avg_confidence, 3),
            raw_text=texts,
            language_detected=languages,
        )

        # Extract document number
        result.document_number = self._extract_document_number(texts, doc_type)

        # Extract dates
        dates = self._extract_dates(texts)
        result.date_of_birth = dates.get("dob")
        result.date_of_issue = dates.get("doi")
        result.date_of_expiry = dates.get("doe")

        # Extract personal details
        result.name = self._extract_name(text_regions)
        result.father_name = self._extract_parent_name(texts, "father")
        result.mother_name = self._extract_parent_name(texts, "mother")
        
        # Extract contact details
        result.address = self._extract_address(texts)
        result.mobile = self._extract_mobile(texts)
        result.pincode = self._extract_pincode(texts)
        
        # Extract other fields
        result.gender = self._extract_gender(texts)
        result.blood_group = self._extract_blood_group(texts)

        logger.info(f"Extraction complete. Fields found: {sum(1 for k, v in result.to_dict().items() if v is not None and k not in ['raw_text', 'language_detected'])}")

        return result

    def _detect_document_type(self, texts: List[str]) -> str:
        """Enhanced document type detection with header matching."""
        combined_text = " ".join(texts).upper()

        # Score-based detection
        scores = {doc_type: 0 for doc_type in DocumentType}

        # Check headers (high weight)
        for doc_type, headers in self.doc_headers.items():
            for header in headers:
                if header.upper() in combined_text:
                    scores[doc_type] += 3

        # Check document number patterns (medium weight)
        for doc_type, patterns in self.patterns.items():
            if doc_type in [dt.value for dt in DocumentType]:
                for pattern in patterns:
                    if re.search(pattern, combined_text):
                        scores[DocumentType(doc_type)] += 2

        # Return highest scoring type
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0].value if best_match[1] > 0 else DocumentType.UNKNOWN

    def _extract_document_number(self, texts: List[str], doc_type: str) -> Optional[str]:
        """Extract document number based on type."""
        combined_text = " ".join(texts)

        if doc_type in self.patterns:
            for pattern in self.patterns[doc_type]:
                match = re.search(pattern, combined_text)
                if match:
                    # Clean up the number (remove extra spaces/hyphens for Aadhaar)
                    number = match.group(1)
                    if doc_type == DocumentType.AADHAAR:
                        number = re.sub(r'[\s\-]', '', number)  # Remove spaces/hyphens
                        number = f"{number[:4]} {number[4:8]} {number[8:]}"  # Format: 1234 5678 9012
                    logger.info(f"Extracted {doc_type} number: {number}")
                    return number

        return None

    def _extract_dates(self, texts: List[str]) -> Dict[str, Optional[str]]:
        """Extract and classify all dates."""
        dates = {"dob": None, "doi": None, "doe": None}
        
        # Find all dates first
        all_dates = []
        combined_text = " ".join(texts)
        for pattern in self.patterns["date"]:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                all_dates.append(match.group(1))

        if not all_dates:
            return dates

        # Classify by context
        for i, text in enumerate(texts):
            text_lower = text.lower()

            # Check for DOB keywords
            if any(kw in text_lower for kw in self.keywords["dob"]):
                # Look in current or next 2 lines
                for j in range(i, min(i + 3, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["dob"] = self._normalize_date(date)
                            logger.info(f"Extracted DOB: {dates['dob']}")
                            break

            # Check for DOI keywords
            elif any(kw in text_lower for kw in self.keywords["doi"]):
                for j in range(i, min(i + 3, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["doi"] = self._normalize_date(date)
                            logger.info(f"Extracted DOI: {dates['doi']}")
                            break

            # Check for DOE keywords
            elif any(kw in text_lower for kw in self.keywords["doe"]):
                for j in range(i, min(i + 3, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["doe"] = self._normalize_date(date)
                            logger.info(f"Extracted DOE: {dates['doe']}")
                            break

        # If DOB not found but dates exist, use first one
        if not dates["dob"] and all_dates:
            dates["dob"] = self._normalize_date(all_dates[0])

        return dates

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to DD-MM-YYYY format."""
        # Already in DD-MM-YYYY or similar
        if re.match(r"\d{2}[\-/\.]\d{2}[\-/\.]\d{4}", date_str):
            return date_str.replace("/", "-").replace(".", "-")
        
        # YYYY-MM-DD to DD-MM-YYYY
        if re.match(r"\d{4}[\-/\.]\d{2}[\-/\.]\d{2}", date_str):
            parts = re.split(r"[\-/\.]", date_str)
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        
        # Handle text dates (25 Jun 2003)
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        for month, num in month_map.items():
            if month in date_str.lower():
                parts = re.split(r"[\s\-/]+", date_str)
                day = parts[0].zfill(2)
                year = parts[2]
                return f"{day}-{num}-{year}"
        
        return date_str

    def _extract_name(self, text_regions: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract name with better logic for English and Hindi.
        """
        # Strategy 1: Look for name keywords
        for i, region in enumerate(text_regions):
            text_lower = region["text"].lower()
            
            if any(kw in text_lower for kw in self.keywords["name"]):
                # Name is likely in next region
                if i + 1 < len(text_regions):
                    candidate = text_regions[i + 1]["text"]
                    # Validate: should be 2+ words, mostly alphabetic
                    if len(candidate.split()) >= 2 and sum(c.isalpha() or c.isspace() for c in candidate) / len(candidate) > 0.7:
                        logger.info(f"Extracted name (by keyword): {candidate}")
                        return candidate.strip()

        # Strategy 2: Find first long capitalized text (likely name)
        skip_words = ["government", "india", "aadhaar", "pan", "card", "election", "passport", 
                      "भारत", "सरकार", "आधार"]
        
        for region in text_regions:
            text = region["text"].strip()
            text_upper = text.upper()
            
            # Skip headers and short texts
            if len(text) < 5 or any(skip in text_upper for skip in skip_words):
                continue
            
            # Check if it's a name-like text
            words = text.split()
            if 2 <= len(words) <= 5:  # Names typically 2-5 words
                # Check if mostly alphabetic
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
                if alpha_ratio > 0.7:
                    logger.info(f"Extracted name (by pattern): {text}")
                    return text

        return None

    def _extract_parent_name(self, texts: List[str], parent_type: str) -> Optional[str]:
        """Extract father's or mother's name."""
        keywords = self.keywords.get(f"{parent_type}_name", [])
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            if any(kw in text_lower for kw in keywords):
                # Look in next 1-2 lines
                for j in range(i + 1, min(i + 3, len(texts))):
                    candidate = texts[j].strip()
                    # Validate: should be name-like
                    if len(candidate.split()) >= 2 and sum(c.isalpha() or c.isspace() for c in candidate) / len(candidate) > 0.6:
                        logger.info(f"Extracted {parent_type} name: {candidate}")
                        return candidate

        return None

    def _extract_address(self, texts: List[str]) -> Optional[str]:
        """Extract address with better multi-line handling."""
        address_lines = []
        collecting = False
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Start collecting when we see address keyword
            if any(kw in text_lower for kw in self.keywords["address"]):
                collecting = True
                # Check if address starts on same line
                if ":" in text:
                    addr_start = text.split(":", 1)[1].strip()
                    if len(addr_start) > 5:
                        address_lines.append(addr_start)
                continue
            
            # Collect subsequent lines
            if collecting and len(text) > 10:
                # Stop if we hit another section (keywords for other fields)
                stop_keywords = ["gender", "dob", "issue", "expiry", "blood"]
                if any(kw in text_lower for kw in stop_keywords):
                    break
                
                address_lines.append(text.strip())
                
                # Stop after 3-4 lines
                if len(address_lines) >= 4:
                    break

        if address_lines:
            address = ", ".join(address_lines)
            logger.info(f"Extracted address: {address[:50]}...")
            return address

        return None

    def _extract_mobile(self, texts: List[str]) -> Optional[str]:
        """Extract mobile number."""
        combined_text = " ".join(texts)
        for pattern in self.patterns["mobile"]:
            match = re.search(pattern, combined_text)
            if match:
                # Return last group (phone number without +91)
                mobile = match.group(2) if match.lastindex >= 2 else match.group(1)
                logger.info(f"Extracted mobile: {mobile}")
                return mobile
        return None

    def _extract_pincode(self, texts: List[str]) -> Optional[str]:
        """Extract pincode from address region."""
        # Look for 6-digit number near end of text
        for text in reversed(texts):
            match = re.search(self.patterns["pincode"][0], text)
            if match:
                pincode = match.group(1)
                logger.info(f"Extracted pincode: {pincode}")
                return pincode
        return None

    def _extract_gender(self, texts: List[str]) -> Optional[str]:
        """Extract gender."""
        combined_text = " ".join(texts).lower()
        
        if "female" in combined_text or "महिला" in combined_text:
            return "Female"
        elif "male" in combined_text and "female" not in combined_text or "पुरुष" in combined_text:
            return "Male"
        elif re.search(r"\bm\b", combined_text):
            return "Male"
        elif re.search(r"\bf\b", combined_text):
            return "Female"
        
        return None

    def _extract_blood_group(self, texts: List[str]) -> Optional[str]:
        """Extract blood group."""
        combined_text = " ".join(texts).upper()
        match = re.search(self.patterns["blood_group"][0], combined_text)
        if match:
            blood_group = match.group(1)
            logger.info(f"Extracted blood group: {blood_group}")
            return blood_group
        return None


# Singleton instance
_ocr_instance: Optional[ImprovedOCRExtractor] = None


def get_ocr_extractor() -> ImprovedOCRExtractor:
    """Get or create singleton OCR extractor instance."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = ImprovedOCRExtractor()
    return _ocr_instance