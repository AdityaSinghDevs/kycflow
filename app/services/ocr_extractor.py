# app/services/ocr_extractor.py

"""
EasyOCR Extractor - Optimized for European/US ID cards
Supports: English, German, Spanish, Portuguese
Simplified for freelancing project - no over-engineering
"""

import easyocr
import re
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="test_yunet.log")



class DocumentType(str, Enum):
    """Supported ID document types."""
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    PASSPORT = "passport"
    RESIDENCE_PERMIT = "residence_permit"
    UNKNOWN = "unknown"


@dataclass
class OCRResult:
    """Structured OCR result for API response."""
    document_type: str
    confidence: float
    
    # Core fields
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    
    # Dates
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    
    # Location
    place_of_birth: Optional[str] = None
    address: Optional[str] = None
    
    # Additional
    gender: Optional[str] = None
    
    # Metadata
    extracted_text: str = ""  # Full raw text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response, omit None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class OCRExtractor:
    """
    Production-ready OCR for European/US IDs.
    Simple, efficient, no redundant preprocessing.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = True
    ):
        """
        Initialize EasyOCR.
        
        Args:
            languages: ['en', 'de', 'es', 'pt']. If None, loads from config.
            gpu: Use GPU acceleration if available
        """
        if languages is None:
            languages = config.get("models", "ocr", "languages", default=["en"])
        
        self.languages = languages
        logger.info(f"Initializing OCR with languages: {languages}")
        
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False
            )
            logger.info("OCR initialized successfully")
        except Exception as e:
            logger.error(f"OCR initialization failed: {e}")
            raise
        
        # Regex patterns - simplified and focused
        self.patterns = {
            "date": r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{4})\b",  # DD.MM.YYYY or DD-MM-YYYY or DD/MM/YYYY
            "doc_number": r"\b([A-Z0-9]{8,12})\b",  # Alphanumeric 8-12 chars
            "passport": r"\b([A-Z]{1,2}\d{7,9})\b",  # P12345678
        }
        
        # Keywords by language - minimal set
        self.keywords = {
            "name": ["name", "nome", "nombre", "nachname", "surname", "apellido"],
            "dob": ["birth", "geburt", "nacimiento", "nascimento", "date of birth", "geburtsdatum"],
            "nationality": ["nationality", "nationalität", "nacionalidad", "nacionalidade"],
            "doc_number": ["number", "nummer", "número", "documento"],
            "expiry": ["expiry", "expires", "válido", "gültig", "validade", "valid until"],
            "issue": ["issue", "issued", "emitido", "ausgestellt"],
            "gender": ["sex", "gender", "geschlecht", "sexo"],
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Minimal preprocessing - CLAHE contrast enhancement only.
        European IDs are usually good quality, no need for heavy processing.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def extract_text(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> List[str]:
        """
        Extract all text from image.
        
        Args:
            image: Input image (BGR)
            confidence_threshold: Min confidence for text regions
            
        Returns:
            List of extracted text strings
        """
        try:
            # Try original first
            results = self.reader.readtext(image, detail=1)
            
            # If poor results, try preprocessing
            if len(results) < 5:
                logger.debug("Poor OCR results, trying with preprocessing...")
                preprocessed = self.preprocess_image(image)
                results_preprocessed = self.reader.readtext(preprocessed, detail=1)
                
                if len(results_preprocessed) > len(results):
                    results = results_preprocessed
            
            # Filter by confidence and extract text
            texts = [
                text.strip()
                for _, text, conf in results
                if conf >= confidence_threshold and len(text.strip()) > 1
            ]
            
            logger.info(f"Extracted {len(texts)} text regions")
            return texts
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []

    def detect_document_type(self, texts: List[str]) -> DocumentType:
        """
        Detect document type from text content.
        Simple keyword matching - works for 90% of cases.
        """
        combined = " ".join(texts).upper()
        
        # Check for passport indicators
        if any(word in combined for word in ["PASSPORT", "PASSEPORT", "REISEPASS", "PASAPORTE"]):
            return DocumentType.PASSPORT
        
        # Check for driver's license
        if any(word in combined for word in ["DRIVING", "FÜHRERSCHEIN", "PERMIS", "CARTEIRA"]):
            return DocumentType.DRIVERS_LICENSE
        
        # Check for residence permit
        if any(word in combined for word in ["RESIDENCE", "AUFENTHALT", "RESIDENCIA"]):
            return DocumentType.RESIDENCE_PERMIT
        
        # Default to national ID
        if any(word in combined for word in ["IDENTITY", "AUSWEIS", "IDENTIDAD", "IDENTIDADE"]):
            return DocumentType.NATIONAL_ID
        
        return DocumentType.UNKNOWN

    def extract_field(
        self,
        texts: List[str],
        keywords: List[str],
        pattern: Optional[str] = None
    ) -> Optional[str]:
        """
        Generic field extraction using keywords and optional regex.
        
        Args:
            texts: All extracted text lines
            keywords: Keywords to find the field
            pattern: Optional regex pattern to validate
            
        Returns:
            Extracted field value or None
        """
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Check if line contains keyword
            if any(kw in text_lower for kw in keywords):
                # Value might be on same line after colon
                if ":" in text or "/" in text:
                    parts = re.split(r"[:/]", text)
                    if len(parts) > 1:
                        candidate = parts[-1].strip()
                        if len(candidate) > 2:
                            return candidate
                
                # Or on next line
                if i + 1 < len(texts):
                    candidate = texts[i + 1].strip()
                    if pattern:
                        match = re.search(pattern, candidate)
                        if match:
                            return match.group(1)
                    elif len(candidate) > 2:
                        return candidate
        
        return None

    def extract_name(self, texts: List[str]) -> Optional[str]:
        """
        Extract full name - usually the largest non-header text.
        """
        # Try keyword-based first
        name = self.extract_field(texts, self.keywords["name"])
        if name and len(name.split()) >= 2:
            return name
        
        # Fallback: find longest multi-word alphabetic text
        skip_words = {"identity", "card", "ausweis", "passport", "driving", "license"}
        
        candidates = []
        for text in texts:
            # Must be 2+ words, mostly alphabetic
            words = text.split()
            if len(words) >= 2:
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
                if alpha_ratio > 0.7 and not any(skip in text.lower() for skip in skip_words):
                    candidates.append((text, len(text)))
        
        if candidates:
            # Return longest candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None

    def extract_dates(self, texts: List[str]) -> Dict[str, Optional[str]]:
        """Extract birth date, issue date, expiry date."""
        dates = {"dob": None, "issue_date": None, "expiry_date": None}
        
        # Find all dates first
        all_dates = []
        for text in texts:
            matches = re.finditer(self.patterns["date"], text)
            for match in matches:
                all_dates.append(match.group(1))
        
        if not all_dates:
            return dates
        
        # Classify by context
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # DOB
            if any(kw in text_lower for kw in self.keywords["dob"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["dob"] = date
                            break
            
            # Expiry
            elif any(kw in text_lower for kw in self.keywords["expiry"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["expiry_date"] = date
                            break
            
            # Issue
            elif any(kw in text_lower for kw in self.keywords["issue"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["issue_date"] = date
                            break
        
        # If DOB not found, assume first date is DOB
        if not dates["dob"] and all_dates:
            dates["dob"] = all_dates[0]
        
        return dates

    def extract_document_number(
        self,
        texts: List[str],
        doc_type: DocumentType
    ) -> Optional[str]:
        """Extract document number based on type."""
        # Try keyword-based extraction
        doc_num = self.extract_field(
            texts,
            self.keywords["doc_number"],
            self.patterns["doc_number"]
        )
        if doc_num:
            return doc_num
        
        # For passport, try passport pattern
        if doc_type == DocumentType.PASSPORT:
            combined = " ".join(texts)
            match = re.search(self.patterns["passport"], combined)
            if match:
                return match.group(1)
        
        # Fallback: find any alphanumeric 8-12 chars
        combined = " ".join(texts)
        match = re.search(self.patterns["doc_number"], combined)
        if match:
            return match.group(1)
        
        return None

    def extract_gender(self, texts: List[str]) -> Optional[str]:
        """Extract gender - simple M/F detection."""
        combined = " ".join(texts).upper()
        
        if re.search(r"\bM\b", combined) and "FEMALE" not in combined:
            return "M"
        elif re.search(r"\bF\b", combined) or "FEMALE" in combined:
            return "F"
        
        return None

    def extract_structured(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> OCRResult:
        """
        Main extraction method - returns structured OCR result.
        
        Args:
            image: Input ID card image (BGR)
            confidence_threshold: Min confidence for text extraction
            
        Returns:
            OCRResult with all extracted fields
        """
        logger.info("Starting structured extraction...")
        
        # Extract all text
        texts = self.extract_text(image, confidence_threshold)
        
        if not texts:
            logger.warning("No text extracted")
            return OCRResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                extracted_text=""
            )
        
        # Detect document type
        doc_type = self.detect_document_type(texts)
        logger.info(f"Detected document type: {doc_type.value}")
        
        # Extract all fields
        result = OCRResult(
            document_type=doc_type.value,
            confidence=0.8,  # EasyOCR doesn't provide overall confidence
            extracted_text=" | ".join(texts)
        )
        
        result.full_name = self.extract_name(texts)
        result.document_number = self.extract_document_number(texts, doc_type)
        result.nationality = self.extract_field(texts, self.keywords["nationality"])
        result.gender = self.extract_gender(texts)
        
        # Extract dates
        dates = self.extract_dates(texts)
        result.date_of_birth = dates["dob"]
        result.issue_date = dates["issue_date"]
        result.expiry_date = dates["expiry_date"]
        
        logger.info(f"Extraction complete. Found {sum(1 for v in asdict(result).values() if v)} fields")
        
        return result


# Singleton instance
import threading

_ocr_instance: Optional[OCRExtractor] = None
_ocr_lock = threading.Lock()

def get_ocr_extractor() -> OCRExtractor:
    """Thread-safe singleton getter."""
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:
                _ocr_instance = OCRExtractor()
                logger.info("OCR extractor singleton created")
    return _ocr_instance


def reset_ocr_extractor() -> None:
    """Reset singleton (for testing)."""
    global _ocr_instance
    _ocr_instance = None