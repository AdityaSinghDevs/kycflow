# app/services/ocr_extractor.py

"""
EasyOCR Extractor - Optimized for European/US ID cards
FIXED: Separate log file, better error handling
"""

import easyocr
import re
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from configs.config import config
from utils.logger import get_logger

# ✅ FIX: Use separate log file for OCR
logger = get_logger(__name__, log_file="ocr_extractor.log")


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
        gpu: bool = False
    ):
        """
        Initialize EasyOCR.
        
        Args:
            languages: ['en', 'de', 'es', 'pt']. If None, loads from config.
            gpu: Use GPU acceleration if available
        """
        if languages is None:
            languages = config.get("models", "ocr", "languages", default=["en"])
        
        if gpu is None:
            gpu = config.use_gpu
        
        self.languages = languages
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            raise
        
        # Regex patterns
        self.patterns = {
            "date": r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{4})\b",
            "doc_number": r"\b([A-Z0-9]{8,12})\b",
            "passport": r"\b([A-Z]{1,2}\d{7,9})\b",
        }
        
        # Keywords by language
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
        """Minimal preprocessing - CLAHE contrast enhancement."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def extract_text(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> List[str]:
        """Extract all text from image."""
        try:
            results = self.reader.readtext(image, detail=1)
            
            if len(results) < 5:
                logger.debug(f"Only {len(results)} regions, trying preprocessing...")
                preprocessed = self.preprocess_image(image)
                results_preprocessed = self.reader.readtext(preprocessed, detail=1)
                
                if len(results_preprocessed) > len(results):
                    results = results_preprocessed
            
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
        """Detect document type from text content."""
        combined = " ".join(texts).upper()
        
        if any(word in combined for word in ["PASSPORT", "PASSEPORT", "REISEPASS", "PASAPORTE"]):
            return DocumentType.PASSPORT
        
        if any(word in combined for word in ["DRIVING", "FÜHRERSCHEIN", "PERMIS", "CARTEIRA"]):
            return DocumentType.DRIVERS_LICENSE
        
        if any(word in combined for word in ["RESIDENCE", "AUFENTHALT", "RESIDENCIA"]):
            return DocumentType.RESIDENCE_PERMIT
        
        if any(word in combined for word in ["IDENTITY", "AUSWEIS", "IDENTIDAD", "IDENTIDADE"]):
            return DocumentType.NATIONAL_ID
        
        return DocumentType.UNKNOWN

    def extract_field(
        self,
        texts: List[str],
        keywords: List[str],
        pattern: Optional[str] = None
    ) -> Optional[str]:
        """Generic field extraction using keywords and regex."""
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            if any(kw in text_lower for kw in keywords):
                if ":" in text or "/" in text:
                    parts = re.split(r"[:/]", text)
                    if len(parts) > 1:
                        candidate = parts[-1].strip()
                        if len(candidate) > 2:
                            return candidate
                
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
        """Extract full name."""
        name = self.extract_field(texts, self.keywords["name"])
        if name and len(name.split()) >= 2:
            return name
        
        skip_words = {"identity", "card", "ausweis", "passport", "driving", "license"}
        
        candidates = []
        for text in texts:
            words = text.split()
            if len(words) >= 2:
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
                if alpha_ratio > 0.7 and not any(skip in text.lower() for skip in skip_words):
                    candidates.append((text, len(text)))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None

    def extract_dates(self, texts: List[str]) -> Dict[str, Optional[str]]:
        """Extract dates."""
        dates = {"dob": None, "issue_date": None, "expiry_date": None}
        
        all_dates = []
        for text in texts:
            matches = re.finditer(self.patterns["date"], text)
            for match in matches:
                all_dates.append(match.group(1))
        
        if not all_dates:
            return dates
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            if any(kw in text_lower for kw in self.keywords["dob"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["dob"] = date
                            break
            
            elif any(kw in text_lower for kw in self.keywords["expiry"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["expiry_date"] = date
                            break
            
            elif any(kw in text_lower for kw in self.keywords["issue"]):
                for j in range(i, min(i + 2, len(texts))):
                    for date in all_dates:
                        if date in texts[j]:
                            dates["issue_date"] = date
                            break
        
        if not dates["dob"] and all_dates:
            dates["dob"] = all_dates[0]
        
        return dates

    def extract_document_number(
        self,
        texts: List[str],
        doc_type: DocumentType
    ) -> Optional[str]:
        """Extract document number."""
        doc_num = self.extract_field(
            texts,
            self.keywords["doc_number"],
            self.patterns["doc_number"]
        )
        if doc_num:
            return doc_num
        
        if doc_type == DocumentType.PASSPORT:
            combined = " ".join(texts)
            match = re.search(self.patterns["passport"], combined)
            if match:
                return match.group(1)
        
        combined = " ".join(texts)
        match = re.search(self.patterns["doc_number"], combined)
        if match:
            return match.group(1)
        
        return None

    def extract_gender(self, texts: List[str]) -> Optional[str]:
        """Extract gender."""
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
        """Main extraction method - returns structured OCR result."""
        logger.info("Starting structured OCR extraction...")
        
        texts = self.extract_text(image, confidence_threshold)
        
        if not texts:
            logger.warning("No text extracted from image")
            return OCRResult(
                document_type=DocumentType.UNKNOWN.value,
                confidence=0.0,
                extracted_text=""
            )
        
        doc_type = self.detect_document_type(texts)
        logger.info(f"Detected document type: {doc_type.value}")
        
        result = OCRResult(
            document_type=doc_type.value,
            confidence=0.8,
            extracted_text=" | ".join(texts)
        )
        
        result.full_name = self.extract_name(texts)
        result.document_number = self.extract_document_number(texts, doc_type)
        result.nationality = self.extract_field(texts, self.keywords["nationality"])
        result.gender = self.extract_gender(texts)
        
        dates = self.extract_dates(texts)
        result.date_of_birth = dates["dob"]
        result.issue_date = dates["issue_date"]
        result.expiry_date = dates["expiry_date"]
        
        field_count = sum(1 for v in asdict(result).values() if v)
        logger.info(f"OCR complete. Extracted {field_count} fields")
        
        return result


# ============================================================================
# Singleton Pattern
# ============================================================================

_ocr_instance: Optional[OCRExtractor] = None
_ocr_lock = threading.Lock()


def get_ocr_extractor() -> OCRExtractor:
    """Thread-safe singleton getter."""
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:
                _ocr_instance = OCRExtractor(
                    languages=config.get("models", "ocr", "languages", default=["en"]),
                    gpu=config.use_gpu
                )
                logger.info("OCR extractor singleton created")
    return _ocr_instance


def reset_ocr_extractor() -> None:
    """Reset singleton (for testing)."""
    global _ocr_instance
    _ocr_instance = None
    logger.info("OCR extractor singleton reset")