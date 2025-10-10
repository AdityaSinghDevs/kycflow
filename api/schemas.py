# api/schemas.py

"""
Pydantic schemas for KYC API - FIXED for Ballerine frontend compatibility.
Added confidence_score field as required by frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class VerificationStatus(str, Enum):
    """Verification status for KYC workflow."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    ERROR = "error"


# ============================================================================
# OCR Response Models
# ============================================================================

class OCRFields(BaseModel):
    """Structured fields extracted from ID document."""
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    place_of_birth: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None


class OCRData(BaseModel):
    """OCR extraction result."""
    document_type: str = Field(description="Detected document type")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall OCR confidence")
    extracted_text: str = Field(description="Full raw text extracted")
    fields: OCRFields = Field(description="Structured extracted fields")


# ============================================================================
# Face Verification Response Models
# ============================================================================

class SimilarityMetrics(BaseModel):
    """Face similarity metrics."""
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    euclidean_distance: float = Field(ge=0.0)


class FaceMatchData(BaseModel):
    """Face matching verification result."""
    verified: bool = Field(description="Whether faces match")
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence score")
    similarity_metrics: SimilarityMetrics
    threshold_used: float
    message: str


# ============================================================================
# Main KYC Verification Response
# ============================================================================

class KYCVerificationResponse(BaseModel):
    """
    Main KYC verification response - FIXED for Ballerine frontend.
    Added confidence_score as top-level field (required by frontend).
    """
    verification_status: VerificationStatus = Field(
        description="Overall verification status"
    )
    
    # ✅ FIX: Added confidence_score for frontend compatibility
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence (weighted: 60% face + 40% OCR)"
    )
    
    face_match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Face matching confidence (0-1)"
    )
    
    ocr_data: OCRData = Field(description="OCR extraction results")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional detailed breakdown
    face_verification_details: Optional[FaceMatchData] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "verification_status": "approved",
                "confidence_score": 0.89,
                "face_match_score": 0.87,
                "ocr_data": {
                    "document_type": "national_id",
                    "confidence": 0.92,
                    "extracted_text": "Full name: John Doe | DOB: 01/01/1990",
                    "fields": {
                        "full_name": "John Doe",
                        "date_of_birth": "01/01/1990",
                        "document_number": "ABC123456",
                        "nationality": "German"
                    }
                },
                "processing_time_ms": 2500,
                "timestamp": "2024-10-11T10:30:00Z",
                "face_verification_details": {
                    "verified": True,
                    "confidence": 0.87,
                    "similarity_metrics": {
                        "cosine_similarity": 0.85,
                        "euclidean_distance": 0.42
                    },
                    "threshold_used": 0.4,
                    "message": "Faces match (85.0% similarity)"
                }
            }
        }


# ============================================================================
# OCR-Only Response
# ============================================================================

class OCROnlyResponse(BaseModel):
    """Response for OCR-only endpoint."""
    ocr_data: OCRData
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "ocr_data": {
                    "document_type": "drivers_license",
                    "confidence": 0.88,
                    "extracted_text": "Name: Jane Smith | DOB: 15/03/1985",
                    "fields": {
                        "full_name": "Jane Smith",
                        "date_of_birth": "15/03/1985"
                    }
                },
                "processing_time_ms": 3200,
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


# ============================================================================
# Error Response
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "No face detected in ID document",
                "details": {"confidence_threshold": 0.6},
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


# ============================================================================
# Health Check Response
# ============================================================================

class ModelStatus(BaseModel):
    """Individual model status."""
    loaded: bool
    name: str
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response with model status."""
    status: str = Field(description="Overall service status: healthy|degraded|unhealthy")
    version: str
    models: Dict[str, ModelStatus]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models": {
                    "face_detector": {"loaded": True, "name": "yunet", "error": None},
                    "face_matcher": {"loaded": True, "name": "insightface", "error": None},
                    "ocr_extractor": {"loaded": True, "name": "easyocr", "error": None}
                },
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }