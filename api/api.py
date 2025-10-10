# api/api.py

"""
FastAPI application for KYC verification service.
FIXED: Missing config import + response format for frontend
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

# ✅ FIX #1: Added missing import
from configs.config import config
from api.schemas import (
    KYCVerificationResponse,
    OCROnlyResponse,
    ErrorResponse,
    HealthCheckResponse,
    ModelStatus,
    VerificationStatus,
    OCRData,
    OCRFields,
    FaceMatchData,
    SimilarityMetrics,
)
from app.services.face_detector_id import get_face_detector, YuNetFaceDetector
from app.services.face_matcher import get_face_matcher, InsightFaceMatcher
from app.services.ocr_extractor import get_ocr_extractor, OCRExtractor
from utils.logger import get_logger

logger = get_logger(__name__, log_file="api.log")

# Global service instances
face_detector: Optional[YuNetFaceDetector] = None
face_matcher: Optional[InsightFaceMatcher] = None
ocr_extractor: Optional[OCRExtractor] = None

# Semaphore to limit concurrent processing
MAX_CONCURRENT = config.get("processing", "max_concurrent_requests", default=10)
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ============================================================================
# Lifespan Event Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    global face_detector, face_matcher, ocr_extractor
    
    logger.info("🚀 Starting KYC Verification Service...")
    
    try:
        logger.info("Loading face detector...")
        face_detector = await asyncio.to_thread(get_face_detector)
        logger.info("✓ Face detector loaded")
        
        logger.info("Loading face matcher...")
        face_matcher = await asyncio.to_thread(get_face_matcher)
        logger.info("✓ Face matcher loaded")
        
        logger.info("Loading OCR extractor...")
        ocr_extractor = await asyncio.to_thread(get_ocr_extractor)
        logger.info("✓ OCR extractor loaded")
        
        logger.info("✅ All models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down service...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=config.get("project", "name", default="KYC Verification Service"),
    description=config.get("project", "description", default="KYC with face matching and OCR"),
    version=config.get("project", "version", default="1.0.0"),
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred. Please try again.",
            details={"type": exc.__class__.__name__}
        ).model_dump()
    )


# ============================================================================
# Utility Functions
# ============================================================================

async def read_upload_file(upload_file: UploadFile) -> np.ndarray:
    """Read and validate uploaded image file."""
    max_size = config.max_upload_size
    content = await upload_file.read()
    
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {max_size / (1024*1024):.1f}MB"
        )
    
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Supported: JPG, PNG"
            )
        
        max_dim = config.get("upload", "image_max_dimension", default=4096)
        h, w = image.shape[:2]
        if h > max_dim or w > max_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too large. Max dimension: {max_dim}px"
            )
        
        return image
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Image decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image"
        )


def determine_verification_status(face_verified: bool, ocr_confidence: float) -> VerificationStatus:
    """Determine overall verification status."""
    if not face_verified:
        return VerificationStatus.REJECTED
    
    if ocr_confidence < 0.5:
        return VerificationStatus.PENDING
    
    return VerificationStatus.APPROVED


# ✅ FIX #2: Calculate overall confidence score
def calculate_confidence_score(face_confidence: float, ocr_confidence: float, face_verified: bool) -> float:
    """
    Calculate overall confidence combining face match and OCR quality.
    Frontend expects this field.
    """
    if not face_verified:
        return 0.0
    
    # Weighted average: face match 60%, OCR 40%
    return (face_confidence * 0.6) + (ocr_confidence * 0.4)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": config.get("project", "name"),
        "version": config.get("project", "version"),
        "status": "running",
        "docs": "/api/v1/docs"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check with model status."""
    models_status = {
        "face_detector": ModelStatus(
            loaded=face_detector is not None,
            name="yunet",
            error=None if face_detector else "Not loaded"
        ),
        "face_matcher": ModelStatus(
            loaded=face_matcher is not None,
            name="insightface",
            error=None if face_matcher else "Not loaded"
        ),
        "ocr_extractor": ModelStatus(
            loaded=ocr_extractor is not None,
            name="easyocr",
            error=None if ocr_extractor else "Not loaded"
        ),
    }
    
    all_loaded = all(m.loaded for m in models_status.values())
    
    return HealthCheckResponse(
        status="healthy" if all_loaded else "unhealthy",
        version=config.get("project", "version", default="1.0.0"),
        models=models_status
    )


@app.post(
    "/api/v1/kyc/verify",
    response_model=KYCVerificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"]
)
async def verify_kyc(
    id_document: UploadFile = File(..., description="ID card/passport image"),
    selfie_image: UploadFile = File(..., description="Selfie photo")
):
    """
    Complete KYC verification: face detection + matching + OCR.
    """
    start_time = time.time()
    
    if not all([face_detector, face_matcher, ocr_extractor]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Models still loading."
        )
    
    async with processing_semaphore:
        try:
            logger.info("Reading uploaded files...")
            id_image = await read_upload_file(id_document)
            selfie_img = await read_upload_file(selfie_image)
            
            # Detect faces
            logger.info("Detecting faces...")
            id_face_result, selfie_face_result = await asyncio.gather(
                asyncio.to_thread(face_detector.detect_and_extract, id_image),
                asyncio.to_thread(face_detector.detect_and_extract, selfie_img)
            )
            
            if id_face_result is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in ID document"
                )
            
            if selfie_face_result is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in selfie image"
                )
            
            # ✅ OPTIMIZATION: Run face matching and OCR in parallel
            logger.info("Running face matching and OCR in parallel...")
            match_result, ocr_result = await asyncio.gather(
                asyncio.to_thread(
                    face_matcher.verify,
                    id_face_result.face_crop,
                    selfie_face_result.face_crop
                ),
                asyncio.to_thread(
                    ocr_extractor.extract_structured,
                    id_image
                )
            )
            
            # Determine verification status
            verification_status = determine_verification_status(
                face_verified=match_result.verified,
                ocr_confidence=ocr_result.confidence
            )
            
            # ✅ FIX #2: Calculate overall confidence score for frontend
            confidence_score = calculate_confidence_score(
                face_confidence=match_result.confidence,
                ocr_confidence=ocr_result.confidence,
                face_verified=match_result.verified
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            response = KYCVerificationResponse(
                verification_status=verification_status,
                confidence_score=confidence_score,  # ✅ Added for frontend
                face_match_score=match_result.confidence,
                ocr_data=OCRData(
                    document_type=ocr_result.document_type,
                    confidence=ocr_result.confidence,
                    extracted_text=ocr_result.extracted_text,
                    fields=OCRFields(**{
                        k: v for k, v in ocr_result.to_dict().items()
                        if k in OCRFields.model_fields
                    })
                ),
                processing_time_ms=processing_time_ms,
                face_verification_details=FaceMatchData(
                    verified=match_result.verified,
                    confidence=match_result.confidence,
                    similarity_metrics=SimilarityMetrics(
                        cosine_similarity=match_result.cosine_similarity,
                        euclidean_distance=match_result.euclidean_distance
                    ),
                    threshold_used=match_result.threshold_used,
                    message=match_result.message
                )
            )
            
            logger.info(f"✓ Verification complete: {verification_status.value} ({processing_time_ms}ms)")
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Verification error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Verification failed: {str(e)}"
            )


@app.post(
    "/api/v1/kyc/ocr",
    response_model=OCROnlyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"]
)
async def extract_ocr(
    document_image: UploadFile = File(..., description="ID card/document image")
):
    """OCR-only endpoint."""
    start_time = time.time()
    
    if not ocr_extractor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service not ready"
        )
    
    async with processing_semaphore:
        try:
            image = await read_upload_file(document_image)
            
            logger.info("Extracting OCR data...")
            ocr_result = await asyncio.to_thread(
                ocr_extractor.extract_structured,
                image
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            response = OCROnlyResponse(
                ocr_data=OCRData(
                    document_type=ocr_result.document_type,
                    confidence=ocr_result.confidence,
                    extracted_text=ocr_result.extracted_text,
                    fields=OCRFields(**{
                        k: v for k, v in ocr_result.to_dict().items()
                        if k in OCRFields.model_fields
                    })
                ),
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"✓ OCR extraction complete ({processing_time_ms}ms)")
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OCR error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR extraction failed: {str(e)}"
            )