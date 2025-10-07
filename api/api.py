# api/api.py

"""
FastAPI application for KYC verification service.
Production-ready with async support, proper error handling, and Ballerine frontend integration.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

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
from app.services.face_detector import get_face_detector, YuNetFaceDetector
from app.services.face_matcher import get_face_matcher, InsightFaceMatcher
from app.services.ocr_extractor import get_ocr_extractor, OCRExtractor
from configs.config import config

# Setup logging
logger = logging.getLogger(__name__)

# Global service instances (loaded at startup)
face_detector: Optional[YuNetFaceDetector] = None
face_matcher: Optional[InsightFaceMatcher] = None
ocr_extractor: Optional[OCRExtractor] = None

# Semaphore to limit concurrent processing (prevent memory overload)
MAX_CONCURRENT = config.get("processing", "max_concurrent_requests", default=10)
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ============================================================================
# Lifespan Event Handler - Model Loading/Cleanup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown logic for loading models.
    This ensures models are loaded once at startup, not per-request.
    """
    global face_detector, face_matcher, ocr_extractor
    
    logger.info("🚀 Starting KYC Verification Service...")
    
    # Load models at startup (in thread pool to not block)
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
    
    yield  # Application runs here
    
    # Cleanup (if needed)
    logger.info("Shutting down service...")


# ============================================================================
# FastAPI App Initialization
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
    """
    Read UploadFile and convert to OpenCV image.
    Validates file size and format.
    """
    # Check file size
    max_size = config.max_upload_size
    content = await upload_file.read()
    
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {max_size / (1024*1024):.1f}MB"
        )
    
    # Decode image
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Supported: JPG, PNG"
            )
        
        # Check dimensions
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
    """
    Determine overall verification status based on face match and OCR quality.
    Simple logic for freelance project - can be made more complex later.
    """
    if not face_verified:
        return VerificationStatus.REJECTED
    
    # If face matches but OCR confidence is low, mark as pending for manual review
    if ocr_confidence < 0.5:
        return VerificationStatus.PENDING
    
    return VerificationStatus.APPROVED


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - service info."""
    return {
        "service": config.get("project", "name"),
        "version": config.get("project", "version"),
        "status": "running",
        "docs": "/api/v1/docs"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and model loading state.
    """
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
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["KYC"]
)
async def verify_kyc(
    id_document: UploadFile = File(..., description="ID card/passport image"),
    selfie_image: UploadFile = File(..., description="Selfie photo for face matching")
):
    """
    Complete KYC verification workflow:
    1. Detect faces in ID document and selfie
    2. Match faces using InsightFace
    3. Extract text from ID using OCR
    4. Return verification result
    
    This is the main endpoint the Ballerine frontend will call.
    """
    start_time = time.time()
    
    # Validate models are loaded
    if not all([face_detector, face_matcher, ocr_extractor]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Models still loading."
        )
    
    # Rate limiting via semaphore
    async with processing_semaphore:
        try:
            # Read and decode images
            logger.info("Reading uploaded files...")
            id_image = await read_upload_file(id_document)
            selfie_img = await read_upload_file(selfie_image)
            
            # Step 1: Detect faces (run in thread pool)
            logger.info("Detecting faces...")
            id_face_result = await asyncio.to_thread(
                face_detector.detect_and_extract,
                id_image
            )
            selfie_face_result = await asyncio.to_thread(
                face_detector.detect_and_extract,
                selfie_img
            )
            
            # Validate face detection
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
            
            # Step 2: Face matching
            logger.info("Matching faces...")
            match_result = await asyncio.to_thread(
                face_matcher.verify,
                id_face_result.face_crop,
                selfie_face_result.face_crop
            )
            
            # Step 3: OCR extraction (parallel with face matching could be optimized)
            logger.info("Extracting text from ID...")
            ocr_result = await asyncio.to_thread(
                ocr_extractor.extract_structured,
                id_image
            )
            
            # Step 4: Determine verification status
            verification_status = determine_verification_status(
                face_verified=match_result.verified,
                ocr_confidence=ocr_result.confidence
            )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            response = KYCVerificationResponse(
                verification_status=verification_status,
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
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["KYC"]
)
async def extract_ocr(
    document_image: UploadFile = File(..., description="ID card/document image")
):
    """
    OCR-only endpoint.
    Extracts text from document without face verification.
    Useful for debugging or partial verification workflows.
    """
    start_time = time.time()
    
    if not ocr_extractor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service not ready"
        )
    
    async with processing_semaphore:
        try:
            # Read image
            image = await read_upload_file(document_image)
            
            # Extract OCR
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


# ============================================================================
# Run with: uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================