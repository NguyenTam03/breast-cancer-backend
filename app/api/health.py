"""
Health check endpoints for BreastCare AI Backend
"""

from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "message": "BreastCare AI API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "debug": settings.DEBUG,
            "database": "MongoDB",
            "upload_dir": settings.UPLOAD_DIR,
            "max_upload_size": settings.MAX_UPLOAD_SIZE
        },
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users", 
            "analysis": "/api/v1/analysis",
            "health": "/api/v1/health"
        }
    }


@router.get("/status")
async def service_status():
    """Service status endpoint"""
    return {
        "api": "online",
        "database": "connected",
        "ml_service": "ready",
        "file_upload": "enabled"
    }
