"""
BreastCare AI - FastAPI Backend Application
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_database
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.analysis import router as analysis_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    try:
        await init_database()
        print("✅ Database connected successfully!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️  App will continue without database")
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered breast cancer diagnosis mobile application backend",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Static files for uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# API routes
app.include_router(health_router, prefix="/api/v1/health", tags=["health"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BreastCare AI API",
        "version": settings.APP_VERSION,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
