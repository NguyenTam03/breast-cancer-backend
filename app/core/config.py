"""
Configuration settings for BreastCare AI Backend
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    MONGODB_URL: str = ""
    DATABASE_NAME: str = "breastcare"
    
    # Security
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 1 day (24 hours * 60 minutes)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30  # 30 days
    
    # App Settings
    APP_NAME: str = "BreastCare AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    
    # ML Model Settings
    MODEL_PATH: str = "./models"
    FEATURE_EXTRACTOR_MODEL: str = "feature_extractor.h5"
    GWO_MODEL: str = "model_gwo_selected_feature.h5"
    GWO_FEATURE_INDICES: str = "gwo_feature_indices.npy"
    IMAGE_TARGET_SIZE: int = 224
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:19006",  # Expo dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:19006"
    ]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
