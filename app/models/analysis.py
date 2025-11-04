"""
Analysis model for BreastCare AI Backend
"""

from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class PredictionResult(str, Enum):
    BENIGN = "BENIGN"
    MALIGNANT = "MALIGNANT"


class ImageInfo(BaseModel):
    """Image information"""
    originalName: str
    filePath: str
    fileSize: int  # in bytes
    mimeType: str
    dimensions: dict  # {"width": int, "height": int}
    uploadDate: datetime = datetime.utcnow()


class MLResults(BaseModel):
    """Machine learning prediction results"""
    prediction: PredictionResult
    confidence: float  # 0.0 - 1.0
    processingTime: float  # in seconds (can be decimal)
    modelVersion: str = "gwo-cnn-v1.0"
    features: List[float] = []  # GWO selected features
    rawOutput: float  # Raw model output before threshold


class AnalysisMetadata(BaseModel):
    """Analysis metadata"""
    analysisDate: datetime = datetime.utcnow()
    deviceInfo: Optional[str] = None
    appVersion: str
    apiVersion: str = "v1.0"


class Analysis(Document):
    """Analysis document model"""
    userId: PydanticObjectId
    imageInfo: ImageInfo
    mlResults: MLResults
    metadata: AnalysisMetadata
    userNotes: Optional[str] = None
    isBookmarked: bool = False
    tags: List[str] = []
    createdAt: datetime = datetime.utcnow()
    updatedAt: datetime = datetime.utcnow()
    
    class Settings:
        name = "analyses"
        
    def __repr__(self) -> str:
        return f"<Analysis {self.id} - {self.mlResults.prediction}>"
    
    @property
    def confidence_percentage(self) -> float:
        """Get confidence as percentage"""
        return self.mlResults.confidence * 100
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence (>80%)"""
        return self.mlResults.confidence > 0.8
    
    async def add_note(self, note: str):
        """Add or update user note"""
        self.userNotes = note
        self.updatedAt = datetime.utcnow()
        await self.save()
    
    async def toggle_bookmark(self):
        """Toggle bookmark status"""
        self.isBookmarked = not self.isBookmarked
        self.updatedAt = datetime.utcnow()
        await self.save()
    
    async def add_tag(self, tag: str):
        """Add a tag to the analysis"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updatedAt = datetime.utcnow()
            await self.save()
    
    async def remove_tag(self, tag: str):
        """Remove a tag from the analysis"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updatedAt = datetime.utcnow()
            await self.save()
