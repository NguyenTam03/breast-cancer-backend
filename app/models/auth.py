"""
Authentication models for BreastCare AI Backend
"""

from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum
from .user import UserRole, DoctorInfo


class Platform(str, Enum):
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"


class DeviceInfo(BaseModel):
    """Device information"""
    platform: Platform
    deviceId: str
    appVersion: str


class RefreshToken(Document):
    """Refresh token document model"""
    userId: PydanticObjectId
    refreshToken: str  # Hashed token
    deviceInfo: DeviceInfo
    expiresAt: datetime
    isActive: bool = True
    createdAt: datetime = datetime.utcnow()
    lastUsed: datetime = datetime.utcnow()
    
    class Settings:
        name = "refresh_tokens"
        
    def __repr__(self) -> str:
        return f"<RefreshToken {self.userId}>"
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.utcnow() > self.expiresAt
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid (active and not expired)"""
        return self.isActive and not self.is_expired
    
    async def deactivate(self):
        """Deactivate the refresh token"""
        self.isActive = False
        await self.save()
    
    async def update_last_used(self):
        """Update last used timestamp"""
        self.lastUsed = datetime.utcnow()
        await self.save()
    
    @classmethod
    async def cleanup_expired_tokens(cls):
        """Remove expired tokens from database"""
        await cls.find(cls.expiresAt < datetime.utcnow()).delete()
    
    @classmethod
    async def revoke_user_tokens(cls, user_id: PydanticObjectId):
        """Revoke all tokens for a specific user"""
        await cls.find(cls.userId == user_id).update({"$set": {"isActive": False}})


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds


class LoginRequest(BaseModel):
    """Login request model"""
    email: str
    password: str
    deviceInfo: DeviceInfo


class RegisterRequest(BaseModel):
    """User registration request model"""
    email: str
    password: str
    firstName: str
    lastName: str
    role: UserRole = UserRole.PATIENT  # Default role is patient
    doctorInfo: Optional[DoctorInfo] = None  # Required if role is doctor
    deviceInfo: DeviceInfo


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str
