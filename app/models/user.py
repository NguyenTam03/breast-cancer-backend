"""
User model for BreastCare AI Backend
"""

from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class Language(str, Enum):
    VI = "vi"
    EN = "en"


class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"


class UserProfile(BaseModel):
    """User profile information"""
    firstName: str
    lastName: str
    dateOfBirth: Optional[datetime] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    gender: Optional[str] = None


class DoctorInfo(BaseModel):
    """Additional information for doctors"""
    specialization: Optional[str] = None  # Chuyên khoa
    licenseNumber: Optional[str] = None  # Số chứng chỉ hành nghề
    hospital: Optional[str] = None  # Bệnh viện/Phòng khám
    experience: Optional[int] = None  # Số năm kinh nghiệm
    qualifications: Optional[List[str]] = []  # Bằng cấp
    consultationFee: Optional[float] = None  # Phí tư vấn


class UserPreferences(BaseModel):
    """User preferences"""
    theme: Theme = Theme.LIGHT
    language: Language = Language.VI
    notifications: bool = True


class User(Document):
    """User document model"""
    email: EmailStr
    password: str  # Hashed password
    role: UserRole = UserRole.PATIENT  # Default role is patient
    profile: UserProfile
    doctorInfo: Optional[DoctorInfo] = None  # Chỉ có khi role là doctor
    preferences: UserPreferences = UserPreferences()
    createdAt: datetime = datetime.utcnow()
    updatedAt: datetime = datetime.utcnow()
    isActive: bool = True
    lastLogin: Optional[datetime] = None
    
    class Settings:
        name = "users"
        
    def __repr__(self) -> str:
        return f"<User {self.email}>"
    
    def full_name(self) -> str:
        """Get user's full name"""
        return f"{self.profile.firstName} {self.profile.lastName}"
    
    def is_doctor(self) -> bool:
        """Check if user is a doctor"""
        return self.role == UserRole.DOCTOR
    
    def is_patient(self) -> bool:
        """Check if user is a patient"""
        return self.role == UserRole.PATIENT
    
    async def update_last_login(self):
        """Update last login timestamp"""
        self.lastLogin = datetime.utcnow()
        self.updatedAt = datetime.utcnow()
        await self.save()
    
    async def update_profile(self, profile_data: dict):
        """Update user profile"""
        for key, value in profile_data.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self.updatedAt = datetime.utcnow()
        await self.save()
    
    async def update_doctor_info(self, doctor_data: dict):
        """Update doctor information"""
        if self.role == UserRole.DOCTOR:
            if self.doctorInfo is None:
                self.doctorInfo = DoctorInfo()
            for key, value in doctor_data.items():
                if hasattr(self.doctorInfo, key):
                    setattr(self.doctorInfo, key, value)
            self.updatedAt = datetime.utcnow()
            await self.save()
