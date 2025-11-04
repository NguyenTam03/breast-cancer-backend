"""
Dependencies for BreastCare AI Backend
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from beanie import PydanticObjectId

from app.models.user import User, UserRole
from app.utils.security import jwt_helper

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    
    # Verify token and get user ID
    user_id = jwt_helper.get_user_id_from_token(token)
    
    # Get user from database
    user = await User.get(PydanticObjectId(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.isActive:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    return current_user


async def get_current_doctor(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify they are a doctor"""
    if current_user.role != UserRole.DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor access required"
        )
    return current_user


async def get_current_patient(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify they are a patient"""
    if current_user.role != UserRole.PATIENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Patient access required"
        )
    return current_user


def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_id = jwt_helper.get_user_id_from_token(token)
        # In a real app, you'd want to async fetch the user here
        # For now, we'll return None for optional authentication
        return None
    except:
        return None
