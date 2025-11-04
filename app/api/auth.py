"""
Authentication endpoints for BreastCare AI Backend
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from datetime import datetime, timedelta
import hashlib
import secrets

from app.models.auth import LoginRequest, RegisterRequest, TokenResponse, RefreshTokenRequest, RefreshToken
from app.models.user import User, UserProfile, UserRole
from app.core.config import settings
from app.utils.security import password_helper, jwt_helper
from app.utils.dependencies import get_current_user

router = APIRouter()
security = HTTPBearer()


@router.post("/register", response_model=TokenResponse)
async def register_user(request: RegisterRequest):
    """Register a new user"""
    # Check if user exists
    existing_user = await User.find_one(User.email == request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate doctor info if role is doctor
    if request.role == UserRole.DOCTOR:
        if not request.doctorInfo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Doctor information is required for doctor registration"
            )
        if not request.doctorInfo.specialization or not request.doctorInfo.licenseNumber:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Specialization and license number are required for doctors"
            )
    
    # Hash password
    hashed_password = password_helper.hash_password(request.password)
    
    # Create user profile
    user_profile = UserProfile(
        firstName=request.firstName,
        lastName=request.lastName
    )
    
    # Create new user
    new_user = User(
        email=request.email,
        password=hashed_password,
        role=request.role,
        profile=user_profile,
        doctorInfo=request.doctorInfo if request.role == UserRole.DOCTOR else None
    )
    
    await new_user.save()
    
    # Generate tokens
    token_data = {
        "sub": str(new_user.id),
        "email": new_user.email,
        "role": new_user.role
    }
    
    access_token = jwt_helper.create_access_token(token_data)
    refresh_token = jwt_helper.create_refresh_token(token_data)
    
    # Store refresh token in database
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    refresh_token_doc = RefreshToken(
        userId=new_user.id,
        refreshToken=refresh_token_hash,
        deviceInfo=request.deviceInfo,
        expiresAt=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    await refresh_token_doc.save()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(request: LoginRequest):
    """Login user"""
    # Find user by email
    user = await User.find_one(User.email == request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not password_helper.verify_password(request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if user is active
    if not user.isActive:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive"
        )
    
    # Update last login
    await user.update_last_login()
    
    # Generate tokens
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role
    }
    
    access_token = jwt_helper.create_access_token(token_data)
    refresh_token = jwt_helper.create_refresh_token(token_data)
    
    # Store refresh token in database
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    refresh_token_doc = RefreshToken(
        userId=user.id,
        refreshToken=refresh_token_hash,
        deviceInfo=request.deviceInfo,
        expiresAt=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    await refresh_token_doc.save()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token"""
    # Verify refresh token
    try:
        payload = jwt_helper.verify_token(request.refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if refresh token exists in database
        refresh_token_hash = hashlib.sha256(request.refresh_token.encode()).hexdigest()
        refresh_token_doc = await RefreshToken.find_one(
            RefreshToken.refreshToken == refresh_token_hash,
            RefreshToken.userId == user_id
        )
        
        if not refresh_token_doc or not refresh_token_doc.is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Get user
        user = await User.get(user_id)
        if not user or not user.isActive:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Generate new access token
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role
        }
        
        access_token = jwt_helper.create_access_token(token_data)
        new_refresh_token = jwt_helper.create_refresh_token(token_data)
        
        # Update refresh token in database
        await refresh_token_doc.deactivate()
        new_refresh_token_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()
        new_refresh_token_doc = RefreshToken(
            userId=user.id,
            refreshToken=new_refresh_token_hash,
            deviceInfo=refresh_token_doc.deviceInfo,
            expiresAt=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        await new_refresh_token_doc.save()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout_user(current_user: User = Depends(get_current_user)):
    """Logout user"""
    # Revoke all refresh tokens for the user
    await RefreshToken.revoke_user_tokens(current_user.id)
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "role": current_user.role,
        "profile": current_user.profile,
        "doctorInfo": current_user.doctorInfo,
        "preferences": current_user.preferences,
        "lastLogin": current_user.lastLogin,
        "isActive": current_user.isActive
    }


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    # Verify old password
    if not password_helper.verify_password(old_password, current_user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid current password"
        )
    
    # Hash new password
    new_hashed_password = password_helper.hash_password(new_password)
    
    # Update password
    current_user.password = new_hashed_password
    current_user.updatedAt = datetime.utcnow()
    await current_user.save()
    
    # Revoke all existing refresh tokens
    await RefreshToken.revoke_user_tokens(current_user.id)
    
    return {"message": "Password changed successfully"}
