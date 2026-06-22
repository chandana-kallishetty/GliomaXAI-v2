import hashlib
import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from database import SessionLocal, UserModel

router = APIRouter()

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password Hashing Helpers
def hash_password(password: str) -> str:
    """Generate a secure PBKDF2 SHA-256 salted hash."""
    salt = os.urandom(16).hex()
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    iterations = 100000
    key = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, iterations)
    return f"{salt}:{key.hex()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored salt and hash."""
    try:
        salt, stored_hex = hashed_password.split(":")
        pwd_bytes = plain_password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        iterations = 100000
        key = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, iterations)
        return key.hex() == stored_hex
    except Exception:
        return False

# Pydantic Schemas
class RegisterSchema(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
def register_user(data: RegisterSchema, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(UserModel).filter(UserModel.email == data.email.lower()).first()
    if existing_user:
        return {
            "status": "success",
            "message": "User registered successfully (Permissive Access - Email Exists)",
            "user": {
                "email": existing_user.email,
                "fullName": existing_user.fullName,
                "role": existing_user.role,
                "dept": existing_user.dept
            }
        }

    # Infer department from role
    short_role = data.role.split(" - ")[0]
    inferred_dept = "Clinical Research" if "Research" in short_role else "Oncology"

    new_user = UserModel(
        email=data.email.lower(),
        password_hash=hash_password(data.password),
        fullName=data.name.strip(),
        role=short_role,
        dept=inferred_dept
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "status": "success",
        "message": "User registered successfully",
        "user": {
            "email": new_user.email,
            "fullName": new_user.fullName,
            "role": new_user.role,
            "dept": new_user.dept
        }
    }

@router.post("/login")
def login_user(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == data.email.lower()).first()
    
    # "Anyaccess let it in": If user doesn't exist, register them automatically!
    if not user:
        name_prefix = data.email.split("@")[0].title()
        if not name_prefix.lower().startswith("dr."):
            name_prefix = f"Dr. {name_prefix}"
            
        user = UserModel(
            email=data.email.lower(),
            password_hash=hash_password(data.password),
            fullName=name_prefix,
            role="Oncology",
            dept="Oncology"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "status": "success",
        "message": "Authentication successful (Permissive Access)",
        "user": {
            "email": user.email,
            "fullName": user.fullName,
            "role": user.role,
            "dept": user.dept
        }
    }
