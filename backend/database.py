from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Default to PostgreSQL, but allow override
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gliomaxai")

# Fallback to SQLite if psycopg2 fails (useful for local testing)
try:
    engine = create_engine(DATABASE_URL)
    engine.connect()
except Exception as e:
    print(f"[database] PostgreSQL connection failed: {e}. Falling back to SQLite.")
    DATABASE_URL = "sqlite:///./cases.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CaseModel(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    caseId = Column(String, unique=True, index=True)
    patientId = Column(String, index=True)
    age = Column(String)
    grade = Column(String)
    kps = Column(Integer)
    size = Column(String)
    symptoms = Column(String)
    scanType = Column(String)
    timestamp = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    imagePreview = Column(String) # could be large base64 or URL
    heatmap = Column(String) # base64
    segmentation_mask = Column(String) # base64
    filename = Column(String)
    insight = Column(JSON)
    status = Column(String)

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    fullName = Column(String)
    role = Column(String)
    dept = Column(String)

Base.metadata.create_all(bind=engine)
