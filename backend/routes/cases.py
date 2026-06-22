from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, CaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/cases")
def get_cases(db: Session = Depends(get_db)):
    cases = db.query(CaseModel).order_by(CaseModel.id.desc()).all()
    # Serialize correctly
    return [
        {
            "caseId": c.caseId,
            "patientId": c.patientId,
            "age": c.age,
            "grade": c.grade,
            "kps": c.kps,
            "size": c.size,
            "symptoms": c.symptoms,
            "scanType": c.scanType,
            "timestamp": c.timestamp,
            "prediction": c.prediction,
            "confidence": c.confidence,
            "imagePreview": c.imagePreview,
            "heatmap": c.heatmap,
            "segmentation_mask": c.segmentation_mask,
            "filename": c.filename,
            "insight": c.insight,
            "status": c.status
        }
        for c in cases
    ]

@router.post("/cases")
def save_case(data: dict, db: Session = Depends(get_db)):
    new_case = CaseModel(
        caseId=data.get("caseId"),
        patientId=data.get("patientId"),
        age=str(data.get("age", "")),
        grade=str(data.get("grade", "")),
        kps=int(data.get("kps") or 80),
        size=str(data.get("size", "")),
        symptoms=data.get("symptoms", ""),
        scanType=data.get("scanType", ""),
        timestamp=data.get("timestamp", ""),
        prediction=data.get("prediction", ""),
        confidence=float(data.get("confidence") or 0.0),
        imagePreview=data.get("imagePreview", ""),
        heatmap=data.get("heatmap", ""),
        segmentation_mask=data.get("segmentation_mask", ""),
        filename=data.get("filename", ""),
        insight=data.get("insight", {}),
        status=data.get("status", "")
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)
    return {"status": "success", "caseId": new_case.caseId}
