"""
analytics.py  –  /api/analytics  and  /api/latest-result  endpoints
=====================================================================
"""
from fastapi import APIRouter
from services.inference import get_latest_result

router = APIRouter()


@router.get("/analytics")
def get_analytics():
    """
    Static aggregate analytics used by the dashboard charts / snapshot cards.
    In a production system these would be computed from a real database.
    """
    return {
        "total_cases":      42,
        "high_risk":        12,
        "moderate_risk":    15,
        "low_risk":         15,
        "avg_survival":     22,
        "who_grade":        "III",
        "recent_activity":  "BraTS Multi-modal Processed",
        "segmentation_success_rate": "98%",
        "riskData": [
            {"name": "Low",      "value": 15},
            {"name": "Moderate", "value": 15},
            {"name": "High",     "value": 12},
        ],
        "tumorDistribution": [
            {"name": "Glioma", "count": 18},
            {"name": "Meningioma", "count": 12},
            {"name": "Pituitary", "count": 7},
            {"name": "No Tumor", "count": 5}
        ],
        "modalityUsage": [
            {"modality": "T1", "usage": 100},
            {"modality": "T1CE", "usage": 85},
            {"modality": "T2", "usage": 92},
            {"modality": "FLAIR", "usage": 78}
        ]
    }


@router.get("/latest-result")
def get_latest():
    """
    Returns the most recent MRI inference result stored in memory.
    Returns null payload when no scan has been processed yet in this session.
    """
    result = get_latest_result()
    if result is None:
        return {"status": "no_scan", "result": None}
    return {"status": "ok", "result": result}
