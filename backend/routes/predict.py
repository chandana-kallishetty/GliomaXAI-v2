"""
predict.py
==========
API routes for MRI prediction with real-time progress streaming.
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import asyncio
import json
import uuid
import time
from typing import List
import zipfile
import io

from services.model_service import predict_mri
from services.preprocessing import preprocess_mri
from services.inference import run_inference, get_progress, MODEL_PROFILES, _radiomics_cache, _segmentation_cache

router = APIRouter()

_current_patient_id = None


@router.get("/predict")
def predict_demo():
    """Simple demo endpoint to test the model pipeline."""
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    prediction, confidence, _ = predict_mri(dummy_image)
    return {
        "status": "success",
        "test_prediction": prediction,
        "test_confidence": confidence,
        "message": "Model pipeline is functional"
    }


@router.get("/api/model-profiles")
def get_model_profiles():
    """Return model profile metadata (strengths, weaknesses, accuracy, AUC)."""
    return {"models": MODEL_PROFILES}


@router.get("/api/predict/progress/{session_id}")
async def stream_progress(session_id: str):
    """
    Server-Sent Events endpoint: stream real-time prediction progress.
    Frontend subscribes to this before submitting the predict POST request.
    """
    async def event_generator():
        prev_pct = -1
        max_wait = 300  # 5-minute timeout
        start = time.time()

        while time.time() - start < max_wait:
            progress = get_progress(session_id)
            pct = progress.get("pct", 0)
            stage = progress.get("stage", "idle")
            done = progress.get("done", False)

            if pct != prev_pct:
                data = json.dumps({
                    "session_id": session_id,
                    "stage": stage,
                    "pct": pct,
                    "done": done
                })
                yield f"data: {data}\n\n"
                prev_pct = pct

            if done:
                break

            await asyncio.sleep(0.25)

        yield f"data: {json.dumps({'done': True, 'pct': 100})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/predict")
async def predict_real(
    patientId: str = Form(...),
    age: int = Form(...),
    grade: str = Form(...),
    notes: str = Form(""),
    kps: int = Form(80),
    size: float = Form(2.5),
    algorithm: str = Form("XGBoost"),
    files: List[UploadFile] = File(...)
):
    global _current_patient_id

    try:
        import os

        # Generate a unique session ID for this request
        session_id = str(uuid.uuid4())[:8]

        # Clear caches if patient changes to prevent stale result leakage
        if _current_patient_id != patientId:
            _radiomics_cache.clear()
            _segmentation_cache.clear()
            _current_patient_id = patientId

        # Create session folder on disk
        timestamp_str = str(int(time.time()))
        session_dir = os.path.join(os.getcwd(), "uploads", f"{patientId}_{timestamp_str}")
        os.makedirs(session_dir, exist_ok=True)

        results = []

        for file in files:
            raw_bytes = await file.read()
            filename = file.filename or ""

            # Save to disk
            file_path = os.path.join(session_dir, filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(raw_bytes)
            except Exception as e:
                print(f"[predict] Failed to save {filename}: {e}")

            # ZIP archive handling
            if filename.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                        for zinfo in z.infolist():
                            if zinfo.is_dir():
                                continue
                            zname = zinfo.filename.lower()
                            if not any(zname.endswith(ext) for ext in [
                                ".gz", ".nii", ".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
                            ]):
                                continue
                            try:
                                z_bytes = z.read(zinfo.filename)
                                image_arrays = preprocess_mri(z_bytes, zinfo.filename)
                                for img_arr in image_arrays:
                                    res = run_inference(
                                        img_arr, zinfo.filename,
                                        age=age, grade=grade, kps=kps,
                                        size=size, notes=notes,
                                        algorithm=algorithm,
                                        session_id=session_id
                                    )
                                    results.append(res)
                            except Exception as e:
                                print(f"[predict] Error processing zip entry {zinfo.filename}: {e}")
                except Exception as e:
                    print(f"[predict] Error extracting zip {filename}: {e}")
                continue

            # Direct file handling
            if not any(filename.lower().endswith(ext) for ext in [
                ".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
            ]):
                continue

            try:
                image_arrays = preprocess_mri(raw_bytes, filename)
                for img_arr in image_arrays:
                    res = run_inference(
                        img_arr, filename,
                        age=age, grade=grade, kps=kps,
                        size=size, notes=notes,
                        algorithm=algorithm,
                        session_id=session_id
                    )
                    results.append(res)
            except Exception as e:
                print(f"[predict] Error processing {filename}: {e}")

        if not results:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Non-compliant clinical sequence format. "
                    "Diagnostics must be initialized using NIfTI (.nii, .nii.gz), "
                    "DICOM (.dcm), or standard MRI images."
                )
            )

        # Select the most clinically significant result
        severity_rank = {"glioma": 4, "meningioma": 3, "pituitary tumor": 2, "no tumor": 1, "unknown": 0}
        primary_result = results[0]
        for r in results:
            r_rank = severity_rank.get(r.get("prediction", "").lower(), 0)
            p_rank = severity_rank.get(primary_result.get("prediction", "").lower(), 0)
            if r_rank > p_rank or (r_rank == p_rank and r.get("confidence", 0) > primary_result.get("confidence", 0)):
                primary_result = r

        prediction    = primary_result.get("prediction", "Unknown")
        confidence    = primary_result.get("confidence", 0)
        risk_level    = primary_result.get("risk_level", "MEDIUM")
        survival      = primary_result.get("survival_estimate", 0)
        heatmap       = primary_result.get("heatmap", "")
        segmentation  = primary_result.get("segmentation_mask", "")
        image_preview = primary_result.get("image_preview", "")

        summary = (
            f"AI completed multi-sequence scan study of {len(files)} sequence(s). "
            f"Primary mass classification: {prediction} ({confidence}% confidence)."
        )

        return {
            "session_id":        session_id,
            "prediction":        prediction,
            "confidence":        confidence,
            "riskLevel":         risk_level,
            "survivalEstimate":  survival,
            "summary":           summary,
            "heatmapUrl":        heatmap,
            "segmentationUrl":   segmentation,
            "imagePreview":      f"data:image/jpeg;base64,{image_preview}" if image_preview else "",
            "sequences":         results,
            "feature_importance": primary_result.get("feature_importance", []),
            "shap_values":       primary_result.get("shap_values", {}),
            "base_value":        primary_result.get("base_value", 45.0),
            "waterfall":         primary_result.get("waterfall", []),
            "summary_plot":      primary_result.get("summary_plot", ""),
            "algorithm":         primary_result.get("algorithm", algorithm),
            "model_comparison":  primary_result.get("model_comparison", {}),
            "radiomic_features": primary_result.get("radiomic_features", {}),
            "prediction_trace":  primary_result.get("prediction_trace", {}),
            "patient_fingerprint": primary_result.get("patient_fingerprint", {}),
            "elapsed_seconds":   primary_result.get("elapsed_seconds", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))