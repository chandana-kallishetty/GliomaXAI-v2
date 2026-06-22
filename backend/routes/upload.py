"""
upload.py  –  /upload endpoint
================================
Accepts brain MRI files (standard images OR NIfTI .nii / .nii.gz),
preprocesses them, runs AI inference, and returns a rich clinical result.

Pipeline:
  1. MIME / extension validation
  2. Read raw bytes
  3. preprocessing.preprocess_mri()  - NIfTI or image -> (224, 224, 3) float32
  4. inference.run_inference()       – classify + derive risk/survival
  5. Return JSON result
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.preprocessing import preprocess_mri, is_nifti, IMAGE_MIME_TYPES
from services.inference import run_inference

router = APIRouter()

# All MIME types we are willing to accept (images + octet-stream for .nii)
ALLOWED_MIME_TYPES = IMAGE_MIME_TYPES | {
    "application/octet-stream",   # common for .nii files
    "application/gzip",           # .nii.gz
    "application/x-gzip",         # alternate MIME for gzip
    "application/x-nifti",        # some clients send this
    "",                            # fallback when browser sends no content-type
}


@router.post("/upload")
async def upload_mri(file: UploadFile = File(...)):
    """
    Accepts a brain MRI image (JPEG/PNG/BMP/TIFF/WebP)
    OR a NIfTI volume (.nii / .nii.gz).

    Returns
    -------
    JSON with:
      prediction, confidence, risk_level, survival_estimate, heatmap, filename
    """
    filename = file.filename or ""
    content_type = (file.content_type or "").lower()

    # -- 1. Validate: must be an image MIME or a NIfTI filename ---------------
    mime_ok = content_type in ALLOWED_MIME_TYPES
    nifti_file = is_nifti(filename)
    image_file = content_type in IMAGE_MIME_TYPES

    if not (mime_ok or nifti_file or image_file):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Please upload a JPEG, PNG, BMP, TIFF, WebP image "
                "or a NIfTI file (.nii / .nii.gz)."
            ),
        )

    import time
    start_time = time.time()

    try:
        # -- 2. Read raw bytes -------------------------------------------------
        raw_bytes = await file.read()
        await file.close()

        # -- 3. Preprocess (handles NIfTI & standard images) ------------------
        try:
            image_array = preprocess_mri(raw_bytes, filename)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # -- 4. Run inference --------------------------------------------------
        result = run_inference(image_array, filename)
        
        # -- 5. Add Phase 1 fields ---------------------------------------------
        processing_time = round(time.time() - start_time, 2)
        result["processing_time"] = f"{processing_time}s"
        result["model_version"] = "v1.0.0-TensorFlow"
        result["heatmap_placeholder"] = not bool(result.get("heatmap"))

        print(
            f"[upload] SUCCESS  file={filename}  "
            f"prediction={result['prediction']}  "
            f"confidence={result['confidence']}%  "
            f"risk={result['risk_level']}  "
            f"time={result['processing_time']}"
        )

        return result

    except HTTPException:
        raise   # re-raise FastAPI exceptions unchanged

    except RuntimeError as exc:
        print(f"[upload] MODEL ERROR  {exc}")
        raise HTTPException(
            status_code=503,
            detail=(
                "ML model is not available. "
                "Please ensure the model file exists and restart the server."
            ),
        )

    except Exception as exc:
        print(f"[upload] ERROR  {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(exc)}",
        )