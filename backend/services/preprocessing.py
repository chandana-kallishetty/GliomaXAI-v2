"""
preprocessing.py
================
Modular MRI preprocessing service for GliomaXAI.

Performance Optimizations (v2)
------------------------------
- N4 Bias Field Correction is applied ONLY to the primary/middle slice of
  3D volumes. Plain 2D images and non-primary slices skip N4 entirely.
- N4 iterations reduced from [5,5,5] to [3,3] for real-time inference.
- Fast histogram-equalization fallback when SimpleITK is unavailable.
- Skull stripping and intensity standardization run on all slices but are
  lightweight pure-numpy/OpenCV operations.
"""

import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# -- Optional NIfTI support ----------------------------------------------------
try:
    import nibabel as nib
    _NIBABEL_AVAILABLE = True
except ImportError:
    _NIBABEL_AVAILABLE = False
    print("[preprocessing] WARNING: nibabel not installed - NIfTI support disabled.")

# -- Optional DICOM support ----------------------------------------------------
try:
    import pydicom
    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False
    print("[preprocessing] WARNING: pydicom not installed - DICOM support disabled.")

TARGET_SIZE = (224, 224)

IMAGE_MIME_TYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
}

NIFTI_EXTENSIONS = {".nii", ".gz"}


def is_nifti(filename: str) -> bool:
    p = Path(filename.lower())
    return p.suffix == ".nii" or (
        len(p.suffixes) >= 2 and "".join(p.suffixes[-2:]) == ".nii.gz"
    )


def is_dicom(filename: str) -> bool:
    return Path(filename.lower()).suffix == ".dcm"


def _load_nifti_volume(raw_bytes: bytes) -> list[np.ndarray]:
    """
    Load a NIfTI volume from raw bytes, extract up to 8 axial slices,
    normalise to uint8 [0, 255], and return as list of (H, W, 3) float32 arrays.
    """
    import tempfile, os

    if not _NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel is not installed. Install it with: pip install nibabel")

    is_gz = raw_bytes[:2] == b"\x1f\x8b"
    suffix = ".nii.gz" if is_gz else ".nii"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        vol = img.get_fdata(dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse NIfTI data: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if vol.ndim == 4:
        vol = vol[..., 0]

    slices_to_extract = []
    if vol.ndim == 2:
        slices_to_extract.append(vol)
    else:
        depth = vol.shape[2]
        num_slices = min(8, depth)
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        for idx in indices:
            slices_to_extract.append(vol[:, :, idx])

    results = []
    for slice_2d in slices_to_extract:
        vmin, vmax = slice_2d.min(), slice_2d.max()
        if vmax > vmin:
            slice_2d = (slice_2d - vmin) / (vmax - vmin) * 255.0
        else:
            slice_2d = np.zeros_like(slice_2d)

        slice_uint8 = slice_2d.astype(np.uint8)
        pil_img = Image.fromarray(slice_uint8, mode="L").convert("RGB")
        results.append(np.array(pil_img, dtype=np.float32))

    return results


def _load_image_bytes(raw_bytes: bytes) -> list[np.ndarray]:
    """
    Decode a standard image (JPEG/PNG/…) from raw bytes and return
    as list of one (H, W, 3) float32 array.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return [np.array(img, dtype=np.float32)]
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Could not decode the file as an image. "
            "Please upload a valid MRI image (JPEG, PNG, BMP, TIFF, WebP)."
        ) from exc


def _load_dicom_bytes(raw_bytes: bytes) -> list[np.ndarray]:
    """
    Load a DICOM file from raw bytes, normalise to uint8 [0, 255],
    and return as list of (H, W, 3) float32 arrays.
    """
    if not _PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed. Install it with: pip install pydicom")

    try:
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))
        vol = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DICOM data: {exc}") from exc

    slices_to_extract = []
    if vol.ndim >= 3:
        depth = vol.shape[0]
        num_slices = min(8, depth)
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        for idx in indices:
            slices_to_extract.append(vol[idx])
    else:
        slices_to_extract.append(vol)

    results = []
    for slice_2d in slices_to_extract:
        vmin, vmax = slice_2d.min(), slice_2d.max()
        if vmax > vmin:
            slice_2d = (slice_2d - vmin) / (vmax - vmin) * 255.0
        else:
            slice_2d = np.zeros_like(slice_2d)

        slice_uint8 = slice_2d.astype(np.uint8)
        pil_img = Image.fromarray(slice_uint8, mode="L").convert("RGB")
        results.append(np.array(pil_img, dtype=np.float32))

    return results


def apply_n4_bias_correction(image_array: np.ndarray) -> np.ndarray:
    """
    Apply N4 Bias Field Correction (optimised for speed).
    Uses reduced iterations [3,3] and falls back to histogram
    equalisation if SimpleITK is unavailable or fails.
    """
    try:
        import SimpleITK as sitk
        gray = np.mean(image_array, axis=2).astype(np.float32) if image_array.ndim == 3 else image_array.astype(np.float32)
        sitk_img = sitk.GetImageFromArray(gray)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # Reduced iterations for real-time inference (was [5,5,5])
        corrector.SetMaximumNumberOfIterations([3, 3])
        corrected_img = corrector.Execute(sitk_img)
        corrected_arr = sitk.GetArrayFromImage(corrected_img)

        if image_array.ndim == 3:
            res = np.stack([corrected_arr] * 3, axis=-1)
            min_orig, max_orig = image_array.min(), image_array.max()
            min_new, max_new = res.min(), res.max()
            if max_new > min_new:
                res = (res - min_new) / (max_new - min_new) * (max_orig - min_orig) + min_orig
            return res.astype(np.float32)
        return corrected_arr.astype(np.float32)

    except Exception as e:
        print(f"[preprocessing] N4 Bias correction skipped, using CLAHE fallback: {e}")
        # Fast fallback: CLAHE histogram equalisation via OpenCV
        try:
            import cv2
            gray = np.mean(image_array, axis=2).astype(np.uint8) if image_array.ndim == 3 else image_array.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            if image_array.ndim == 3:
                return np.stack([eq.astype(np.float32)] * 3, axis=-1)
            return eq.astype(np.float32)
        except Exception:
            return image_array


def apply_skull_stripping(image_array: np.ndarray) -> np.ndarray:
    """
    Isolate brain tissue using Otsu thresholding + morphological closure.
    """
    try:
        import cv2
        gray = np.mean(image_array, axis=2) if image_array.ndim == 3 else image_array
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

        _, thresh = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        mask = dilated > 0
        stripped = np.copy(image_array)
        if stripped.ndim == 3:
            for c in range(3):
                stripped[:, :, c] = stripped[:, :, c] * mask
        else:
            stripped = stripped * mask
        return stripped
    except Exception as e:
        print(f"[preprocessing] Skull stripping skipped: {e}")
        return image_array


def standardize_intensity(image_array: np.ndarray) -> np.ndarray:
    """
    Standardise image intensities to Z-score, then re-scale to [0, 255].
    """
    try:
        mean_val = np.mean(image_array)
        std_val = np.std(image_array)
        if std_val > 1e-4:
            res = (image_array - mean_val) / std_val
            res_min, res_max = res.min(), res.max()
            if res_max > res_min:
                res = (res - res_min) / (res_max - res_min) * 255.0
            return res
        return image_array
    except Exception as e:
        print(f"[preprocessing] Intensity standardisation skipped: {e}")
        return image_array


def preprocess_mri(raw_bytes: bytes, filename: str) -> list[np.ndarray]:
    """
    Load raw bytes, apply preprocessing pipeline, resize to TARGET_SIZE,
    and return list of float32 arrays.

    Optimisation: N4 bias correction is applied ONLY to the primary
    (middle/first) slice of multi-slice volumes to save CPU time.
    2D images (JPEG/PNG) skip N4 entirely.

    Parameters
    ----------
    raw_bytes : bytes
    filename : str

    Returns
    -------
    list[np.ndarray]
        List of float32 arrays of shape (224, 224, 3) with values in [0, 255].
    """
    is_3d_volume = is_nifti(filename) or is_dicom(filename)

    if is_nifti(filename):
        arrays = _load_nifti_volume(raw_bytes)
    elif is_dicom(filename):
        arrays = _load_dicom_bytes(raw_bytes)
    else:
        arrays = _load_image_bytes(raw_bytes)

    # Find primary (middle) slice index for N4 application
    primary_idx = len(arrays) // 2

    final_arrays = []
    for i, arr in enumerate(arrays):
        # Resize first to reduce computational load for subsequent operations
        pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
            TARGET_SIZE, Image.LANCZOS
        )
        arr_resized = np.array(pil_img, dtype=np.float32)

        # N4 bias correction: only on primary slice of 3D volumes
        if is_3d_volume and i == primary_idx:
            arr_resized = apply_n4_bias_correction(arr_resized)

        # Skull stripping + intensity normalization on all slices (fast ops)
        arr_resized = apply_skull_stripping(arr_resized)
        arr_resized = standardize_intensity(arr_resized)

        final_arrays.append(arr_resized)

    return final_arrays
