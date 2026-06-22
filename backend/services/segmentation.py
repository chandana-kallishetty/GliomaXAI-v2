"""
segmentation.py
===============
Patient-specific tumor segmentation service for GliomaXAI.
Uses Otsu thresholding + connected-component analysis to isolate
the brightest/most-likely-tumor region from the actual MRI pixel data.
"""
import numpy as np


def generate_segmentation_mask(image_array: np.ndarray) -> np.ndarray:
    """
    Generate a patient-specific tumor segmentation mask using:
      1. Grayscale conversion
      2. Otsu adaptive thresholding to find hyper-intense regions
      3. Connected-component labelling to isolate the dominant ROI
      4. Morphological dilation to smooth boundaries

    Returns a uint8 mask (255 = tumor, 0 = background) of the same
    spatial resolution as the input image.
    """
    gray = np.mean(image_array, axis=2).astype(np.float32) if image_array.ndim == 3 else image_array.astype(np.float32)

    # Normalise to [0, 255] uint8 for OpenCV
    g_min, g_max = gray.min(), gray.max()
    if g_max > g_min:
        gray_u8 = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
    else:
        gray_u8 = np.zeros_like(gray, dtype=np.uint8)

    try:
        import cv2

        # --- Otsu thresholding -----------------------------------------------
        _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Keep only upper 25% intensity pixels (hyper-intense = likely tumor)
        high_thresh = int(g_min + 0.75 * (g_max - g_min)) if g_max > g_min else 128
        _, bright_mask = cv2.threshold(gray_u8, high_thresh, 255, cv2.THRESH_BINARY)

        # Combine: must pass both Otsu AND high-intensity threshold
        combined = cv2.bitwise_and(thresh, bright_mask)

        # --- Morphological clean-up ------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- Connected components: keep the largest blob ---------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        mask = np.zeros_like(cleaned, dtype=np.uint8)
        if num_labels > 1:
            # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA; label 0 is background
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            blob_area = stats[largest_label, cv2.CC_STAT_AREA]
            total_area = gray.size

            # Only use detected blob if it represents 0.5%–40% of the image
            if 0.005 * total_area <= blob_area <= 0.40 * total_area:
                mask[labels == largest_label] = 255
                # Dilate slightly to capture peri-tumoral edge
                mask = cv2.dilate(mask, kernel, iterations=2)

        # Fallback: if nothing detected, create a small center ellipse
        if np.sum(mask > 0) == 0:
            h, w = mask.shape
            cy, cx = h // 2, w // 2
            # Size proportional to image (roughly 15% diameter)
            ry, rx = max(12, h // 7), max(12, w // 7)
            y_idx, x_idx = np.ogrid[:h, :w]
            ellipse = ((y_idx - cy) / ry) ** 2 + ((x_idx - cx) / rx) ** 2
            mask[ellipse <= 1.0] = 255

        return mask

    except ImportError:
        # Pure-numpy fallback when OpenCV is unavailable
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        ry, rx = max(12, h // 7), max(12, w // 7)
        y_idx, x_idx = np.ogrid[:h, :w]
        ellipse = ((y_idx - cy) / ry) ** 2 + ((x_idx - cx) / rx) ** 2
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ellipse <= 1.0] = 255
        return mask


def apply_segmentation_overlay(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a semi-transparent red overlay to the original MRI on
    the segmented tumor region, and a green contour border.
    """
    overlay = image_array.copy().astype(np.float32)

    # Red fill inside mask (blended)
    tumor_region = mask == 255
    if tumor_region.any():
        overlay[tumor_region, 0] = np.clip(overlay[tumor_region, 0] * 0.4 + 255 * 0.6, 0, 255)
        overlay[tumor_region, 1] = np.clip(overlay[tumor_region, 1] * 0.4, 0, 255)
        overlay[tumor_region, 2] = np.clip(overlay[tumor_region, 2] * 0.4, 0, 255)

    # Green contour boundary
    try:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    except ImportError:
        pass

    return overlay.astype(np.uint8)
