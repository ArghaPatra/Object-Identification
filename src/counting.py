import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from src.preprocess import shape_mask, SIZE, PAD
from src.features import contour_features

MIN_AREA = 150
SPLIT_SOLIDITY = 0.92   # blobs less convex than this are candidates for splitting


def mask_to_silhouette(filled, size=SIZE, pad=PAD):
    ys, xs = np.nonzero(filled)
    crop = filled[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = max(h, w)
    square = np.zeros((s, s), np.uint8)
    oy, ox = (s - h) // 2, (s - w) // 2
    square[oy:oy + h, ox:ox + w] = crop
    inner = size - 2 * pad
    interp = cv2.INTER_AREA if s > inner else cv2.INTER_LINEAR
    out = cv2.resize(square, (inner, inner), interpolation=interp)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def split_blob(filled, min_radius=6, min_distance=10):
    """Original splitter (used for the 300 px experiments)."""
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 2)
    peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=min_radius,
                           labels=(filled > 0).astype(int), exclude_border=False)
    if len(peaks) < 2:
        return [filled]
    markers = np.zeros(filled.shape, np.int32)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    lab = watershed(-dist, markers, mask=filled > 0)
    parts = []
    for i in range(1, len(peaks) + 1):
        m = (lab == i).astype(np.uint8) * 255
        if cv2.countNonZero(m) >= MIN_AREA:
            parts.append(m)
    return parts or [filled]


def split_blob_hmax(filled, min_area):
    """Robust splitter for large images: a flat ridge counts as one peak."""
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax < 3:
        return [filled]
    dist = cv2.GaussianBlur(dist, (0, 0), max(1.0, 0.03 * dmax))
    markers, n = ndi.label(h_maxima(dist, max(1.0, 0.15 * dmax)))
    if n < 2:
        return [filled]
    lab = watershed(-dist, markers, mask=filled > 0)
    parts = []
    for i in range(1, n + 1):
        m = (lab == i).astype(np.uint8) * 255
        if cv2.countNonZero(m) >= min_area:
            parts.append(m)
    return parts or [filled]


def detect_shapes(img_bgr, model, feature_names, min_area=MIN_AREA, split=True, robust=False):
    if robust:
        mask = shape_mask(img_bgr, low_contrast=True)
        min_area = max(min_area, int(1.5e-4 * mask.size))
    else:
        mask = shape_mask(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, feats = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        parts = [filled]
        if split and hull_area > 0 and area / hull_area < SPLIT_SOLIDITY:
            parts = split_blob_hmax(filled, min_area) if robust else split_blob(filled)
        for p in parts:
            feats.append(contour_features(mask_to_silhouette(p)))
            boxes.append(cv2.boundingRect(cv2.findNonZero(p)))
    if not boxes:
        return []
    labels = model.predict(pd.DataFrame(feats)[feature_names])
    return list(zip(boxes, labels))


def draw_result(img_bgr, results):
    out = img_bgr.copy()
    s = max(1.0, max(out.shape[:2]) / 500)      # scale text for large images
    t = max(1, int(round(s)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h), label in results:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 160, 0), 2 * t)
        cv2.putText(out, label, (x, max(y - 5, int(14 * s))), font,
                    0.5 * s, (0, 0, 200), t, cv2.LINE_AA)
    cv2.putText(out, f"count: {len(results)}", (8, int(22 * s)), font,
                0.65 * s, (0, 0, 0), 2 * t, cv2.LINE_AA)
    return out