import cv2
import numpy as np
import pandas as pd
from src.preprocess import shape_mask, SIZE, PAD
from src.features import contour_features

MIN_AREA = 150


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


def detect_shapes(img_bgr, model, feature_names, min_area=MIN_AREA):
    mask = shape_mask(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, feats = [], []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        feats.append(contour_features(mask_to_silhouette(filled)))
        boxes.append(cv2.boundingRect(c))
    if not boxes:
        return []
    labels = model.predict(pd.DataFrame(feats)[feature_names])
    return list(zip(boxes, labels))


def draw_result(img_bgr, results):
    out = img_bgr.copy()
    for (x, y, w, h), label in results:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 160, 0), 2)
        cv2.putText(out, label, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"count: {len(results)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out