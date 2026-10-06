import os
import sys
from collections import Counter

import cv2
import joblib

from src.counting import detect_shapes, draw_result

MAX_SIDE = 1000
_bundle = None


def predict_image(path, save=True):
    """Return (annotated BGR image, results, per-class counts)."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load("models/shape_rf.joblib")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    results = detect_shapes(img, _bundle["model"], _bundle["features"], robust=True)
    out = draw_result(img, results)
    if save:
        os.makedirs("outputs/predictions", exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f"outputs/predictions/{name}_result.png", out)
    return out, results, Counter(label for _, label in results)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    if not args:
        print("usage: python predict.py image.png [more.png] [--show]")
        sys.exit()
    for p in args:
        out, results, counts = predict_image(p)
        print(f"{p}: {len(results)} shapes {dict(counts)}")
        if "--show" in sys.argv:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 9))
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()