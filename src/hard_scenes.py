import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.generate_shapes import draw_shape, random_color, CLASSES

OUT = Path("data/generated/hard")

LEVELS = {
    "clean":    dict(noise=5,  blur=0, gap=8),
    "noisy":    dict(noise=25, blur=0, gap=8),
    "blurry":   dict(noise=10, blur=7, gap=8),
    "touching": dict(noise=5,  blur=0, gap=-10),
    "overlap":  dict(noise=5,  blur=0, gap=-25),
}


def make_scene(rng, size=300, noise=5, blur=0, gap=8):
    bg = rng.integers(170, 256, 3).tolist()
    img = np.full((size, size, 3), bg, np.uint8)
    placed, tries = [], 0
    target = int(rng.integers(2, 7))
    while len(placed) < target and tries < 300:
        tries += 1
        r = int(rng.integers(20, 40))
        cx = int(rng.integers(r + 5, size - r - 5))
        cy = int(rng.integers(r + 5, size - r - 5))
        if any(np.hypot(cx - x, cy - y) < r + pr + gap for x, y, pr, _ in placed):
            continue
        name = str(rng.choice(list(CLASSES)))
        draw_shape(img, name, cx, cy, r, rng.uniform(0, 2 * np.pi), random_color(rng))
        placed.append((cx, cy, r, name))
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img, placed


def generate_hard(n_per_level=40):
    rows = []
    for li, (lvl, cfg) in enumerate(LEVELS.items()):
        rng = np.random.default_rng(100 + li)
        folder = OUT / lvl
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_level):
            img, placed = make_scene(rng, **cfg)
            path = folder / f"{lvl}_{i:03d}.png"
            cv2.imwrite(str(path), img)
            row = {"path": str(path), "level": lvl, "total_count": len(placed)}
            for c in CLASSES:
                row[f"n_{c}"] = sum(p[3] == c for p in placed)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metadata.csv", index=False)
    return df