import cv2
import numpy as np
import pandas as pd
from pathlib import Path

CLASSES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "circle": 0}
OUT = Path("data/generated")


def draw_shape(img, name, cx, cy, r, angle, color):
    if name == "circle":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        n = CLASSES[name]
        pts = [(cx + r * np.cos(angle + 2 * np.pi * k / n),
                cy + r * np.sin(angle + 2 * np.pi * k / n)) for k in range(n)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def make_background(size, rng, noise=0):
    bg = rng.integers(170, 256, 3).tolist()
    img = np.full((size, size, 3), bg, np.uint8)
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def random_color(rng):
    return rng.integers(0, 120, 3).tolist()


def generate_single(split, per_class, seed, size=128):
    rng = np.random.default_rng(seed)
    rows = []
    for name in CLASSES:
        folder = OUT / "single" / split / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = make_background(size, rng, noise=rng.choice([0, 5, 10]))
            r = int(rng.integers(25, 50))
            cx = int(rng.integers(r + 5, size - r - 5))
            cy = int(rng.integers(r + 5, size - r - 5))
            angle = rng.uniform(0, 2 * np.pi)
            draw_shape(img, name, cx, cy, r, angle, random_color(rng))
            path = folder / f"{name}_{i:04d}.png"
            cv2.imwrite(str(path), img)
            rows.append({"path": str(path), "label": name, "split": split,
                         "width": size, "height": size})
    return rows


def generate_scenes(split, n_images, seed, size=300):
    rng = np.random.default_rng(seed)
    folder = OUT / "scenes" / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_images):
        img = make_background(size, rng, noise=rng.choice([0, 5, 10]))
        placed = []
        target = int(rng.integers(2, 7))
        tries = 0
        while len(placed) < target and tries < 200:
            tries += 1
            r = int(rng.integers(20, 40))
            cx = int(rng.integers(r + 5, size - r - 5))
            cy = int(rng.integers(r + 5, size - r - 5))
            if any(np.hypot(cx - x, cy - y) < r + pr + 8 for x, y, pr, _ in placed):
                continue
            name = str(rng.choice(list(CLASSES)))
            draw_shape(img, name, cx, cy, r, rng.uniform(0, 2 * np.pi), random_color(rng))
            placed.append((cx, cy, r, name))
        path = folder / f"scene_{i:04d}.png"
        cv2.imwrite(str(path), img)
        row = {"path": str(path), "split": split, "total_count": len(placed)}
        for name in CLASSES:
            row[f"n_{name}"] = sum(p[3] == name for p in placed)
        rows.append(row)
    return rows


if __name__ == "__main__":
    single = generate_single("train", 200, seed=1) + generate_single("test", 50, seed=2)
    pd.DataFrame(single).to_csv(OUT / "single_metadata.csv", index=False)

    scenes = generate_scenes("train", 100, seed=3) + generate_scenes("test", 40, seed=4)
    pd.DataFrame(scenes).to_csv(OUT / "scenes_metadata.csv", index=False)
    print("Done:", len(single), "single images,", len(scenes), "scenes")