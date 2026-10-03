import pandas as pd

single = pd.read_csv("data/generated/single_metadata.csv")
scenes = pd.read_csv("data/generated/scenes_metadata.csv")

print(single.groupby(["split", "label"]).size())
print(scenes.groupby("split").size())
print(len(single), len(scenes))

import cv2, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, (label, g) in zip(axes, single[single.split == "train"].groupby("label")):
    img = cv2.imread(g.iloc[0].path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(label)
    ax.axis("off")
plt.show()

# one multi-shape scene with its true count
row = scenes.iloc[0]
img = cv2.imread(row.path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"True count: {row.total_count}")
plt.axis("off")
plt.show()
