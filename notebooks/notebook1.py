import pandas as pd, cv2, matplotlib.pyplot as plt

df = pd.read_csv("data/generated/single_metadata.csv")
print(df.groupby(["split", "label"]).size())

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, (label, g) in zip(axes, df[df.split == "train"].groupby("label")):
    img = cv2.imread(g.iloc[0].path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(label); ax.axis("off")
plt.show()