import os
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

feats = pd.read_csv("data/processed/features.csv")
FULL = [c for c in feats.columns if c not in ("id", "label", "source", "split")]
REDUCED = ["hu1", "rect_fill", "circle_fill", "n_vertices_4",
           "rect_aspect", "area", "hu3", "solidity", "hu2"]
MINIMAL = ["n_vertices_4", "circle_fill", "rect_fill", "rect_aspect", "hu1"]

tr = feats[feats["split"] == "train"]
ytr = tr["label"]

models = {
    "LogReg": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced")),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "SVM": make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", class_weight="balanced")),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for fs_name, fs in [("full", FULL), ("reduced", REDUCED), ("minimal", MINIMAL)]:
    for name, m in models.items():
        r = cross_validate(m, tr[fs], ytr, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({"features": fs_name, "model": name, "n_features": len(fs),
                     "accuracy": r["test_accuracy"].mean(),
                     "f1_macro": r["test_f1_macro"].mean(),
                     "f1_std": r["test_f1_macro"].std()})

res = pd.DataFrame(rows)
print(res.pivot(index="model", columns="features", values="f1_macro").round(4))
res.to_csv("outputs/cv_results.csv", index=False)