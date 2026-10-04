\# Geometric Shape Detection and Counting



IDC409 project, topic 6: \*\*Object Identification\*\*. Identify and count simple geometric shapes (circle, triangle, square, pentagon, hexagon) in generated images and in Kaggle datasets.



Authors: ARGHA PATRA (MS24159); SUJAY MITRA (MS23058); KAVYAJEET SINGH (MS24223)



\## Overview



The pipeline has two parts:



1\. \*\*Classification.\*\* Each image is reduced to a centred 64x64 binary silhouette. Five geometric features are extracted and classified with a Random Forest.

2\. \*\*Counting.\*\* In multi-shape scenes, every contour is detected, normalised in exactly the same way as the training silhouettes, and classified. Touching shapes are separated with a distance transform and watershed.



\## Data



| Source | Images used | Classes |

|---|---|---|

| Generated (`src/generate\_shapes.py`) | 1,250 single shapes + 140 multi-shape scenes | all 5 |

| \[Kaggle 2D shapes (17 shapes)](https://www.kaggle.com/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes) | 2,000 (400 per class, sampled) | all 5 |

| \[Kaggle simple shapes](https://www.kaggle.com/datasets/dineshpiyasamara/geometric-shapes-dataset) | 900 (300 per class, sampled) | circle, square, triangle |



Total: 4,150 single-shape images with a stratified 80/20 train/test split (3,320 / 830). The Kaggle datasets are not included in the repo; download them into `data/raw/kaggle\_2d\_shapes/` and `data/raw/kaggle\_simple\_shapes/`.



Metadata is stored in a SQLite database (`data/shapes.db`, schema in `src/schema.sql`) and read, written, updated and deleted with SQL from the notebooks. The database file is not committed; notebook 04 rebuilds it.



\## Method



\*\*Preprocessing (`src/preprocess.py`).\*\* The sources differ in size, background colour, and shape size, so grayscale thresholding is unreliable (for example, yellow on white). Instead, the background colour is estimated from the image border, the colour distance of every pixel to it is thresholded with Otsu, and the largest contour is cropped, padded to a square and resized to 64x64.



\*\*Features (`src/features.py`).\*\* Contour features were computed (area, circularity, solidity, fill ratios, vertex counts, Hu moments) and reduced from 16 to 5 using correlation analysis, PCA and Random Forest importances: `n\_vertices\_4`, `circle\_fill`, `rect\_fill`, `rect\_aspect`, `hu1`.



\*\*Models.\*\* Logistic regression, decision tree, KNN, SVM and Random Forest, compared with leave-one-source-out cross-validation (train on two sources, validate on the third).



\*\*Counting (`src/counting.py`).\*\* Shape mask, contours, 64x64 silhouette per contour, features, classifier. Blobs with solidity below 0.92 are split with a distance transform and watershed.



\## Results



\### Classification



| Evaluation | Result |

|---|---|

| Test set (830 images) | accuracy 1.000, macro-F1 1.000 |

| Leave-one-source-out, held-out `generated` | accuracy 1.000 |

| Leave-one-source-out, held-out `kaggle\_2d` | accuracy 1.000 |

| Leave-one-source-out, held-out `kaggle\_simple` | accuracy 0.9989 |



!\[Confusion matrix](outputs/confusion\_matrix.png)



Random Forest on the 5-feature set was selected. Several features separate the classes almost perfectly on their own (`hu1`, `circle\_fill` and `rect\_fill` each score at least 0.99 alone), so this task is easy for these features. 111 test silhouettes are identical to training ones (axis-aligned squares and circles look the same after normalisation), so the random split is slightly optimistic, and the leave-one-source-out numbers are the more reliable ones.



\### Counting on clean scenes



100 train and 40 test scenes (2 to 6 shapes, light noise, shapes separated): exact total count and every per-class count were correct in all scenes (MAE 0.0). The classifier was trained only on single-shape silhouettes.



!\[Counting demo](outputs/counting\_demo.png)



\### Extensions: noise, blur, touching and overlapping shapes



40 scenes per difficulty level. An exact count means the predicted total equals the true total.



| Level | Contours only | With watershed split |

|---|---|---|

| clean | 1.000 | 1.000 |

| noisy (sigma 25) | 1.000 | 1.000 |

| blurry | 1.000 | 1.000 |

| touching | 0.750 | \*\*1.000\*\* |

| overlap | 0.475 | \*\*0.825\*\* |



!\[Before and after watershed](outputs/extension\_before\_after.png)



\*\*Limitations.\*\* Per-class correctness is lower than the exact count on the hardest levels (touching 0.925, overlap 0.650). The classifier was trained on complete shapes, so fragments of partly hidden shapes can be misclassified, and a small shape lying on a large one can stay merged because it has no separate peak in the distance map. Handling heavy occlusion would need a different approach, for example, training on partial shapes.



\## Repository layout



```

src/            generate\_shapes.py, hard\_scenes.py, preprocess.py, features.py, counting.py, schema.sql

notebooks/      01 setup check, 02 data check, 03 load data, 04 database, 05 EDA,

&#x20;               06 features, 07 models, 08 evaluation, 09 counting, 10 extensions

outputs/        figures and result tables

environment.yml conda environment

```



\## Reproducing



```bash

conda env create -f environment.yml

conda activate shapes

python src/generate\_shapes.py        # generated single shapes and scenes

```



Then download the Kaggle data into `data/raw/` and run the notebooks in order (03 to 10). Run notebook 04 only once; it rebuilds the database, and notebook 05 adds the statistics columns.

