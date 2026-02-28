# Introduction to the Case Study

## Overview

In module 09, you learned how to build binary classifiers, evaluate them with precision, recall, F1, and AUC, and think carefully about what different types of errors cost. Now you'll apply all of that to a real-world dataset that makes the stakes viscerally clear: a mushroom that is predicted "edible" when it is actually poisonous can cause serious harm. This case study walks you through the full classification workflow — from a raw CSV of single-letter codes to a tuned model — and forces you to think about model quality in terms of real-world consequences, not just accuracy scores.

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/10-mushroom/01_mushroom-case-study_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Learning Objectives

By the end of this case study, you will have learned how to:

- Frame a classification problem around domain constraints and error costs.
- Apply the full ML pipeline end-to-end on an entirely categorical dataset.

## What Is This Case Study?

Module 08 applied data preparation to a regression problem (house rent prediction). This module applies the same discipline — EDA, cleaning, feature engineering — to a **classification** problem, and goes one step further: it includes **model selection and evaluation**.

The structure mirrors module 08:

1. **Exploratory Data Analysis** — Understand the data's distributions, feature cardinality, and early signals of what predicts edibility.
2. **Data Cleaning** — Handle missing values encoded as `?`, remove zero-variance features, and validate the dataset.
3. **Feature Engineering** — Encode all categorical features into numbers a classifier can use.
4. **Model Selection** — Train multiple classifiers, compare them, and tune the decision threshold for the safety constraints of this domain.
5. **Summary** — Synthesize findings, interpret the best model, and reflect on the practical implications.

## The Dataset

The **UCI Mushroom Dataset** was contributed by Jeff Schlimmer to the UCI Machine Learning Repository in 1987, based on physical descriptions in *The Audubon Society Field Guide to North American Mushrooms*. It describes hypothetical samples corresponding to 23 species of gilled mushrooms.

**Download:** The dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom).

### Columns

| Column | Description | Example values |
|--------|-------------|----------------|
| `class` | **Target**: edible or poisonous | `e` = edible, `p` = poisonous |
| `cap_shape` | Shape of the cap | `b`=bell, `c`=conical, `x`=convex, `f`=flat, `k`=knobbed, `s`=sunken |
| `cap_surface` | Surface texture of the cap | `f`=fibrous, `g`=grooves, `y`=scaly, `s`=smooth |
| `cap_color` | Color of the cap | `n`=brown, `b`=buff, `c`=cinnamon, `g`=gray, `r`=green, `p`=pink, `u`=purple, `e`=red, `w`=white, `y`=yellow |
| `bruises` | Whether the mushroom bruises | `t`=bruises, `f`=no |
| `odor` | Smell of the mushroom | `a`=almond, `l`=anise, `c`=creosote, `y`=fishy, `f`=foul, `m`=musty, `n`=none, `p`=pungent, `s`=spicy |
| `gill_attachment` | How gills attach to the stalk | `a`=attached, `f`=free, `n`=notched |
| `gill_spacing` | Spacing between gills | `c`=close, `w`=crowded |
| `gill_size` | Width of gills | `b`=broad, `n`=narrow |
| `gill_color` | Color of gills | 12 possible colors |
| `stalk_shape` | Shape of the stalk | `e`=enlarging, `t`=tapering |
| `stalk_root` | Root type of the stalk | `b`=bulbous, `c`=club, `u`=cup, `e`=equal, `z`=rhizomorphs, `r`=rooted, `?`=missing |
| `stalk_surface_above` | Stalk texture above ring | `f`=fibrous, `y`=scaly, `k`=silky, `s`=smooth |
| `stalk_surface_below` | Stalk texture below ring | `f`=fibrous, `y`=scaly, `k`=silky, `s`=smooth |
| `stalk_color_above` | Stalk color above ring | 9 possible colors |
| `stalk_color_below` | Stalk color below ring | 9 possible colors |
| `veil_type` | Type of veil | `p`=partial, `u`=universal |
| `veil_color` | Color of veil | `n`=brown, `o`=orange, `w`=white, `y`=yellow |
| `ring_number` | Number of rings | `n`=none, `o`=one, `t`=two |
| `ring_type` | Type of ring | `c`=cobwebby, `e`=evanescent, `f`=flaring, `l`=large, `n`=none, `p`=pendant, `s`=sheathing, `z`=zone |
| `spore_print_color` | Color of spore print | 9 possible colors |
| `population` | How mushrooms grow | `a`=abundant, `c`=clustered, `n`=numerous, `s`=scattered, `v`=several, `y`=solitary |
| `habitat` | Where mushrooms grow | `g`=grasses, `l`=leaves, `m`=meadows, `p`=paths, `u`=urban, `w`=waste, `d`=woods |

### What Makes This Dataset Unusual

Unlike most datasets, every feature in the mushroom dataset is **categorical**. There are no numeric columns — every observation is encoded as a single letter. This means all features must be encoded before any ML algorithm can use them, making feature engineering a central challenge of this case study.

### The Business Question

> **Can we reliably distinguish edible from poisonous mushrooms from physical features alone — with a high enough recall on poisonous mushrooms to be trusted in a safety-critical application?**

Note the emphasis on *recall on poisonous mushrooms*. A model that achieves 95% accuracy by predicting nearly all mushrooms as edible would be dangerous. You need near-perfect recall on the poisonous class — missing a poisonous mushroom is a far worse outcome than incorrectly labeling an edible one as dangerous.

## Starter Code

Use the included [*Colaboratory notebook*](#) to follow along with the case study. The notebook contains all code blocks from all five lessons.

## Loading the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Column names (the raw file has no header row)
columns = [
    "class", "cap_shape", "cap_surface", "cap_color",
    "bruises", "odor", "gill_attachment", "gill_spacing",
    "gill_size", "gill_color", "stalk_shape", "stalk_root",
    "stalk_surface_above", "stalk_surface_below",
    "stalk_color_above", "stalk_color_below",
    "veil_type", "veil_color", "ring_number", "ring_type",
    "spore_print_color", "population", "habitat"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
df = pd.read_csv(url, header=None, names=columns)

print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
df.head()
```

Output:
```
Shape: (8124, 23)

Data types:
object    23
dtype: int64
```

8,124 mushroom samples across 23 columns — including the target. Every single column is an `object` (string) type, confirming that this is an all-categorical dataset.

## Initial Impressions

```python
# Class distribution
print("Class distribution:")
print(df["class"].value_counts())
print(f"\nPoisonous fraction: {(df['class'] == 'p').mean():.1%}")
```

Output:
```
Class distribution:
e    4208
p    3916

Poisonous fraction: 48.2%
```

The dataset is nearly balanced — 52% edible, 48% poisonous. This is more balanced than most real-world classification problems, which means accuracy will be a more meaningful metric here than usual. Even so, you'll weight recall on the poisonous class above all else.

```python
# Quick scan for missing values (standard NaN)
print(f"\nStandard missing values: {df.isnull().sum().sum()}")

# Check for the '?' encoding used in this dataset
print("\nCount of '?' per column:")
print((df == "?").sum()[df.columns[(df == "?").sum() > 0]])
```

Output:
```
Standard missing values: 0

Count of '?' per column:
stalk_root    2480
```

No standard `NaN` values — but `stalk_root` uses `?` to encode 2,480 missing values (30.5% of rows). This will need to be addressed during cleaning.

```python
# Check for constant columns (zero-variance features)
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique == 1:
        print(f"{col}: only 1 unique value → {df[col].unique()}")
```

Output:
```
veil_type: only 1 unique value → ['p']
```

Every single mushroom has `veil_type = 'p'` (partial). A column with one unique value carries zero information for classification — it will be dropped during cleaning.

## What's Next

The initial scan surfaces three immediate tasks:
1. **Decode the data** — understand what the single-letter codes actually represent (done above)
2. **Handle `stalk_root = '?'`** — 30% missingness needs a principled decision
3. **Drop `veil_type`** — zero-variance feature provides no signal

In the next lesson, you'll dig deeper with EDA before making any changes — exploring which features most strongly separate edible from poisonous mushrooms, and building intuition for the decision boundaries a classifier will need to draw.
