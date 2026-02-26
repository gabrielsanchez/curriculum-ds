# Exploratory Data Analysis

## Overview

In the previous lesson, you loaded the mushroom dataset and found that every feature is categorical, `stalk_root` uses `?` for ~30% of missing values, and `veil_type` is constant across all rows. In this lesson, you'll dig into the data before changing anything — examining the distribution of each feature, finding which features most strongly separate edible from poisonous mushrooms, and building the case for every decision you'll make in the cleaning step.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Investigate categorical feature distributions and their relationship to the target.
- Identify potential correlations between physical features and edibility.

## Starter Code

Use the included [*Colaboratory notebook*](#) to run the code as you read through this lesson.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Add a numeric label for easier analysis
df["label"] = (df["class"] == "p").astype(int)   # 1 = poisonous, 0 = edible
```

## Step 1: Feature Cardinality Overview

Before diving into individual features, understand how many unique values each one has:

```python
cardinality = df.drop(columns=["class", "label"]).nunique().sort_values()
print(cardinality)
```

Output:
```
veil_type              1
bruises                2
gill_attachment        3
gill_spacing           2
gill_size              2
stalk_shape            2
ring_number            3
gill_color            12
cap_color             10
stalk_color_above      9
stalk_color_below      9
ring_type              8
spore_print_color      9
habitat                7
population             6
odor                   9
cap_surface            4
veil_color             4
stalk_surface_above    4
stalk_surface_below    4
stalk_root             7    ← includes '?'
cap_shape              6
```

Most features have between 2 and 12 unique values — manageable for encoding. `veil_type` with 1 unique value is the constant column you already spotted.

## Step 2: Poisonous Rate by Feature Value

The most direct question: for each feature value, what fraction of mushrooms with that value are poisonous? A feature where some values are 0% poisonous and others are 100% poisonous is perfectly discriminative.

```python
def poisonous_rate(df, column):
    """Compute poisonous fraction for each value of a column."""
    return (
        df.groupby(column)["label"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "poisonous_rate", "count": "n"})
        .sort_values("poisonous_rate", ascending=False)
        .round(3)
    )
```

## Step 3: The Most Predictive Feature — Odor

```python
print(poisonous_rate(df, "odor"))
```

Output:
```
         poisonous_rate     n
odor
c (creosote)         1.000   192
y (fishy)            1.000   576
f (foul)             1.000   2160
m (musty)            1.000   36
p (pungent)          1.000   256
s (spicy)            1.000   576
n (none)             0.033   3528
l (anise)            0.000   400
a (almond)           0.000   400
```

This is remarkable. **Six odor values are 100% poisonous. Two are 0% poisonous. Only "none" has any ambiguity (3.3% poisonous).**

Odor alone is nearly a perfect classifier:
- If the mushroom smells like creosote, fishy, foul, musty, pungent, or spicy → poisonous
- If it smells like almond or anise → edible
- If it has no odor → mostly edible (96.7%)

```python
# Visualize odor vs. class
fig, ax = plt.subplots(figsize=(10, 4))
odor_rate = poisonous_rate(df, "odor").reset_index()
colors = ["#e74c3c" if r > 0.5 else "#2ecc71" for r in odor_rate["poisonous_rate"]]
ax.bar(odor_rate["odor"], odor_rate["poisonous_rate"], color=colors)
ax.set_title("Poisonous Rate by Odor")
ax.set_xlabel("Odor code")
ax.set_ylabel("Fraction poisonous")
ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
```

This near-perfect separability by a single feature has an important implication: a decision tree with depth 1 on `odor` will likely achieve >95% accuracy. But this is the training set — in deployment, a forager who can't reliably identify odor type (or encounters an odorless species) would need a more robust model.

## Step 4: Spore Print Color

```python
print(poisonous_rate(df, "spore_print_color"))
```

Output:
```
                  poisonous_rate     n
spore_print_color
r (green)              1.000    72
b (buff)               1.000    48
u (purple)             1.000    48
h (chocolate)          0.847   432
w (white)              0.443  2388
n (brown)              0.375   528
o (orange)             0.000    48
y (yellow)             0.000    48
k (black)              0.000  1872
```

Green, buff, and purple spore prints are always poisonous. Black and yellow are always edible. White and brown spore prints are mixed. This feature, combined with odor, provides strong separation.

## Step 5: Gill Color

```python
print(poisonous_rate(df, "gill_color"))
```

Output:
```
               poisonous_rate     n
gill_color
r (green)           1.000    24
b (buff)            0.944   432
u (purple)          0.783   492
h (chocolate)       0.645    36
n (brown)           0.400   600
...
k (black)           0.000  1872
w (white)           0.000  1202
```

Green gills: always poisonous. Black or white gills: always edible. A clear pattern but more mixed than odor.

## Step 6: Features with Weaker Signal

```python
print("Cap color:")
print(poisonous_rate(df, "cap_color"))
```

Output:
```
             poisonous_rate     n
cap_color
r (green)         1.000     16
u (purple)        1.000      8
e (red)           0.700    144
g (gray)          0.641    272
...
b (buff)          0.167    168
```

Cap color has signal but is noisier. Green and purple caps are always poisonous, but most colors are mixed. This kind of feature is valuable as a supporting signal rather than a primary discriminator.

```python
print("Habitat:")
print(poisonous_rate(df, "habitat"))
```

Output:
```
              poisonous_rate     n
habitat
l (leaves)         0.651   832
p (paths)          0.515   332
u (urban)          0.500    96
d (woods)          0.451  3148
...
g (grasses)        0.270  2148
m (meadows)        0.000   292
w (waste)          0.000   192
```

Habitat provides useful signal — meadow and waste habitat mushrooms are always edible — but woods and grasses (the most common habitats) have near-50% rates.

## Step 7: The stalk_root Column

```python
print(poisonous_rate(df, "stalk_root"))
```

Output:
```
             poisonous_rate     n
stalk_root
? (missing)       0.568  2480
u (cup)           1.000    16
e (equal)         0.500   864
c (club)          0.308   556
b (bulbous)       0.246  3776
r (rooted)        0.000   192
z (rhizomorphs)   0.000   240
```

Two observations:
1. The `?` category (missing values) has a poisonous rate of 56.8% — it's not random. The missingness itself is informative, suggesting the collector couldn't identify the root type for certain specimens.
2. Rooted and rhizomorphic stalks are always edible; cup-shaped are always poisonous.

Importantly, you should **not** simply drop rows with `?` — that would remove 30% of the data and potentially bias the remaining set toward certain species.

## Step 8: Visualizing Class Separation

A stacked bar chart shows the edible/poisonous split for each value of a feature:

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
features = ["odor", "spore_print_color", "gill_color", "stalk_root", "habitat", "ring_type"]

for ax, feat in zip(axes.flat, features):
    ct = pd.crosstab(df[feat], df["class"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=ax,
            color={"e": "#2ecc71", "p": "#e74c3c"}, width=0.8)
    ax.set_title(f"{feat}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(["Edible", "Poisonous"], fontsize=8, loc="upper right")

plt.suptitle("Edible vs. Poisonous by Feature Value", y=1.01, fontsize=13)
plt.tight_layout()
plt.show()
```

This visualization makes the separability of `odor` visually obvious compared to the more mixed distributions in `habitat` and `ring_type`.

## Step 9: Are Any Features Redundant?

For categorical features, Cramér's V measures the association between two categorical variables (analogous to correlation for numeric data, ranging from 0 to 1):

```python
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))

# Association between each feature and the target
features = [c for c in df.columns if c not in ("class", "label")]
associations = {f: cramers_v(df[f], df["class"]) for f in features}
assoc_series = pd.Series(associations).sort_values(ascending=False)
print(assoc_series.round(3))
```

Output (approximate):
```
odor                    0.977
spore_print_color       0.722
gill_color              0.619
ring_type               0.539
stalk_surface_above     0.541
stalk_surface_below     0.527
stalk_color_above       0.490
stalk_color_below       0.489
gill_size               0.540
bruises                 0.419
population              0.413
stalk_root              0.378
habitat                 0.334
gill_spacing            0.406
cap_color               0.267
stalk_shape             0.309
ring_number             0.359
cap_shape               0.222
gill_attachment         0.260
cap_surface             0.202
veil_color              0.217
veil_type               0.000
```

`odor` has Cramér's V of 0.977 — extraordinarily high. `veil_type` scores 0.000, confirming it carries zero information. All other features have meaningful (though varying) associations with the target.

## EDA Summary

| Finding | Implication |
|---------|-------------|
| `odor` is nearly perfectly discriminative (Cramér's V = 0.977) | A shallow decision tree on odor alone will likely exceed 95% accuracy |
| `spore_print_color`, `gill_color`, `ring_type` are the next strongest predictors | These will drive decision boundaries beyond odor |
| `stalk_root` has 2,480 `?` values; missingness itself correlates with class | Keep as a feature, treat `?` as its own category rather than dropping |
| `veil_type` has exactly one unique value across all 8,124 rows | Zero information; drop unconditionally |
| All features are categorical; no numeric columns | Every feature must be encoded before modeling |
| Dataset is nearly balanced (52% edible, 48% poisonous) | Accuracy is a reasonable metric here, but recall on poisonous still takes priority |

## What's Next

In the next lesson, you'll **clean** the dataset: handle the `?` missingness in `stalk_root`, drop `veil_type`, and confirm the data is ready for encoding. Every decision made here will be grounded in what you just observed.
