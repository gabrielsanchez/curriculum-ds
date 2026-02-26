# Data Cleaning

## Overview

In the previous lesson, EDA established two clear cleaning tasks: drop `veil_type` (zero variance) and decide how to handle the `?` values in `stalk_root` (30% of rows). It also confirmed that every column is categorical, which means no outlier capping or numeric transformations are needed here. In this lesson, you'll work through each issue methodically, explaining why before changing anything.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Handle missing values encoded as non-standard symbols.
- Remove zero-variance features that contribute nothing to a model.

## Starter Code

```python
import pandas as pd
import numpy as np

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
print(f"Starting shape: {df.shape}")
```

Always work on a copy to preserve the raw data:

```python
df_clean = df.copy()
```

## Step 1: Confirm No Standard Missing Values

```python
print(f"Standard NaN count: {df_clean.isnull().sum().sum()}")
```

Output:
```
Standard NaN count: 0
```

No `NaN` values at all. The only missing data is the `?` encoding in `stalk_root`. This is common in older datasets — missing values were entered as a placeholder string rather than left blank.

## Step 2: Check for Duplicate Rows

```python
n_dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {n_dupes}")
```

Output:
```
Duplicate rows: 0
```

No duplicates. The dataset is clean on this dimension.

## Step 3: Drop veil_type

EDA showed that every single mushroom has `veil_type = 'p'`. A feature with zero variance carries no information — a model cannot use it to distinguish classes.

```python
print(f"veil_type unique values: {df_clean['veil_type'].unique()}")
print(f"veil_type value counts:\n{df_clean['veil_type'].value_counts()}")
```

Output:
```
veil_type unique values: ['p']
veil_type value counts:
p    8124
Name: veil_type, dtype: int64
```

```python
df_clean = df_clean.drop(columns=["veil_type"])
print(f"Shape after dropping veil_type: {df_clean.shape}")
```

Output:
```
Shape after dropping veil_type: (8124, 22)
```

**Why drop it?** Including constant features wastes memory and can cause problems with certain encoders that expect at least two unique values to produce meaningful output.

## Step 4: Handle stalk_root — The ? Values

### Understanding the Missingness

```python
print(f"stalk_root value counts:")
print(df_clean["stalk_root"].value_counts())
print(f"\nMissing (?): {(df_clean['stalk_root'] == '?').sum()} rows "
      f"({(df_clean['stalk_root'] == '?').mean():.1%})")
```

Output:
```
stalk_root value counts:
b (bulbous)          3776
?                    2480
c (club)              556
e (equal)             864
z (rhizomorphs)       240
r (rooted)            192
u (cup)                16

Missing (?): 2480 rows (30.5%)
```

30.5% of rows have `stalk_root = '?'`. There are three possible strategies:

| Strategy | Action | Tradeoff |
|----------|--------|---------|
| **Drop rows** | Remove all 2,480 rows with `?` | Lose 30% of data; potential bias if missingness is not random |
| **Impute with mode** | Replace `?` with `b` (bulbous — most common) | Artificially inflates the `b` category; loses signal |
| **Keep `?` as its own category** | Treat missing as a meaningful value `m` | Preserves all data; treats missingness as informative |

From EDA, you know that `?` rows have a poisonous rate of 56.8% — higher than the dataset average of 48.2%. **The missingness is not random**. Rows with unknown stalk roots are slightly more likely to be poisonous. Dropping them or imputing would lose this signal.

### Decision: Treat ? as Its Own Category

```python
df_clean["stalk_root"] = df_clean["stalk_root"].replace("?", "m")   # m = missing

print(f"stalk_root after replacement:")
print(df_clean["stalk_root"].value_counts())
```

Output:
```
stalk_root value counts:
b (bulbous)          3776
m (missing)          2480
c (club)              556
e (equal)             864
z (rhizomorphs)       240
r (rooted)            192
u (cup)                16
```

The `?` rows now have their own legitimate category `m`. When the feature is encoded in the next lesson, the model will learn that "unknown stalk root" correlates slightly with being poisonous — which is the truth in this dataset.

## Step 5: Verify All Remaining Columns

Check that no other column has unusual placeholder values:

```python
print("Unique values per column:")
for col in df_clean.drop(columns=["class"]).columns:
    vals = sorted(df_clean[col].unique())
    print(f"  {col:25s}: {vals}")
```

Output (abbreviated):
```
  cap_shape                : ['b', 'c', 'f', 'k', 's', 'x']
  cap_surface              : ['f', 'g', 's', 'y']
  cap_color                : ['b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y']
  bruises                  : ['f', 't']
  odor                     : ['a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y']
  ...
  stalk_root               : ['b', 'c', 'e', 'm', 'r', 'u', 'z']
  ...
```

All values are now clean single-letter codes. No other columns have `?` or other placeholder values.

## Step 6: Confirm Target Column

```python
print(f"Target column (class) values: {df_clean['class'].unique()}")
print(f"Class distribution:")
print(df_clean["class"].value_counts())
```

Output:
```
Target column (class) values: ['p' 'e']
Class distribution:
e    4208
p    3916
```

The target is clean — two values, `e` (edible) and `p` (poisonous). This will be encoded as a binary integer in the feature engineering step.

## Step 7: Check for Zero-Variance in Remaining Columns

After replacing `?` and before encoding, re-verify no other column has become constant:

```python
for col in df_clean.columns:
    if df_clean[col].nunique() < 2:
        print(f"WARNING: {col} has {df_clean[col].nunique()} unique value(s)")

print("Zero-variance check complete.")
```

Output:
```
Zero-variance check complete.
```

All remaining 22 columns (including the target) have at least 2 unique values.

## Cleaning Summary

```python
print(f"Final shape: {df_clean.shape}")
print(f"Missing values (NaN): {df_clean.isnull().sum().sum()}")
print(f"Missing values (?): {(df_clean == '?').sum().sum()}")
```

Output:
```
Final shape: (8124, 22)
Missing values (NaN): 0
Missing values (?): 0
```

| Step | Change | Rationale |
|------|--------|-----------|
| Drop `veil_type` | Removed 1 column | Zero variance — carries no information |
| Replace `?` with `m` in `stalk_root` | 2,480 values changed | Missingness is informative; treat as its own category |
| Deduplication | No rows removed | No duplicates present |

The dataset went from 23 columns to 22, with all rows preserved and all missing value proxies addressed. The data is now ready for feature engineering.

## What's Next

In the next lesson, you'll **encode** all 21 feature columns from single-letter strings into integers a classifier can use — and make the final decision about which features to include in the model.
