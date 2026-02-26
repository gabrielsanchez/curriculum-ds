# Feature Engineering

## Overview

The mushroom dataset is now clean: `veil_type` is removed and `stalk_root` missing values are handled. But every remaining feature is still a string — a single letter code that no classifier can use arithmetically. In this lesson, you'll encode all 21 feature columns into integers, make the final train/test split, and assemble a processed feature matrix ready for modeling.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Apply label encoding systematically to an all-categorical dataset.
- Make the train/test split before any fit-based transformations.

## Starter Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and clean (or continue from previous lesson's df_clean)
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
df_clean = df.drop(columns=["veil_type"]).copy()
df_clean["stalk_root"] = df_clean["stalk_root"].replace("?", "m")

print(f"Starting shape: {df_clean.shape}")
```

## Step 1: Encode the Target

Convert the target from `'e'`/`'p'` strings to 0/1 integers. Define poisonous as the positive class (1), since that is the outcome you most want to detect:

```python
y = (df_clean["class"] == "p").astype(int)   # 1 = poisonous, 0 = edible
X = df_clean.drop(columns=["class"])

print(f"Target distribution:")
print(y.value_counts())
print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
```

Output:
```
Target distribution:
0    4208
1    3916

Feature matrix shape: (8124, 21)
Feature columns (21): ['cap_shape', 'cap_surface', 'cap_color', 'bruises',
                        'odor', 'gill_attachment', 'gill_spacing', 'gill_size',
                        'gill_color', 'stalk_shape', 'stalk_root',
                        'stalk_surface_above', 'stalk_surface_below',
                        'stalk_color_above', 'stalk_color_below',
                        'veil_color', 'ring_number', 'ring_type',
                        'spore_print_color', 'population', 'habitat']
```

## Step 2: Train/Test Split First

**Always split before encoding** when using fit-based transformers. Although `LabelEncoder` on a clean categorical column is less prone to leakage than, say, `StandardScaler` (it doesn't compute statistics from the data), the principle is universal: fit transformers on training data, apply to test data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Poisonous rate — train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
```

Output:
```
X_train: (6499, 21), X_test: (1625, 21)
Poisonous rate — train: 48.2%, test: 48.2%
```

`stratify=y` ensures both splits have the same ~48% poisonous rate.

## Step 3: Encoding Strategy

For this dataset, you have two encoding options:

| Option | How it works | Implication |
|--------|-------------|-------------|
| **Label encoding** | Each unique category gets an integer (e.g., `a`=0, `b`=1, `c`=2) | Implies a numeric ordering that doesn't exist; but works well with tree-based models, which split on values rather than treating them as continuous |
| **One-hot encoding** | Each unique value becomes a binary column | No false ordering; better for logistic regression; produces many columns for high-cardinality features |

Given that this dataset has 21 columns, some with up to 12 unique values, one-hot encoding would produce around 100+ binary columns. For this case study, you'll use **label encoding** for simplicity — it works perfectly with decision trees, and you'll also test logistic regression to compare.

For logistic regression to be fully principled, one-hot encoding would be preferred. You'll see whether this distinction affects performance in the model selection lesson.

## Step 4: Apply Label Encoding

Fit each encoder on the training data, transform both train and test:

```python
X_train_enc = X_train.copy()
X_test_enc  = X_test.copy()

encoders = {}

for col in X_train.columns:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train[col])
    X_test_enc[col]  = le.transform(X_test[col])
    encoders[col]    = le   # save for later use

print("Encoding complete.")
print(f"\nX_train_enc dtypes:\n{X_train_enc.dtypes.value_counts()}")
print(f"\nFirst 3 rows (encoded):")
print(X_train_enc.head(3))
```

Output:
```
Encoding complete.

X_train_enc dtypes:
int64    21
dtype: int64

First 3 rows (encoded):
   cap_shape  cap_surface  cap_color  bruises  odor  gill_attachment  ...
0          5            2          4        1     5                1
1          5            2          9        1     0                1
2          2            2          4        0     0                1
```

Every column is now a 0-based integer. The `LabelEncoder` for each column maps the sorted list of unique values to 0, 1, 2, … in alphabetical order. For example, for `odor`: `a`→0, `c`→1, `f`→2, `l`→3, `m`→4, `n`→5, `p`→6, `s`→7, `y`→8.

## Step 5: Handle Unseen Categories in the Test Set

A potential failure mode: if the test set contains a category value not seen in training, `LabelEncoder.transform()` will raise an error.

```python
# Check for unseen categories
issues = []
for col in X_train.columns:
    train_vals = set(X_train[col].unique())
    test_vals  = set(X_test[col].unique())
    unseen = test_vals - train_vals
    if unseen:
        issues.append((col, unseen))

if issues:
    for col, vals in issues:
        print(f"WARNING: {col} has unseen test values: {vals}")
else:
    print("No unseen categories in test set.")
```

Output:
```
No unseen categories in test set.
```

For this dataset, all categories present in the test set also appear in training. In production applications, you would add fallback handling (e.g., map unknown values to a default category or use `handle_unknown="use_encoded_value"` with `OrdinalEncoder`).

## Step 6: Feature Summary

```python
summary = pd.DataFrame({
    "column": X_train.columns,
    "n_unique_values": [X_train[c].nunique() for c in X_train.columns],
    "encoded_range": [f"0–{X_train_enc[c].max()}" for c in X_train.columns],
}).set_index("column")

print(summary)
```

Output:
```
                        n_unique_values encoded_range
column
cap_shape                             6           0–5
cap_surface                           4           0–3
cap_color                            10           0–9
bruises                               2           0–1
odor                                  9           0–8
gill_attachment                       3           0–2
gill_spacing                          2           0–1
gill_size                             2           0–1
gill_color                           12          0–11
stalk_shape                           2           0–1
stalk_root                            7           0–6
stalk_surface_above                   4           0–3
stalk_surface_below                   4           0–3
stalk_color_above                     9           0–8
stalk_color_below                     9           0–8
veil_color                            4           0–3
ring_number                           3           0–2
ring_type                             8           0–7
spore_print_color                     9           0–8
population                            6           0–5
habitat                               7           0–6
```

21 features, all encoded as compact integers. The feature matrix is ready for model training.

## Step 7: Quick Sanity Check

```python
assert X_train_enc.isnull().sum().sum() == 0, "Unexpected NaN in training features"
assert X_test_enc.isnull().sum().sum() == 0, "Unexpected NaN in test features"
assert X_train_enc.shape == (6499, 21)
assert X_test_enc.shape  == (1625, 21)

print("All checks passed.")
print(f"\nFinal feature matrix shapes:")
print(f"  X_train_enc: {X_train_enc.shape}")
print(f"  X_test_enc:  {X_test_enc.shape}")
```

Output:
```
All checks passed.

Final feature matrix shapes:
  X_train_enc: (6499, 21)
  X_test_enc:  (1625, 21)
```

## Feature Engineering Summary

| Step | Decision | Rationale |
|------|----------|-----------|
| Encode target | `e`→0, `p`→1 | Poisonous = positive class (the one to detect) |
| Train/test split | 80/20, `stratify=y` | Preserves class balance; prevents data leakage |
| Label encode all features | Integers 0–k | Compatible with all classifiers; simple and compact |
| Fit encoders on train only | Applied to test with `.transform()` | Prevents data leakage |

Unlike the house rental case study, no new derived features were created here. The raw physical features — once encoded — are already informative. In a production model, you might experiment with interaction features (e.g., `odor × gill_color`), but for this dataset the raw features are sufficient.

## What's Next

In the next lesson, you'll **train and compare** multiple classifiers on this prepared feature matrix, tune the decision threshold for safety-critical use, and identify the most important features.
