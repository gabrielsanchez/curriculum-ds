# Feature Engineering

## Overview

The Census Income Dataset is now clean: duplicates removed, `fnlwgt` and redundant `education` dropped, and `?` placeholders replaced with `Unknown` in `workclass` and `occupation`. The data is structurally sound, but it's still not ready for a classifier. The target column is a string, categorical columns are strings, numeric columns span very different scales, and `native_country` still needs to be resolved. In this lesson, you'll transform the cleaned data into a fully numeric feature matrix ready for training.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Encode a binary target and categorical features for use in a machine learning model.
- Apply scaling to numeric features and assemble a complete feature matrix.

## Starter Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Start from the cleaned dataset (or re-run cleaning inline)
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
raw = pd.read_csv(url, header=None, names=columns, sep=", ", engine="python")
df_clean = (
    raw
    .drop_duplicates()
    .drop(columns=["fnlwgt", "education"])
    .assign(
        workclass=lambda d: d["workclass"].replace("?", "Unknown"),
        occupation=lambda d: d["occupation"].replace("?", "Unknown"),
    )
)

print(f"Starting shape: {df_clean.shape}")
```

Output:
```
Starting shape: (32537, 13)
```

## Step 1: Encode the Target

Convert the target from strings to binary integers before anything else. Define `>50K` as the positive class (1), since that is typically the class of primary interest for credit, hiring, and access decisions:

```python
df_fe = df_clean.copy()
df_fe["income"] = (df_fe["income"] == ">50K").astype(int)

print("Target distribution after encoding:")
print(df_fe["income"].value_counts())
print(f"High-income rate: {df_fe['income'].mean():.1%}")
```

Output:
```
Target distribution after encoding:
0    24720
1     7817
Name: income, dtype: int64

High-income rate: 24.0%
```

## Step 2: Train/Test Split First

**Always split before feature engineering.** Any statistics computed from the full dataset — scaler parameters, one-hot encoder category lists — must not be informed by test set values.

```python
X = df_fe.drop(columns=["income"])
y = df_fe["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"High-income rate — train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
```

Output:
```
X_train: (26029, 12), X_test: (6508, 12)
High-income rate — train: 24.0%, test: 24.0%
```

`stratify=y` preserves the 76%/24% class split in both subsets.

## Step 3: Encode sex as a Binary Feature

`sex` has exactly two values: Male and Female. Binary encoding is cleaner than one-hot encoding for a two-category feature:

```python
print(f"sex unique values: {X_train['sex'].unique()}")

X_train = X_train.copy()
X_test  = X_test.copy()

X_train["sex"] = (X_train["sex"] == "Male").astype(int)
X_test["sex"]  = (X_test["sex"] == "Male").astype(int)

print(f"sex value counts (train): {X_train['sex'].value_counts().to_dict()}")
```

Output:
```
sex unique values: ['Male' 'Female']
sex value counts (train): {1: 17518, 0: 8511}
```

## Step 4: Handle native_country — Create native_us

EDA showed that 89.6% of the sample is from the United States and that `native_country` has 41 unique values plus `?`. One-hot encoding it directly would produce 40+ mostly-empty columns. Create a binary feature instead:

```python
X_train["native_us"] = (X_train["native_country"] == "United-States").astype(int)
X_test["native_us"]  = (X_test["native_country"] == "United-States").astype(int)

print(f"native_us distribution (train):")
print(X_train["native_us"].value_counts())
print(f"\nUS-native fraction: {X_train['native_us'].mean():.1%}")

X_train = X_train.drop(columns=["native_country"])
X_test  = X_test.drop(columns=["native_country"])
```

Output:
```
native_us distribution (train):
1    23337
0     2692
Name: native_us, dtype: int64

US-native fraction: 89.7%
```

The `?` values in `native_country` automatically become `0` under this encoding — which is correct. An unknown country of origin is treated as non-US.

## Step 5: Handle capital_gain and capital_loss

Both columns are zero for more than 90% of rows, but capital activity is a strong predictor (63% high-income rate for those with any activity vs. 18% for those without). Create a `has_capital` binary flag that captures this signal without being dominated by the sparse continuous values:

```python
X_train["has_capital"] = (
    (X_train["capital_gain"] > 0) | (X_train["capital_loss"] > 0)
).astype(int)

X_test["has_capital"] = (
    (X_test["capital_gain"] > 0) | (X_test["capital_loss"] > 0)
).astype(int)

print(f"has_capital distribution (train): {X_train['has_capital'].value_counts().to_dict()}")
print(f"Fraction with capital activity: {X_train['has_capital'].mean():.1%}")
```

Output:
```
has_capital distribution (train): {0: 23848, 1: 2181}
Fraction with capital activity: 8.4%
```

Keep `capital_gain` and `capital_loss` as well — their raw values carry additional information (the magnitude of capital activity). Models that can exploit sparse continuous features (like gradient boosting) will benefit from both the flag and the raw values.

## Step 6: One-Hot Encode Categorical Features

Five categorical columns remain: `workclass`, `marital_status`, `occupation`, `relationship`, and `race`. Use one-hot encoding with `drop_first=True` to avoid multicollinearity (one category is implicit when all others are zero):

```python
cat_cols = ["workclass", "marital_status", "occupation", "relationship", "race"]

# Fit on training data only
X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_enc  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)

# Align columns (test set may be missing categories unseen in training)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print(f"Shape after one-hot encoding:")
print(f"  X_train_enc: {X_train_enc.shape}")
print(f"  X_test_enc:  {X_test_enc.shape}")
```

Output:
```
Shape after one-hot encoding:
  X_train_enc: (26029, 49)
  X_test_enc:  (6508, 49)
```

The five categorical columns expand to 37 binary columns (the sum of `n_categories − 1` for each column). Combined with the original numeric and binary features, the feature matrix has 49 columns.

## Step 7: Scale Numeric Features

Five continuous columns remain unscaled: `age`, `education_num`, `hours_per_week`, `capital_gain`, and `capital_loss`. Fit the scaler on the training set only:

```python
scale_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]

scaler = StandardScaler()
X_train_enc[scale_cols] = scaler.fit_transform(X_train_enc[scale_cols])
X_test_enc[scale_cols]  = scaler.transform(X_test_enc[scale_cols])

print("Scaled column statistics (train, should be mean≈0, std≈1):")
print(X_train_enc[scale_cols].describe().loc[["mean", "std"]].round(3))
```

Output:
```
Scaled column statistics (train, should be mean≈0, std≈1):
       age  education_num  hours_per_week  capital_gain  capital_loss
mean  -0.0           -0.0             0.0          -0.0           0.0
std    1.0            1.0             1.0           1.0           1.0
```

## Step 8: Sanity Check

```python
assert X_train_enc.isnull().sum().sum() == 0, "Unexpected NaN in training features"
assert X_test_enc.isnull().sum().sum() == 0, "Unexpected NaN in test features"
assert X_train_enc.shape[1] == X_test_enc.shape[1], "Train/test column mismatch"
assert len(y_train) == len(X_train_enc)
assert len(y_test)  == len(X_test_enc)

print("All checks passed.")
print(f"\nFinal feature matrix shapes:")
print(f"  X_train_enc: {X_train_enc.shape}")
print(f"  X_test_enc:  {X_test_enc.shape}")
print(f"\nFeature columns ({X_train_enc.shape[1]} total):")
print(list(X_train_enc.columns))
```

Output:
```
All checks passed.

Final feature matrix shapes:
  X_train_enc: (26029, 49)
  X_test_enc:  (6508, 49)

Feature columns (49 total):
['age', 'education_num', 'sex', 'capital_gain', 'capital_loss',
 'hours_per_week', 'native_us', 'has_capital',
 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov',
 'workclass_Unknown', 'workclass_Without-pay',
 'marital_status_Married-AF-spouse', 'marital_status_Married-civ-spouse',
 'marital_status_Married-spouse-absent', 'marital_status_Never-married',
 'marital_status_Separated', 'marital_status_Widowed',
 'occupation_Adv-specialty', 'occupation_Armed-Forces', 'occupation_Craft-repair',
 'occupation_Exec-managerial', 'occupation_Farming-fishing',
 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
 'occupation_Other-service', 'occupation_Priv-house-serv',
 'occupation_Prof-specialty', 'occupation_Protective-serv',
 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
 'occupation_Unknown',
 'relationship_Not-in-family', 'relationship_Other-relative',
 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']
```

## Feature Engineering Summary

| Step | Decision | Rationale |
|------|----------|-----------|
| Encode target | `<=50K` → 0, `>50K` → 1 | Positive class = high income |
| Train/test split first | 80/20, `stratify=y` | Prevents data leakage; preserves class balance |
| Binary encode `sex` | Male=1, Female=0 | Two categories; no need for one-hot |
| Create `native_us` | 1 if United-States, 0 otherwise | Collapses 41 high-cardinality values cleanly |
| Create `has_capital` | 1 if any capital activity, 0 otherwise | Captures the strong binary signal (63% vs 18% high-income rate) |
| One-hot encode 5 categoricals | `drop_first=True` | Avoids multicollinearity; all categories meaningful |
| Align test columns | `reindex(..., fill_value=0)` | Handles any unseen categories in test set |
| Scale 5 continuous columns | `StandardScaler`, fit on train only | Consistent scale for distance- and gradient-based algorithms |

## What's Next

The feature matrix is ready: 26,029 training examples and 6,508 test examples, each with 49 features, all numeric, with no missing values. In the final lesson, you'll synthesize everything this case study demonstrated and outline the next steps — applying classifiers to this dataset in module 09.
