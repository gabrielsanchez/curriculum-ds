# Feature Engineering

## Overview

In the previous lesson, you cleaned the House Rent dataset — parsing the `Floor` column, removing implausible sizes, capping outliers, and simplifying `Area Locality`. The data is now structurally sound, but it's still not ready for a regression model. Categorical columns are strings, numeric columns span very different scales, and there are derived signals hidden in the raw variables that a model can't discover on its own. In this lesson, you'll transform the cleaned data into a numeric feature matrix ready for training.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Create location-based or derived features (e.g., price per sq. ft.).
- Encode categorical variables (e.g., type of property).

## Starter Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load cleaned dataset (output from the previous lesson)
# In the notebook, df_clean is already in memory from the previous cells
df_fe = df_clean.copy()
print(f"Starting shape: {df_fe.shape}")
print(df_fe.dtypes)
```

## Step 1: Train/Test Split First

**Always split before feature engineering.** Any statistic computed from the full dataset (mean rent by city, target encoding, scaler parameters) must not be informed by test set values. Split first, engineer on training data, apply to test data.

```python
X = df_fe.drop(columns=["Rent"])
y = df_fe["Rent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train median: ₹{y_train.median():,.0f}")
print(f"y_test  median: ₹{y_test.median():,.0f}")
```

## Step 2: Log-Transform the Target

EDA showed that `Rent` is right-skewed. Many regression algorithms perform better when the target is approximately normally distributed. Log-transform `y`:

```python
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Rent — Raw")
axes[0].set_xlabel("Rent (INR)")

axes[1].hist(y_train_log, bins=50, color="steelblue", edgecolor="white")
axes[1].set_title("Rent — Log-Transformed")
axes[1].set_xlabel("log(1 + Rent)")
plt.tight_layout()
plt.show()
```

When making predictions, remember to invert the transformation: `predicted_rent = np.expm1(model.predict(X_test))`.

## Step 3: Create Derived Features

New features that combine or transform existing ones can capture patterns the raw variables don't express directly.

### Price Per Square Foot

The most natural derived feature for real estate — it normalizes rent by size:

```python
# Compute on train set only, then apply to test
# (no leakage here since this is purely structural, not stat-based)
X_train = X_train.copy()
X_test  = X_test.copy()

X_train["size_per_bhk"]     = X_train["Size"] / X_train["BHK"]
X_test["size_per_bhk"]      = X_test["Size"]  / X_test["BHK"]
```

> Note: We can't compute `price_per_sqft` at this stage because it requires `Rent` — using the target to create a feature would be **data leakage**. `size_per_bhk` is a valid alternative: it captures how spacious each unit is relative to its size category.

### Is Ground Floor

Ground floor units are often preferred (no elevator needed) or avoided (security concerns). Create a binary indicator:

```python
X_train["is_ground_floor"] = (X_train["current_floor"] == 0).astype(int)
X_test["is_ground_floor"]  = (X_test["current_floor"]  == 0).astype(int)
```

### Bathroom to BHK Ratio

Captures whether a unit has more or fewer bathrooms than expected for its size:

```python
X_train["bath_per_bhk"] = X_train["Bathroom"] / X_train["BHK"]
X_test["bath_per_bhk"]  = X_test["Bathroom"]  / X_test["BHK"]
```

### Log of Size

Size is also right-skewed — log-transforming it produces a more symmetric distribution and helps linear models:

```python
X_train["log_size"] = np.log1p(X_train["Size"])
X_test["log_size"]  = np.log1p(X_test["Size"])
```

## Step 4: Inspect Categorical Columns

Identify which columns need encoding:

```python
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns ({len(num_cols)}): {num_cols}")
```

Output:
```
Categorical columns (5): ['Area Type', 'Area Locality', 'City',
                           'Furnishing Status', 'Tenant Preferred',
                           'Point of Contact']
Numeric columns (10): ['BHK', 'Size', 'Bathroom', 'current_floor',
                        'total_floors', 'post_month', 'post_year',
                        'size_per_bhk', 'is_ground_floor',
                        'bath_per_bhk', 'log_size']
```

### Cardinality Check

```python
for col in cat_cols:
    print(f"{col:25s} — {X_train[col].nunique()} unique values")
```

Output:
```
Area Type                 —  3 unique values
Area Locality             — 51 unique values
City                      —  6 unique values
Furnishing Status         —  3 unique values
Tenant Preferred          —  3 unique values
Point of Contact          —  3 unique values
```

`Area Locality` has 51 values (post-cleaning). One-hot encoding it will create ~50 binary columns — manageable but verbose. The others have 3–6 values each and are straightforward to encode.

## Step 5: Encoding Strategy

| Column | Values | Strategy | Reason |
|--------|--------|----------|--------|
| `Furnishing Status` | Unfurnished → Semi-Furnished → Furnished | Ordinal encoding | Clear ordering |
| `Area Type` | No natural order | One-hot encoding | Nominal |
| `City` | No natural order | One-hot encoding | Nominal |
| `Tenant Preferred` | No natural order | One-hot encoding | Nominal |
| `Point of Contact` | No natural order | One-hot encoding | Nominal |
| `Area Locality` | No natural order (51 values) | One-hot encoding with `drop="first"` | Nominal |

### Manual Ordinal Encoding for Furnishing Status

```python
furnish_order = [["Unfurnished", "Semi-Furnished", "Furnished"]]
ord_enc = OrdinalEncoder(categories=furnish_order)

X_train["furnishing_encoded"] = ord_enc.fit_transform(
    X_train[["Furnishing Status"]]
)
X_test["furnishing_encoded"] = ord_enc.transform(
    X_test[["Furnishing Status"]]
)

print(X_train[["Furnishing Status", "furnishing_encoded"]].value_counts())
```

Output:
```
Furnishing Status  furnishing_encoded
Semi-Furnished     1.0                   1485
Unfurnished        0.0                   1148
Furnished          2.0                    945
```

## Step 6: Assemble the `ColumnTransformer` Pipeline

Define which columns get which treatment and assemble everything into a single preprocessing object:

```python
# After adding derived features and ordinal encoding, update the column lists
numeric_features = [
    "BHK", "Size", "Bathroom", "current_floor", "total_floors",
    "post_month", "post_year", "size_per_bhk", "is_ground_floor",
    "bath_per_bhk", "log_size", "furnishing_encoded"
]

ohe_features = [
    "Area Type", "Area Locality", "City",
    "Tenant Preferred", "Point of Contact"
]

# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Pipeline for one-hot encoded features
ohe_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

# Assemble
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("ohe", ohe_transformer,     ohe_features)
])

# Fit on training data, transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape:  {X_test_processed.shape}")
```

Output:
```
Processed train shape: (3785, 71)
Processed test shape:  (947, 71)
```

The 71 columns consist of:
- 12 scaled numeric features
- ~59 one-hot encoded columns from `Area Type` (2), `Area Locality` (50), `City` (5), `Tenant Preferred` (2), `Point of Contact` (2)

## Step 7: Verify the Processed Data

```python
# No missing values
nan_count = np.isnan(X_train_processed).sum()
print(f"NaN values in processed train: {nan_count}")

# Numeric block should be approximately z-scored (mean ≈ 0, std ≈ 1)
n_numeric = len(numeric_features)
numeric_block = X_train_processed[:, :n_numeric]
print(f"\nNumeric block mean (should be ~0): {numeric_block.mean(axis=0).round(2)}")
print(f"Numeric block std  (should be ~1): {numeric_block.std(axis=0).round(2)}")
```

## Step 8: Feature Summary

Before modeling, it helps to document the features you've built:

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `BHK` | Original | Numeric | Scaled |
| `Size` | Original | Numeric | Scaled; outliers capped at 5000 |
| `Bathroom` | Original | Numeric | Scaled; capped at BHK + 1 |
| `current_floor` | Parsed from `Floor` | Numeric | Scaled |
| `total_floors` | Parsed from `Floor` | Numeric | Scaled |
| `post_month` | Parsed from `Posted On` | Numeric | Scaled |
| `post_year` | Parsed from `Posted On` | Numeric | Scaled |
| `size_per_bhk` | `Size / BHK` | Numeric | Spaciousness per unit |
| `is_ground_floor` | `current_floor == 0` | Binary | Ground floor indicator |
| `bath_per_bhk` | `Bathroom / BHK` | Numeric | Bathroom density |
| `log_size` | `log1p(Size)` | Numeric | Reduces skew |
| `furnishing_encoded` | `Furnishing Status` | Ordinal | 0/1/2 |
| `Area Type` (OHE) | Original | Binary (2 cols) | One-hot, drop first |
| `Area Locality` (OHE) | Original | Binary (50 cols) | One-hot, drop first |
| `City` (OHE) | Original | Binary (5 cols) | One-hot, drop first |
| `Tenant Preferred` (OHE) | Original | Binary (2 cols) | One-hot, drop first |
| `Point of Contact` (OHE) | Original | Binary (2 cols) | One-hot, drop first |

## What's Next

The preprocessing pipeline is complete. In the final lesson, you'll **summarize** the full case study — the key insights from EDA, the decisions made during cleaning, the features built, and a preview of what modeling on this dataset would look like.
