# Summary

## Overview

You've completed the full data preparation pipeline for the House Rent Prediction Dataset — from a raw CSV file with a complex `Floor` column and extreme outliers to a clean, scaled, encoded feature matrix with 71 features ready for a regression model. In this final lesson, you'll consolidate what you've built, draw interpretable insights from the data, and outline the modeling steps that would follow.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Present final insights from EDA and data preprocessing.
- Propose next steps for modeling.

## Starter Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# All variables from previous lessons should be in memory in your notebook:
# df         — raw data
# df_clean   — cleaned data
# X_train, X_test, y_train, y_test   — split raw features and target
# X_train_processed, X_test_processed — final feature matrices
# y_train_log, y_test_log            — log-transformed targets
```

## What Was Built

Across three lessons, the raw dataset went through a complete transformation:

```
Raw CSV (4,746 rows × 12 columns)
    ↓ EDA: identify distributions, anomalies, relationships
    ↓ Cleaning: parse Floor, fix Size outliers, cap Rent, fix Bathroom, simplify Locality
    ↓ Feature Engineering: derived features, encoding, scaling, ColumnTransformer
Final feature matrix (3,785 train rows × 71 features)
```

### Cleaning Decisions Made

| Issue Found in EDA | Cleaning Decision | Rationale |
|--------------------|-------------------|-----------|
| `Floor` encodes two values as string | Parsed into `current_floor` + `total_floors` | Two separate numeric signals |
| 14 listings with Size < 100 sq ft | Removed | Physically implausible for given BHK |
| Extreme Rent outliers (max ₹3.5M) | Winsorized at 99th percentile | Preserve rows, reduce outlier influence |
| 3 listings with Bathroom > BHK + 1 | Capped Bathroom at BHK + 1 | Data entry error, not genuine |
| 2,235 unique Area Locality values | Kept top 50, collapsed rest to "Other" | Prevent high-cardinality one-hot explosion |
| `Posted On` is a date string | Extracted `post_month`, `post_year` | Enable seasonality signal |

### Features Created

| Category | Features |
|----------|----------|
| Parsed from raw columns | `current_floor`, `total_floors`, `post_month`, `post_year` |
| Derived numeric | `size_per_bhk`, `bath_per_bhk`, `log_size`, `is_ground_floor` |
| Ordinal encoded | `furnishing_encoded` (0/1/2) |
| One-hot encoded | `City`, `Area Type`, `Area Locality`, `Tenant Preferred`, `Point of Contact` |
| Target transformed | `log(1 + Rent)` |

## Key Insights from EDA

### 1. Location Is the Dominant Driver

```python
print(df_clean.groupby("City")["Rent"].median().sort_values(ascending=False))
```

Output:
```
City
Mumbai       35000
Delhi        22000
Bangalore    18000
Hyderabad    15000
Chennai      12000
Kolkata      10000
Name: Rent, dtype: int64
```

Mumbai's median rent is 3.5× Kolkata's. Any model that excludes city information will systematically misprice listings. The `City` one-hot columns will almost certainly be among the most important features.

### 2. Rent Is Not Linearly Scaled

The raw `Rent` distribution is highly right-skewed:

```python
print(f"Mean rent:   ₹{df_clean['Rent'].mean():,.0f}")
print(f"Median rent: ₹{df_clean['Rent'].median():,.0f}")
print(f"Std dev:     ₹{df_clean['Rent'].std():,.0f}")
```

Output:
```
Mean rent:   ₹33,947
Median rent: ₹16,000
Std dev:     ₹44,219
```

The mean is more than 2× the median — a classic sign of right skew. A model predicting the raw rent on a linear scale will struggle with this. Log-transforming the target `y` produces a near-normal distribution and typically improves regression performance significantly.

### 3. Furnishing Status Has Clear, Interpretable Effect

```python
print(df_clean.groupby("Furnishing Status")["Rent"].median().sort_values(ascending=False))
```

Output:
```
Furnishing Status
Furnished        22000
Semi-Furnished   15000
Unfurnished      10000
```

Furnished units command a 2.2× premium over unfurnished at the median. This is both statistically meaningful and domain-sensible — tenants pay for convenience.

### 4. Size and BHK Are Correlated but Not Redundant

```python
print(df_clean[["BHK", "Size", "Bathroom", "Rent"]].corr().round(2))
```

Output:
```
          BHK  Size  Bathroom  Rent
BHK      1.00  0.61      0.66  0.41
Size     0.61  1.00      0.57  0.58
Bathroom 0.66  0.57      1.00  0.39
Rent     0.41  0.58      0.39  1.00
```

`Size` (r = 0.58) is a stronger predictor of rent than `BHK` (r = 0.41) or `Bathroom` (r = 0.39). The engineered `size_per_bhk` feature captures spaciousness in a way neither column does alone.

### 5. Area Locality Signal Is Real but Noisy

The top localities by median rent cluster in high-demand neighborhoods of Mumbai, Delhi, and Bangalore. However, most localities appear fewer than 5 times — the signal is diluted by sparsity. Grouping rare localities into "Other" is a reasonable first step; a production model might use target encoding to capture the locality signal more cleanly.

## Final Dataset Snapshot

```python
print("=== Cleaned Dataset ===")
print(f"Rows: {df_clean.shape[0]}")
print(f"Columns: {df_clean.shape[1]}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")

print("\n=== Processed Feature Matrix ===")
print(f"Train: {X_train_processed.shape}")
print(f"Test:  {X_test_processed.shape}")
print(f"NaN in processed train: {np.isnan(X_train_processed).sum()}")
```

Output:
```
=== Cleaned Dataset ===
Rows: 4732
Columns: 14
Missing values: 0

=== Processed Feature Matrix ===
Train: (3785, 71)
Test:  (947, 71)
NaN in processed train: 0
```

## Next Steps: Modeling

The feature matrix is ready. The natural next step is to train a regression model to predict `log(1 + Rent)` and evaluate it on the held-out test set. Here's what that would look like:

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Train a regularized linear regression model
model = Ridge(alpha=1.0)
model.fit(X_train_processed, y_train_log)

# Predict and invert the log transform
y_pred_log  = model.predict(X_test_processed)
y_pred      = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_log)

# Evaluate
mae = mean_absolute_error(y_test_orig, y_pred)
r2  = r2_score(y_test_orig, y_pred)

print(f"Mean Absolute Error: ₹{mae:,.0f}")
print(f"R² Score:            {r2:.3f}")
```

Expected output (approximate):
```
Mean Absolute Error: ₹8,200
R² Score:            0.71
```

A Ridge regression on this feature matrix achieves an R² of ~0.71 — meaning the features explain about 71% of the variance in rental price. The remaining 29% reflects factors the dataset doesn't capture: exact floor plan, building amenities, proximity to transit, renovation quality, and negotiation.

### Where to Go From Here

| Improvement | Approach |
|------------|----------|
| Better locality signal | Replace one-hot with target encoding for `Area Locality` |
| Non-linear patterns | Try Random Forest or Gradient Boosting |
| Hyperparameter tuning | Use `GridSearchCV` or `RandomizedSearchCV` |
| Cross-validation | Replace single split with 5-fold CV for more reliable R² estimate |
| Model interpretability | Use `eli5` or SHAP values to explain predictions |

## What This Case Study Demonstrated

This case study is a compressed version of what data scientists do before any model is trained. The actual modeling took one code cell — the preceding work took three full lessons. That ratio is realistic.

| Phase | Skills applied |
|-------|----------------|
| EDA | `describe()`, histograms, box plots, scatter plots, correlation matrix |
| Cleaning | Column parsing, outlier capping, Winsorization, constraint validation, cardinality reduction |
| Feature Engineering | Log transforms, ratio features, binary indicators, ordinal and one-hot encoding, `ColumnTransformer`, `Pipeline` |

The dataset, the decisions, and the code are all yours to extend. A good next exercise is to swap out Ridge regression for a tree-based model (which doesn't require scaling) and compare the results — you now have the pipeline to do it.

## Reflection Questions

1. The `Area Locality` column was grouped into top-50 values + "Other." What are the tradeoffs of this approach compared to target encoding (replacing each locality with its mean rent)?

2. You log-transformed both `Rent` (the target) and `Size` (a feature). What property of a distribution makes log transformation helpful, and why does it matter for linear models specifically?

3. If a new listing appeared in a city not in the training data — for example, Pune — what would happen when you called `preprocessor.transform()` on it? How does `handle_unknown="ignore"` in `OneHotEncoder` handle this case?
