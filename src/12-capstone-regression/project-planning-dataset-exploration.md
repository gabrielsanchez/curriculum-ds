# Project Planning & Dataset Exploration

## Overview

In the previous lesson, you chose your dataset and reviewed the deliverables. In this lesson, you'll complete the first half of your capstone notebook: framing the regression problem precisely, exploring your dataset with EDA, cleaning it systematically, and building a feature matrix ready for modeling. This lesson provides scaffolding — guiding questions, checklists, and code templates — but the analysis is yours to design.

## Learning Objectives

By the end of this lesson, you will have:

- Acquired and explored a dataset appropriate for regression.
- Cleaned the data and engineered features in preparation for modeling.

## Starter Code

Use the included [*Colaboratory notebook*](#) to build your capstone. The templates below are starting points — adapt them to your dataset.

---

## Phase 1: Problem Framing

Before writing a line of code, write one paragraph in your notebook that answers these questions:

1. **What are you predicting?** Name the exact target variable and its units.
2. **Why does it matter?** Who would use this prediction and for what decision?
3. **What data do you have?** Brief description of the source, size, and features.
4. **How will you measure success?** Which metric matters most for your use case, and why?

**Example problem statement (Ames Housing):**
> "This project predicts the sale price (in USD) of residential properties in Ames, Iowa using the Ames Housing dataset from Kaggle (1,460 training samples, 79 features). Accurate price predictions would help real estate agents set listing prices and buyers evaluate whether a property is fairly priced. Because large price errors are particularly costly — a $50,000 miss on a $200,000 house is far worse than a $5,000 miss — I will optimize for RMSE over MAE."

---

## Phase 2: Loading and Initial Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# ── Basic inspection ──────────────────────────────────────────────────────────
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:")
display(df.head())

# ── Target variable ───────────────────────────────────────────────────────────
target = "your_target_column"
print(f"\nTarget summary:\n{df[target].describe()}")
print(f"Unique values: {df[target].nunique()}")

# ── Missing values ────────────────────────────────────────────────────────────
missing = df.isnull().mean().sort_values(ascending=False)
print(f"\nMissing value rates (top 10):\n{missing[missing > 0].head(10).round(3)}")
```

**After running this cell, write 2–3 sentences in your notebook interpreting what you see.** Don't just report numbers — explain what they mean for your next steps.

---

## Phase 3: Exploratory Data Analysis

EDA is not a checklist to execute — it is a conversation with the data. Every plot should prompt a question or reveal a decision. You need **at least five visualizations**, each followed by a written interpretation.

### Required Visualizations

#### 1. Target Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Raw
axes[0].hist(df[target], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title(f"{target} — Raw Distribution")
axes[0].set_xlabel(target)
axes[0].set_ylabel("Count")

# Log-transformed (if target is right-skewed)
if df[target].skew() > 1:
    axes[1].hist(np.log1p(df[target]), bins=40, color="steelblue", edgecolor="white")
    axes[1].set_title(f"log(1 + {target}) — Log Distribution")
else:
    axes[1].hist(df[target], bins=40, color="coral", edgecolor="white")
    axes[1].set_title(f"{target} — Identical (not skewed)")

plt.tight_layout()
plt.show()

print(f"Skewness: {df[target].skew():.2f}")
```

**Write:** Is the target right-skewed? If so, will you log-transform it? How will this affect how you interpret predictions?

#### 2. Numeric Feature Distributions

```python
numeric_cols = df.select_dtypes(include="number").drop(columns=[target]).columns

# Plot up to 12 histograms
cols_to_plot = numeric_cols[:12]
n = len(cols_to_plot)
fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, (n + 2) // 3 * 3))

for ax, col in zip(axes.flat, cols_to_plot):
    ax.hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("")

for ax in axes.flat[len(cols_to_plot):]:
    ax.set_visible(False)

plt.suptitle("Numeric Feature Distributions", y=1.01)
plt.tight_layout()
plt.show()
```

**Write:** Which features are skewed? Which have potential outliers? Which might benefit from log transformation?

#### 3. Correlation with the Target

```python
correlations = df.select_dtypes(include="number").corr()[target].drop(target)
correlations = correlations.sort_values(key=abs, ascending=False)

plt.figure(figsize=(8, max(4, len(correlations) * 0.3)))
colors = ["#e74c3c" if c > 0 else "#3498db" for c in correlations]
plt.barh(correlations.index, correlations.values, color=colors)
plt.axvline(0, color="black", linewidth=0.5)
plt.title(f"Pearson Correlation with {target}")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

print("Top 5 positive correlations:")
print(correlations.head(5).round(3))
print("\nTop 5 negative correlations:")
print(correlations.tail(5).round(3))
```

**Write:** Which features show the strongest linear relationship with the target? Are there any surprising correlations you didn't expect? Are any features negatively correlated with the target?

#### 4. Scatter Plots of Top Predictors vs. Target

```python
top_features = correlations.abs().sort_values(ascending=False).head(4).index

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, feat in zip(axes.flat, top_features):
    ax.scatter(df[feat], df[target], alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel(feat)
    ax.set_ylabel(target)
    ax.set_title(f"{feat} vs. {target}  (r={df[feat].corr(df[target]):.2f})")

plt.suptitle("Top Features vs. Target", y=1.01)
plt.tight_layout()
plt.show()
```

**Write:** Does the relationship look linear or non-linear? Are there outliers? Are there clusters? Does the variance of the target change across the feature's range (heteroscedasticity)?

#### 5. Categorical Features vs. Target

```python
cat_cols = df.select_dtypes(include="object").columns.tolist()

if cat_cols:
    # Pick the categorical column most associated with the target
    # (use the one with the most obvious grouping, or the one you expect matters most)
    focus_col = cat_cols[0]  # replace with the column you want to examine

    order = df.groupby(focus_col)[target].median().sort_values(ascending=False).index

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=focus_col, y=target, order=order, palette="Blues_r")
    plt.title(f"{target} by {focus_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    print(f"Median {target} by {focus_col}:")
    print(df.groupby(focus_col)[target].median().sort_values(ascending=False).round(2))
```

**Write:** Which category level has the highest/lowest median target value? Is the difference large enough to be practically meaningful? Does this suggest this variable will be a strong feature?

### Additional Visualizations (Choose at Least One More)

- Correlation heatmap of all numeric features
- Box plots of numeric features to identify outliers
- Time series plot if the dataset has a date column
- Geographic scatter if there are lat/lon features

---

## Phase 4: Data Cleaning

Cleaning decisions must be documented. For each decision, write in your notebook: **what you changed, why you changed it, and what evidence from EDA justified it.**

### Cleaning Checklist

Work through each of these in order. Not all will apply to your dataset.

#### Missing Values

```python
# Audit missing values with context
for col in df.columns[df.isnull().any()]:
    missing_pct = df[col].isnull().mean()
    dtype = df[col].dtype
    print(f"{col:30s}  missing: {missing_pct:.1%}  dtype: {dtype}")
```

For each column with missing values, decide:

| Missing rate | Typical decision |
|-------------|-----------------|
| < 5% | Impute with median (numeric) or mode (categorical) |
| 5–30% | Impute, or create a binary "was_missing" indicator feature |
| > 50% | Drop the column unless missingness itself is informative |
| Any | Drop rows only when very few rows are affected and you have evidence it's truly random |

```python
df_clean = df.copy()

# Example: impute median for a numeric column
df_clean["numeric_col"] = df_clean["numeric_col"].fillna(df_clean["numeric_col"].median())

# Example: impute mode for a categorical column
df_clean["cat_col"] = df_clean["cat_col"].fillna(df_clean["cat_col"].mode()[0])

# Example: drop columns with too much missing data
df_clean = df_clean.drop(columns=["col_with_80pct_missing"])
```

#### Outliers

```python
# Identify outliers using the IQR method
numeric_cols = df_clean.select_dtypes(include="number").columns

for col in numeric_cols:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_outliers = ((df_clean[col] < q1 - 3*iqr) | (df_clean[col] > q3 + 3*iqr)).sum()
    if n_outliers > 0:
        print(f"{col}: {n_outliers} extreme outliers (3×IQR)")
```

For each outlier column, decide:
- **Remove rows:** only if the value is physically impossible (e.g., negative age) and affects few rows
- **Cap (Winsorize):** if values are extreme but plausible (large houses, high incomes)
- **Keep:** if the outliers are genuine and the model should learn from them

#### Duplicates

```python
n_dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {n_dupes}")
if n_dupes > 0:
    df_clean = df_clean.drop_duplicates()
```

#### Data Type Fixes

```python
# Fix columns stored as the wrong type
# e.g., a year column stored as float → convert to int
# e.g., a numeric column stored as string due to commas → strip and convert
df_clean["year_col"] = df_clean["year_col"].astype(int)
```

#### Cleaning Summary Table

At the end of cleaning, add a table to your notebook:

| Issue | Columns affected | Decision | Rows/columns changed |
|-------|-----------------|----------|---------------------|
| Missing values | ... | Imputed with median | ... |
| Outliers | ... | Capped at 99th pct | 0 rows removed |
| Duplicates | — | Dropped | ... |

---

## Phase 5: Feature Engineering

Feature engineering is where domain knowledge creates value. You need at least two engineered or transformed features. Document each with a justification.

### Always: Train/Test Split First

```python
X = df_clean.drop(columns=[target])
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")
print(f"y_train mean: {y_train.mean():.2f},  y_test mean: {y_test.mean():.2f}")
```

Split before any fit-based engineering. Statistics (means, encodings, scalers) must be computed on training data only.

### Log-Transform the Target (if skewed)

```python
import numpy as np

if y_train.skew() > 1:
    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)
    print(f"Target skewness before log: {y_train.skew():.2f}")
    print(f"Target skewness after log:  {y_train_log.skew():.2f}")
```

Remember: if you log-transform the target, invert predictions at evaluation time with `np.expm1()`.

### Derived Features

```python
X_train = X_train.copy()
X_test  = X_test.copy()

# Example: ratio feature
X_train["rooms_per_person"] = X_train["total_rooms"] / X_train["population"]
X_test["rooms_per_person"]  = X_test["total_rooms"]  / X_test["population"]

# Example: binary indicator
X_train["is_new"] = (X_train["year_built"] >= 2000).astype(int)
X_test["is_new"]  = (X_test["year_built"]  >= 2000).astype(int)

# Example: log of a skewed feature
X_train["log_area"] = np.log1p(X_train["living_area"])
X_test["log_area"]  = np.log1p(X_test["living_area"])
```

For each engineered feature, write: "I created `[feature]` because `[reasoning from EDA]`."

### Encode Categorical Variables

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Identify categorical columns
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns: {cat_cols}")

# Check cardinality
for col in cat_cols:
    print(f"  {col}: {X_train[col].nunique()} unique values")
```

Follow the encoding strategy from module 07:
- Ordinal (ordered) categories → `OrdinalEncoder` with explicit order
- Nominal categories with ≤15 values → `pd.get_dummies()` or `OneHotEncoder`
- Nominal categories with >15 values → group rare values as "Other", then one-hot

### Scale Numeric Features

```python
from sklearn.preprocessing import StandardScaler

# Scale after encoding — only numeric columns
num_cols = X_train.select_dtypes(include="number").columns.tolist()

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
```

Or use a `ColumnTransformer` Pipeline (preferred — prevents leakage automatically):

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")
```

### Feature Engineering Summary Table

Add this to your notebook:

| Feature | Source | Type | Justification |
|---------|--------|------|---------------|
| (original features) | Raw | Numeric / Categorical | — |
| `your_derived_feature` | `col_a / col_b` | Numeric | EDA showed X and Y interact |
| `log_target` | `log1p(target)` | Numeric | Target was right-skewed (skewness=X) |

---

## Phase 5 Checkpoint

Before moving to modeling, verify:

```python
# No missing values in the processed feature matrix
import numpy as np
assert not np.isnan(X_train_processed).any(), "NaN values in training features"
assert not np.isnan(X_test_processed).any(),  "NaN values in test features"

print(f"Train: {X_train_processed.shape}")
print(f"Test:  {X_test_processed.shape}")
print(f"Target range — train: [{y_train.min():.2f}, {y_train.max():.2f}]")
print("Checkpoint passed — ready for modeling.")
```

## What's Next

In the next lesson, you'll train at least three regression models, compare them rigorously with cross-validation, tune the best model's hyperparameters, and analyze residuals — building the complete evaluation section of your capstone notebook.
