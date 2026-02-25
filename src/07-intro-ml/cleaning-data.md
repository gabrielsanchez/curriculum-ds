# Cleaning Data

## Overview

In the previous lesson, you learned how to source and evaluate data for ML. Now you'll do the hands-on work that bridges raw data and a trained model: **data cleaning**. Real-world data is almost never ready to feed directly into an algorithm — it contains missing values, duplicate rows, inconsistent formatting, and wrong data types. In this lesson, you'll use Pandas to identify and fix these issues systematically, preparing data for the next steps of the ML pipeline.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Identify and handle missing, duplicate, and inconsistent data.
- Use Pandas to preprocess data for machine learning.

## Key terms

**Missing value:** An absent entry in the dataset, represented in Pandas as `NaN` (Not a Number). Can occur due to data collection gaps, recording errors, or features that simply don't apply to a row.

**Imputation:** Replacing missing values with a reasonable estimate — commonly the column mean, median, or mode.

**`SimpleImputer`:** A scikit-learn class that imputes missing values with a chosen strategy (mean, median, most frequent, or constant).

**Duplicate row:** A row that is identical (or nearly identical) to another row in the dataset. Usually indicates a data entry or collection error.

**Data type coercion:** Converting a column to the appropriate type (e.g., converting a string `"42"` to the integer `42`).

**Outlier treatment:** Applying caps, transformations, or removals to extreme values before training (building on the EDA module).

**Leakage:** Accidentally including information in the training features that would not be available at prediction time — for example, including a column that was computed using the target variable.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Data cleaning for ML is more demanding than data cleaning for analysis alone. In analysis, a few missing values or duplicate rows might not change your conclusions. In ML, every row affects what the model learns. An algorithm cannot be fitted on a DataFrame that contains `NaN` values — most will raise an error. Duplicates inflate the influence of certain patterns. Inconsistent strings create phantom categories. Getting the data right before training is not optional.

## Setup: The Titanic Dataset

The [Titanic dataset](https://www.kaggle.com/c/titanic) is a classic ML dataset with realistic data quality issues — missing values, mixed types, and irrelevant columns. You'll use it throughout this lesson:

```python
import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset("titanic")
print(df.shape)          # (891, 15)
print(df.dtypes)
print(df.isnull().sum())
```

Output (missing values):
```
survived       0
pclass         0
sex            0
age          177   ← 19.9% missing
sibsp          0
parch          0
fare           0
embarked       2
class          0
who            0
adult_male     0
deck         688   ← 77.2% missing
embark_town    2
alive          0
alone          0
```

## Step 1: Drop Irrelevant and High-Missing Columns

Before fixing anything, drop columns that won't be useful for ML:

```python
# Columns to drop:
# - 'deck': 77% missing — too little data to be useful
# - 'alive': redundant with 'survived' (would be data leakage)
# - 'class', 'who', 'adult_male', 'embark_town', 'alone':
#   redundant with pclass, sex, age, embarked

df = df.drop(columns=[
    "deck", "alive", "class", "who",
    "adult_male", "embark_town", "alone"
])

print(df.shape)           # (891, 8)
print(df.isnull().sum())
```

**Watch for leakage.** The `alive` column encodes whether the passenger survived — including it as a feature would give the model the answer, making evaluation meaningless. Always audit your features with this question: *would this information be available at the time we need to make a prediction?*

## Step 2: Handle Duplicate Rows

```python
n_dupes = df.duplicated().sum()
print(f"Duplicate rows: {n_dupes}")

# Remove if any exist
df = df.drop_duplicates()
```

In the Titanic dataset there are no exact duplicates, but checking is always good practice. On datasets assembled from multiple sources, duplicates are common.

## Step 3: Handle Missing Values

Missing value strategy depends on the column:

| Strategy | When to use |
|----------|-------------|
| **Drop the rows** | Very few rows are missing, and those rows are not critical |
| **Drop the column** | Most values are missing (>50%) and imputation would be unreliable |
| **Impute with mean** | Numeric column, roughly symmetric distribution |
| **Impute with median** | Numeric column with outliers or skewed distribution |
| **Impute with mode** | Categorical column or integer column |
| **Impute with constant** | When "missing" itself is informative (e.g., fill with "Unknown") |

### Handling `age` — Median Imputation

`age` is numeric and right-skewed (some older passengers), so median is a better fill than mean:

```python
print(f"Age missing: {df['age'].isnull().sum()}")
print(f"Age median: {df['age'].median()}")

df["age"] = df["age"].fillna(df["age"].median())
print(f"Age missing after fill: {df['age'].isnull().sum()}")
```

### Handling `embarked` — Mode Imputation

Only 2 rows are missing. Fill with the most common port:

```python
most_common_port = df["embarked"].mode()[0]
df["embarked"] = df["embarked"].fillna(most_common_port)
```

### Using `SimpleImputer` from scikit-learn

For a pipeline-friendly approach, use `SimpleImputer`. It fits on training data and applies the same fill values to test data — preventing data leakage from the test set:

```python
from sklearn.impute import SimpleImputer

# Separate features
X_numeric = df[["age", "fare", "sibsp", "parch"]]

# Fit on training data only
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_numeric)

# The same median values are used when transforming new/test data:
# imputer.transform(X_test_numeric)
```

This fit-then-transform pattern is important: imputation values must come from the training set only. Computing the mean on the full dataset (including the test set) would constitute data leakage.

## Step 4: Fix Data Types

ML algorithms require numeric input. Check for type mismatches:

```python
print(df.dtypes)
```

Output:
```
survived      int64
pclass        int64
sex          object    ← needs encoding
age         float64
sibsp         int64
parch         int64
fare        float64
embarked     object    ← needs encoding
```

`sex` and `embarked` are strings. You'll encode them in the next lesson (Feature Engineering). For now, verify that numeric columns are actually numeric:

```python
# Convert a column that was accidentally loaded as object
df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
# errors="coerce" replaces unparseable values with NaN
```

## Step 5: Validate Value Ranges

Check for impossible or implausible values:

```python
# Are there negative fares?
print(df[df["fare"] < 0])

# Are there passengers older than 120 or younger than 0?
print(df[(df["age"] < 0) | (df["age"] > 120)])

# Check pclass is only 1, 2, 3
print(df["pclass"].unique())
```

Cap or remove values that are clearly errors:

```python
# Cap age at 100 (implausible values become 100)
df["age"] = df["age"].clip(upper=100)

# Remove rows with fare == 0 (might be data errors)
df = df[df["fare"] > 0]
```

## Step 6: Standardize Inconsistent Strings

String inconsistencies create phantom categories:

```python
# Example: a messy categorical column
df["embarked"] = df["embarked"].str.strip().str.upper()

# Check category counts after standardization
print(df["embarked"].value_counts())
```

In a real dataset, you might find values like `"New York"`, `"new york"`, `"NEW YORK"`, and `"New York City"` all meaning the same thing. Consistent formatting ensures they map to a single category.

## Assembling a Clean DataFrame

After all cleaning steps, confirm the result:

```python
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
```

A clean DataFrame ready for ML should have:
- No `NaN` values
- Correct data types for all columns
- No duplicate rows
- No impossible values
- No leaking columns

## The Cleaning Checklist

| Step | Action | Pandas Method |
|------|--------|---------------|
| Inspect | Check shape, types, missing counts | `df.info()`, `df.isnull().sum()` |
| Drop irrelevant | Remove useless or leaking columns | `df.drop(columns=[...])` |
| Duplicates | Remove identical rows | `df.drop_duplicates()` |
| Missing (numeric) | Fill with median/mean | `df.fillna()`, `SimpleImputer` |
| Missing (categorical) | Fill with mode or "Unknown" | `df.fillna()` |
| Data types | Convert to numeric | `pd.to_numeric()`, `astype()` |
| Value ranges | Cap or remove impossible values | `df.clip()`, boolean filtering |
| String consistency | Strip, lowercase, standardize | `.str.strip()`, `.str.lower()` |

## Conclusion

In this lesson, you applied a systematic cleaning checklist to the Titanic dataset — dropping irrelevant and leaking columns, removing duplicates, imputing missing values with median and mode strategies, fixing data types, validating value ranges, and standardizing string columns. You also learned the important concept of the fit-then-transform pattern in `SimpleImputer`, which prevents test set data from influencing imputation values. A clean dataset is a prerequisite for every model you'll build. In the next lesson, you'll go further and **engineer features** — transforming and encoding the clean data into the numeric representation that ML algorithms require.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: You are preparing a dataset to predict customer churn. The dataset includes a column `cancellation_date` that contains the date a customer cancelled their subscription, and is `NaN` for active customers. Should you include this feature? Why or why not?**
1. Yes — dates are always useful features for ML models.
2. Yes — after imputing the `NaN` values with the median date.
3. No — this is data leakage. At prediction time (when a customer is still active), this value would not exist, so including it gives the model information it won't have in production.
4. No — date columns cannot be used in ML models.

**Correct Answer:**
3. No — this is data leakage. At prediction time (when a customer is still active), this value would not exist, so including it gives the model information it won't have in production.

**Explanation:**
Data leakage occurs when training features contain information that would not be available when making real predictions. `cancellation_date` is only filled in for customers who have already churned — it directly encodes the answer. Including it would make the model appear accurate during training but fail completely in production where the column is always empty for the customers you're trying to score.

---

#### **Question 2: A numeric column has 40% missing values and is moderately right-skewed. What is the best imputation strategy?**
1. Drop the column — 40% missing is too high to impute reliably.
2. Impute with the mean — it's the most common strategy.
3. Impute with the median — it is robust to the right skew and extreme values that would distort the mean.
4. Impute with 0 — it preserves the structure of the dataset.

**Correct Answer:**
3. Impute with the median — it is robust to the right skew and extreme values that would distort the mean.

**Explanation:**
For skewed distributions, the mean is pulled toward the extreme values (the tail), making it a poor representation of the "typical" value. The median is the midpoint of the actual values and is unaffected by extreme observations. While 40% missingness is high and should be noted, imputing with the median is the most defensible strategy for a skewed column. Whether to drop the column depends on its predictive importance — worth investigating before dropping.

---

#### **Question 3: Why must `SimpleImputer` be fitted on training data only, and then used to transform both training and test data — rather than being fitted on the entire dataset?**
1. `SimpleImputer` is too slow to process the entire dataset at once.
2. Fitting on the full dataset would expose the model to test set statistics during training, constituting data leakage that inflates performance estimates.
3. scikit-learn's API requires separate `fit` and `transform` calls for technical reasons unrelated to statistics.
4. Test data always has different missing value patterns, so it cannot be included in fitting.

**Correct Answer:**
2. Fitting on the full dataset would expose the model to test set statistics during training, constituting data leakage that inflates performance estimates.

**Explanation:**
The imputation value (e.g., median age) computed on the full dataset is influenced by the test set. In production, you'll only have training data when deciding how to impute — not future data. Fitting only on the training set and applying those same statistics to the test set simulates the real production scenario accurately. This principle applies to all preprocessing steps: scalers, encoders, and imputers must all be fit on training data only.
