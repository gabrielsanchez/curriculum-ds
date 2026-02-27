# Introduction to Feature Engineering

## Overview

In the previous lesson, you cleaned the raw dataset by removing errors, imputing missing values, and standardizing formatting. Now you'll take the clean data one step further: **feature engineering** — transforming and creating variables so that machine learning algorithms can actually use them. Most ML algorithms require purely numeric input, and even among numeric features, the scale and distribution of values affects model performance. In this lesson, you'll learn how to encode categoricals, scale numeric features, create new derived features, and understand why feature quality drives model quality.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Create and transform features, including encoding categorical variables.
- Recognize the impact of feature quality on model performance.

## Key terms

**Feature engineering:** The process of transforming raw variables into representations that better capture the underlying signal for a machine learning model.

**Encoding:** Converting categorical (text) values into numeric representations that algorithms can process.

**Label encoding:** Assigning each unique category an integer (e.g., `male → 0`, `female → 1`). Suitable for ordinal categories or tree-based models.

**One-hot encoding:** Creating a new binary column for each unique category. Suitable for nominal (unordered) categories.

**`OrdinalEncoder`:** A scikit-learn class for label encoding.

**`OneHotEncoder`:** A scikit-learn class for one-hot encoding.

**Feature scaling:** Transforming numeric features to a common scale so that large-valued columns don't dominate distance-based or gradient-based algorithms.

**StandardScaler:** Scales features to have mean = 0 and standard deviation = 1 (z-score normalization).

**MinMaxScaler:** Scales features to a fixed range, typically [0, 1].

**Feature creation:** Deriving new columns from existing ones — for example, combining `sibsp` and `parch` into a `family_size` variable.

**`ColumnTransformer`:** A scikit-learn utility that applies different preprocessing steps to different columns simultaneously.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/07-intro-ml/05_feature-engineering_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Even after cleaning, a DataFrame isn't ready for most ML algorithms. Two types of problems remain:

1. **Categorical columns** contain strings — algorithms need numbers.
2. **Numeric columns** may be on very different scales — a `fare` of $512 and an `age` of 22 are both valid, but many algorithms treat features with larger numeric values as more important unless they're rescaled.

Feature engineering addresses both issues, and goes further: skilled feature engineers create entirely new variables that capture relationships the raw data doesn't surface directly. The difference between a mediocre model and a great one is often not the algorithm — it's the features fed into it.

## Setup: Cleaned Titanic Data

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load and minimally clean
df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
df["age"]      = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["fare"]     = df["fare"].clip(lower=0.01)
df = df.dropna()

# Split before any feature engineering
X = df.drop(columns="survived")
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

## Encoding Categorical Variables

### Why Encoding Is Necessary

Most algorithms perform arithmetic on inputs (distances, dot products, gradients). The string `"female"` cannot be added or multiplied — it must become a number.

### Label Encoding

Label encoding assigns each unique value an integer. It's fast and compact, but it implies an **order** — which is only appropriate when order actually exists:

```python
from sklearn.preprocessing import OrdinalEncoder

# 'pclass' is already numeric (1, 2, 3 — ordinal)
# For 'sex' (binary): male/female → 0/1 is fine
encoder = OrdinalEncoder()
X_train_encoded = X_train.copy()
X_train_encoded["sex_encoded"] = encoder.fit_transform(X_train[["sex"]])
print(X_train_encoded[["sex", "sex_encoded"]].head())
```

Output:
```
       sex  sex_encoded
0   female          0.0
1     male          1.0
...
```

**Warning:** Don't label-encode nominal categories with more than two values. Encoding `embarked` as `C=0, Q=1, S=2` falsely implies S > Q > C — that Southampton is "more embarked" than Cherbourg, which is meaningless.

### One-Hot Encoding

One-hot encoding creates a new binary column for each category. No false ordering is implied:

```python
# With pandas
embarked_dummies = pd.get_dummies(X_train["embarked"], prefix="embarked", drop_first=True)
print(embarked_dummies.head())
```

Output:
```
   embarked_Q  embarked_S
0           0           1
1           0           0
2           0           1
...
```

`drop_first=True` drops one column to avoid **multicollinearity** (if you know `Q=0` and `S=0`, you know it must be `C=1`, so the third column is redundant).

```python
# With scikit-learn (pipeline-friendly)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop="first", sparse_output=False)
embarked_encoded = ohe.fit_transform(X_train[["embarked"]])
print(ohe.get_feature_names_out())
```

### When to Use Which

| Situation | Recommended encoding |
|-----------|---------------------|
| Binary column (2 values) | Label encoding (0/1) |
| Ordinal column (ordered categories: low/med/high) | Label encoding with correct order |
| Nominal column with few values (3–10) | One-hot encoding |
| Nominal column with many values (10+) | Target encoding or embedding (advanced) |

## Feature Scaling

### Why Scaling Matters

Consider a nearest-neighbor algorithm computing the distance between two passengers:
- Passenger A: age=25, fare=512
- Passenger B: age=30, fare=10

The Euclidean distance is dominated by the `fare` difference (502) and barely affected by the `age` difference (5). This means the algorithm treats `fare` as ~100× more important than `age` — purely because of scale. Scaling removes this artifact.

**Algorithms that require scaling:**
- k-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Linear and Logistic Regression (speeds up convergence)
- Neural Networks
- PCA and other dimensionality reduction

**Algorithms that do NOT require scaling:**
- Decision Trees
- Random Forests
- Gradient Boosting (XGBoost, LightGBM)
- These split on individual feature values and are scale-invariant.

### `StandardScaler` (Z-score Normalization)

Transforms each feature to have mean = 0 and standard deviation = 1:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ["age", "fare", "sibsp", "parch"]

X_train_scaled = X_train.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

print(X_train_scaled[numeric_cols].describe().round(2))
```

Output (mean ≈ 0, std ≈ 1):
```
         age   fare  sibsp  parch
mean    0.00   0.00   0.00   0.00
std     1.00   1.00   1.00   1.00
```

Fit only on training data. Apply the same fitted scaler to the test set:
```python
X_test_scaled = X_test.copy()
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

### `MinMaxScaler`

Scales each feature to the range [0, 1]:

```python
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train[numeric_cols])
```

Use `MinMaxScaler` when you need values in a specific range (e.g., neural network inputs). Use `StandardScaler` as the default for most other cases.

## Creating New Features

New features derived from existing ones can capture patterns the raw variables don't express directly. This is where domain knowledge is most valuable.

### Combining Columns

```python
X_train_fe = X_train.copy()

# Total family members on board
X_train_fe["family_size"] = X_train_fe["sibsp"] + X_train_fe["parch"] + 1

# Is the passenger traveling alone?
X_train_fe["is_alone"] = (X_train_fe["family_size"] == 1).astype(int)

print(X_train_fe[["sibsp", "parch", "family_size", "is_alone"]].head())
```

### Binning Continuous Variables

```python
# Age groups
X_train_fe["age_group"] = pd.cut(
    X_train_fe["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["child", "teen", "young_adult", "adult", "senior"]
)

print(X_train_fe["age_group"].value_counts())
```

### Log Transformation for Skewed Features

```python
import numpy as np

# fare is right-skewed — log transform makes it more symmetric
X_train_fe["log_fare"] = np.log1p(X_train_fe["fare"])
```

### Interaction Features

```python
# Gender × class interaction — historically, women in 1st class had very high survival
X_train_fe["sex_pclass"] = (
    X_train_fe["sex"].map({"female": 0, "male": 1}) * X_train_fe["pclass"]
)
```

## Putting It All Together: `ColumnTransformer`

In a real ML pipeline, you apply different transformations to different columns simultaneously. `ColumnTransformer` organizes this cleanly:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_features     = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num",  numeric_transformer,     numeric_features),
    ("cat",  categorical_transformer, categorical_features)
])

# Fit on training data, transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")
```

This pipeline guarantees that:
- The same imputation values are used for train and test
- The same scaler parameters are applied to both
- The same one-hot categories are encoded consistently

A fitted `preprocessor` can be saved and reused in production with `joblib.dump()`.

## Why Feature Quality Drives Model Quality

To make this concrete, consider two versions of the same Titanic dataset:

| Version | Features used | Typical accuracy |
|---------|---------------|-----------------|
| Raw (no encoding, no scaling) | pclass, age, sibsp, parch, fare | ~65% |
| Cleaned + encoded + scaled | All above + sex_encoded, embarked_OHE | ~78% |
| + Engineered features | + family_size, is_alone, log_fare | ~81% |

The algorithm doesn't change — only the features do. This is why experienced data scientists spend more time on feature engineering than on model selection.

## Conclusion

In this lesson, you learned how to encode categorical variables using label encoding and one-hot encoding, scale numeric features with `StandardScaler` and `MinMaxScaler`, create new features through combination, binning, log transformation, and interactions, and assemble all of these steps cleanly using scikit-learn's `Pipeline` and `ColumnTransformer`. Feature engineering is one of the highest-leverage activities in ML — the same algorithm performs dramatically differently depending on the quality of its inputs. In the next lesson, you'll apply everything from this module in the **module assessment**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/07-intro-ml/05_feature-engineering_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Label encoding is appropriate for ordered categories (e.g., `low=1, medium=2, high=3`). For nominal categories with no natural order like colors, label encoding creates a false numeric relationship — the model may learn that `blue` (3) is arithmetically closer to `green` (2) than to `red` (1), even though the three colors are equally unrelated. One-hot encoding avoids this by representing each color as an independent binary column.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A dataset has a `color` column with values `"red"`, `"green"`, and `"blue"`. A teammate suggests encoding it as `red=1, green=2, blue=3`. What is the problem with this approach?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Scikit-learn does not support label encoding for three or more categories.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The encoding implies an ordering (blue > green > red) that doesn't exist, which may mislead algorithms that treat the column numerically.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>This encoding requires more memory than one-hot encoding.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Colors cannot be used as features in machine learning models.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="KNN computes Euclidean distance between data points. A difference of $10,000 in income contributes far more to the distance than a difference of 10 years in age, simply because of scale. Without normalization, `income` drowns out `age` regardless of its actual predictive importance. Scaling both features to a common range (e.g., 0–1 or z-scores) ensures each feature contributes proportionally.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You are building a k-nearest neighbors model on a dataset where `income` ranges from $20,000 to $200,000 and `age` ranges from 18 to 90. Without scaling, what is the likely effect on the model?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model will perform better because larger values provide more signal.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The model's distance calculations will be dominated by `income`, effectively ignoring `age`, even if `age` is an equally important predictor.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The model will automatically normalize both features during training.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>No effect — KNN is invariant to feature scale.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`fit()` computes the mean and standard deviation from the data. If you fit on the test set (or the combined dataset), the scaler &quot;knows&quot; the test set&#039;s distribution — a form of data leakage that makes performance estimates too optimistic. In production, when a new record comes in, you scale it using the statistics from the training data. The `fit_transform()` on training data and `transform()` on test data pattern correctly simulates this real-world scenario.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Why is it critical to fit the `StandardScaler` on the training data only, and then use `transform()` (not `fit_transform()`) on the test data?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>`transform()` is faster than `fit_transform()` for large datasets.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Fitting on the test set would expose training-time statistics to data the model hasn't seen, and in production you won't have access to future data when scaling new inputs.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>`fit_transform()` changes the data type, which breaks the test set.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>`StandardScaler` can only be fitted once and must be reused for all subsequent data.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

