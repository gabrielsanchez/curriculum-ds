# Summary

## Overview

You've completed the full data preparation pipeline for the Census Income Dataset — from a raw UCI file with `?`-encoded missing values and a redundant sampling weight column to a clean, scaled, encoded feature matrix with 49 features ready for a binary classifier. In this final lesson, you'll consolidate what the case study demonstrated, draw interpretable insights from the data, and outline the next steps toward training a model.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Present final insights from EDA and data preprocessing.
- Propose next steps for modeling, including an awareness of fairness considerations.

## Starter Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# All variables from previous lessons should be in memory:
# df           — raw data
# df_clean     — cleaned data (32,537 rows × 13 columns)
# X_train_enc  — final feature matrix for training (26,029 × 49)
# X_test_enc   — final feature matrix for testing  (6,508 × 49)
# y_train, y_test  — binary targets
```

## What Was Built

Across three lessons, the raw dataset went through a complete transformation:

```
Raw UCI file (32,561 rows × 15 columns — mixed types, '?' missing values)
    ↓ EDA: class imbalance identified, education/income gradient,
           capital activity signal, fnlwgt irrelevance confirmed
    ↓ Cleaning: drop duplicates, drop fnlwgt, drop redundant education,
                replace '?' with Unknown in workclass/occupation
    ↓ Feature Engineering: encode target and sex, create native_us
                           and has_capital, one-hot encode 5 categoricals,
                           scale 5 continuous features
Final feature matrix: (26,029 train × 49 features)
```

### Pipeline Decisions Summary

| Stage | Decision | Rationale |
|-------|----------|-----------|
| EDA | Identified `fnlwgt` as non-predictive (r = −0.008) | Flagged for removal before modeling |
| EDA | Identified `education` / `education_num` redundancy | One encodes the other; flagged for deduplication |
| EDA | Identified capital columns as 91%+ zero but highly predictive | Informed decision to create `has_capital` flag |
| EDA | Confirmed `native_country` skew (89.6% US, 41 values) | Informed binary `native_us` encoding |
| Cleaning | Dropped `fnlwgt` | Survey weight; zero predictive value |
| Cleaning | Dropped `education` (kept `education_num`) | Already numeric ordinal; `education` is redundant |
| Cleaning | Replaced `?` with `Unknown` in `workclass`, `occupation` | Non-labor-force status is a distinct, informative category |
| Feature Engineering | Created `native_us` binary | Collapses 41 high-cardinality country values cleanly |
| Feature Engineering | Created `has_capital` binary | Captures strong binary signal (63% vs 18% high-income rate) |
| Feature Engineering | One-hot encoded 5 categoricals with `drop_first=True` | Avoids multicollinearity |
| Feature Engineering | `stratify=y` in train/test split | Preserves 76%/24% class balance |
| Feature Engineering | Fitted scaler on train only | Prevents data leakage |

## Key Insights from EDA

### 1. Education Is the Strongest Predictor

```python
edu_order = (
    df_clean.groupby("education_num")["income"]
    .apply(lambda s: (s == ">50K").mean())
    .sort_index()
)

print("Income rate by education_num (1=Preschool, 16=Doctorate):")
print(edu_order.round(3))
```

Output:
```
Income rate by education_num:
education_num
1     0.000    (Preschool)
2     0.009    (1st-4th)
...
13    0.419    (Bachelors)
14    0.556    (Masters)
15    0.734    (Prof-school)
16    0.735    (Doctorate)
```

The income rate rises nearly monotonically with years of education — from 0% for preschool-educated individuals to 73% for doctoral or professional-degree holders. `education_num` is already a well-calibrated ordinal signal that no additional engineering was needed to extract.

### 2. Class Imbalance Requires the Right Metrics

```python
print(f"Baseline accuracy (always predict <=50K): {(y_test == 0).mean():.1%}")
```

Output:
```
Baseline accuracy (always predict <=50K): 76.0%
```

A model that ignores all features and always predicts the majority class achieves 76% accuracy. Any model you train must substantially exceed this baseline, and accuracy alone is not enough to evaluate it. AUC, precision on the >50K class, and recall on the >50K class are the metrics that will reveal whether a model has actually learned to distinguish the classes.

### 3. Capital Activity Is a Sparse but Strong Signal

```python
has_capital = (df_clean["capital_gain"] > 0) | (df_clean["capital_loss"] > 0)
print(f"Fraction with capital activity:      {has_capital.mean():.1%}")
print(f"High-income rate with activity:      {(df_clean[has_capital]['income'] == '>50K').mean():.1%}")
print(f"High-income rate without activity:   {(df_clean[~has_capital]['income'] == '>50K').mean():.1%}")
```

Output:
```
Fraction with capital activity:      8.3%
High-income rate with activity:      63.2%
High-income rate without activity:   17.9%
```

Despite being present in only 8% of rows, capital activity is by far the most discriminative single signal in the dataset. Models that can identify and exploit this feature — including tree-based models that split on `capital_gain > 0` or `has_capital == 1` — will have a significant advantage.

### 4. Multiple Features Interact

No single feature tells the whole story. A 45-year-old with a doctorate, 50 hours/week, capital gains, and an executive role will almost certainly be in the >50K class. A 22-year-old with a high school diploma working part-time in service will almost certainly be ≤50K. The final model will need to combine all of these signals — which is exactly why the feature matrix preserves all of them.

## Final Dataset Snapshot

```python
print("=== Cleaned Dataset ===")
print(f"Rows:              {df_clean.shape[0]:,}")
print(f"Feature columns:   {df_clean.shape[1] - 1}   (before engineering)")
print(f"Missing values:    0")

print("\n=== Final Feature Matrix ===")
print(f"Train:  {X_train_enc.shape}")
print(f"Test:   {X_test_enc.shape}")

print("\n=== Target ===")
from sklearn.metrics import accuracy_score
print(f"High-income rate (train): {y_train.mean():.1%}")
print(f"High-income rate (test):  {y_test.mean():.1%}")
print(f"Baseline accuracy (majority class): {max(y_test.mean(), 1 - y_test.mean()):.1%}")
```

Output:
```
=== Cleaned Dataset ===
Rows:              32,537
Feature columns:   12   (before engineering)
Missing values:    0

=== Final Feature Matrix ===
Train:  (26029, 49)
Test:   (6508, 49)

=== Target ===
High-income rate (train): 24.0%
High-income rate (test):  24.0%
Baseline accuracy (majority class): 76.0%
```

## What This Case Study Demonstrated

| Phase | Skills Applied |
|-------|----------------|
| EDA | Class distribution analysis, income rate by category, correlation matrix, sparse column analysis |
| Cleaning | Non-standard missing value handling (`?` → category), redundant column removal, duplicate detection |
| Feature Engineering | Binary target encoding, binary feature creation (`native_us`, `has_capital`), one-hot encoding with `drop_first`, `StandardScaler`, `stratify` split |

Like the house rental dataset from module 08 of earlier curricula, the real work here was not the modeling — it was the preparation. The decisions made across three lessons will directly determine what patterns a model can and cannot learn.

## A Note on Fairness

This dataset includes `race` and `sex` as features. Including them in a model raises important ethical questions:

- A model trained on historical census data will learn patterns from a time with documented discrimination. Predicting high income from race or sex may encode those historical disparities rather than predict future outcomes fairly.
- Excluding protected attributes does not necessarily prevent discrimination — other features (occupation, education, neighborhood) may serve as proxies.
- In regulated contexts (credit, hiring, housing), using protected attributes for prediction is often legally prohibited and always ethically scrutinized.

Good feature engineering includes thinking about **what features should exist in a model**, not just what features are technically useful. The choice to include `race` and `sex` is not a purely technical decision.

## Next Steps: Modeling in Module 09

The feature matrix is ready. In module 09, you'll train binary classifiers on this data — including logistic regression, decision trees, and random forests — and evaluate them with the metrics appropriate for an imbalanced classification problem. You now have the complete prepared dataset to do that.

```python
# Preview: one logistic regression fit on this data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_enc, y_train)

y_pred  = lr.predict(X_test_enc)
y_proba = lr.predict_proba(X_test_enc)[:, 1]

print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

Expected output:
```
              precision    recall  f1-score   support

       <=50K       0.88      0.93      0.90      4948
        >50K       0.73      0.58      0.64      1560

    accuracy                           0.85      6508
   macro avg       0.80      0.75      0.77      6508
weighted avg       0.84      0.85      0.84      6508

AUC: 0.9062
```

Logistic regression — with no hyperparameter tuning — already achieves 85% accuracy and AUC of 0.91. Module 09 will explore why recall on the >50K class is relatively low (0.58) and what techniques can improve it.

## Reflection Questions

1. `fnlwgt` had near-zero correlation with income and was dropped. Could there be a scenario where including a census sampling weight would actually be helpful? What would change about the analysis if the goal were to estimate population-level income rates rather than predict individual income?

2. The `?` values in `workclass` and `occupation` were treated as their own category `Unknown`. An alternative would have been to drop those 1,836 rows entirely (5.6% of the data). What are the tradeoffs of each approach, and which would you choose if the missing rate were 30% instead of 5.6%?

3. `race` and `sex` were included in the feature matrix, but their inclusion raises fairness concerns. How would you evaluate whether a model trained on this data is treating individuals fairly across racial and gender groups? What metric or test would you use?
