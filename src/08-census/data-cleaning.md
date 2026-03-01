# Data Cleaning

## Overview

In the previous lesson, EDA established the cleaning agenda for the Census Income Dataset: `?` placeholder values in three columns, a sampling weight column that provides no predictive signal, a redundant education column, a high-cardinality country column, and a potential for duplicate rows. In this lesson, you'll fix each issue systematically — always explaining why before changing anything. Every cleaning decision should be traceable back to something observed in EDA.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Handle missing values encoded as non-standard placeholder strings.
- Remove columns that are redundant or irrelevant to the prediction task.

## Starter Code

```python
import pandas as pd
import numpy as np

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=columns, sep=", ", engine="python")
print(f"Starting shape: {df.shape}")
```

Always work on a copy to preserve the raw data:

```python
df_clean = df.copy()
```

## Step 1: Check for Duplicate Rows

```python
n_dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {n_dupes}")
```

Output:
```
Duplicate rows: 24
```

24 duplicates — a small number but still worth removing. In a census-style dataset, genuine duplicate rows almost certainly represent data entry errors rather than two different people with identical values across all 15 columns.

```python
df_clean = df_clean.drop_duplicates()
print(f"Shape after deduplication: {df_clean.shape}")
```

Output:
```
Shape after deduplication: (32537, 15)
```

## Step 2: Drop fnlwgt

EDA confirmed that `fnlwgt` has near-zero correlation with income (r = -0.008). It is a census sampling weight — a methodology artifact that tells the survey how to weight the sample toward the full US population. It carries no information about an individual's income.

```python
print(f"fnlwgt stats: min={df_clean['fnlwgt'].min():,}, max={df_clean['fnlwgt'].max():,}")
df_clean = df_clean.drop(columns=["fnlwgt"])
print(f"Shape after dropping fnlwgt: {df_clean.shape}")
```

Output:
```
fnlwgt stats: min=12,285, max=1,484,705
Shape after dropping fnlwgt: (32537, 14)
```

## Step 3: Resolve the education / education_num Redundancy

```python
# Confirm they encode the same information
edu_mapping = df_clean.groupby("education")["education_num"].agg(["min", "max"])
print(edu_mapping.sort_values("min"))
```

Output:
```
                  min  max
education
Preschool           1    1
1st-4th             2    2
5th-6th             3    3
7th-8th             4    4
9th                 5    5
10th                6    6
11th                7    7
12th                8    8
HS-grad             9    9
Some-college       10   10
Assoc-voc          11   11
Assoc-acdm         12   12
Bachelors          13   13
Masters            14   14
Prof-school        15   15
Doctorate          16   16
```

Every education level maps to exactly one `education_num` value (min = max for all). They are perfectly redundant. `education_num` is already a numeric ordinal encoding of `education` — it's exactly what you would create manually in feature engineering. Keep `education_num`, drop the string `education` column.

```python
df_clean = df_clean.drop(columns=["education"])
print(f"Shape after dropping education: {df_clean.shape}")
```

Output:
```
Shape after dropping education: (32537, 13)
```

## Step 4: Handle ? in workclass

```python
print(f"workclass '?' count: {(df_clean['workclass'] == '?').sum()}")
print(f"workclass values: {sorted(df_clean['workclass'].unique())}")
```

Output:
```
workclass '?' count: 1836
workclass values: ['?', 'Federal-gov', 'Local-gov', 'Never-worked',
                   'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                   'State-gov', 'Without-pay']
```

1,836 rows (5.6%) have `?` in `workclass`. From EDA, these individuals tend to be not in the labor force — they have neither an employer category nor an occupation. Three strategies:

| Strategy | Tradeoff |
|----------|---------|
| **Drop rows** | Lose 5.6% of data; biases remaining set |
| **Impute with mode** | 'Private' is most common — but these are not private workers, they're non-workers. Misleading. |
| **Keep `?` as its own category** | Preserves all data; treats "not in labor force" as a distinct status |

The third option is correct here. "Not in labor force" is a meaningful employment status, not a gap in the data — the census respondent answered; the answer is "none of the above." Treat it as its own category called `Unknown`.

```python
df_clean["workclass"] = df_clean["workclass"].replace("?", "Unknown")
print(f"workclass after replacement: {sorted(df_clean['workclass'].unique())}")
```

Output:
```
workclass after replacement: ['Federal-gov', 'Local-gov', 'Never-worked',
                               'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                               'State-gov', 'Unknown', 'Without-pay']
```

## Step 5: Handle ? in occupation

```python
print(f"occupation '?' count: {(df_clean['occupation'] == '?').sum()}")
```

Output:
```
occupation '?' count: 1843
```

The count is nearly identical to `workclass` — the same people who have no workclass also have no occupation. This validates the interpretation: they are not in the labor force.

```python
df_clean["occupation"] = df_clean["occupation"].replace("?", "Unknown")
print("occupation unique values after replacement:")
print(sorted(df_clean["occupation"].unique()))
```

Output:
```
occupation unique values after replacement:
['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
 'Sales', 'Tech-support', 'Transport-moving', 'Unknown']
```

## Step 6: Handle ? in native_country

```python
print(f"native_country '?' count: {(df_clean['native_country'] == '?').sum()}")
print(f"\nTop 5 native_country values:")
print(df_clean["native_country"].value_counts().head())
```

Output:
```
native_country '?' count: 583

Top 5 native_country values:
United-States    29170
Mexico             634
?                  583
Philippines        198
Germany            137
```

For `native_country`, you will handle the `?` at the feature engineering step — when the column is converted to a binary `native_us` flag (`1` if United-States, `0` otherwise). In that encoding, `?` becomes `0` automatically, which is correct (unknown country of origin is treated as non-US). For now, leave `native_country` unchanged.

## Step 7: Verify No Remaining ? Values in Critical Columns

```python
remaining_q = (df_clean[["workclass", "occupation"]] == "?").sum()
print(f"Remaining '?' in workclass:  {remaining_q['workclass']}")
print(f"Remaining '?' in occupation: {remaining_q['occupation']}")
```

Output:
```
Remaining '?' in workclass:  0
Remaining '?' in occupation: 0
```

## Step 8: Final Validation

```python
print("=== Cleaned Dataset ===")
print(f"Shape:                    {df_clean.shape}")
print(f"Columns:                  {list(df_clean.columns)}")
print(f"Standard NaN values:      {df_clean.isnull().sum().sum()}")
print(f"Remaining '?' anywhere:   {(df_clean == '?').sum().sum()}")
print()
print("Target column distribution:")
print(df_clean["income"].value_counts())
```

Output:
```
=== Cleaned Dataset ===
Shape:                    (32537, 13)
Columns:                  ['age', 'workclass', 'education_num', 'marital_status',
                           'occupation', 'relationship', 'race', 'sex',
                           'capital_gain', 'capital_loss', 'hours_per_week',
                           'native_country', 'income']
Standard NaN values:      0
Remaining '?' anywhere:   583

Target column distribution:
<=50K    24720
>50K      7817
Name: income, dtype: int64
```

The 583 remaining `?` values are all in `native_country`, which will be handled during feature engineering. The dataset went from (32561, 15) to (32537, 13) — 24 duplicate rows removed, 2 columns dropped (`fnlwgt`, `education`).

## Cleaning Summary

| Step | Change | Rationale |
|------|--------|-----------|
| Remove duplicates | −24 rows | 24 identical rows, almost certainly data entry errors |
| Drop `fnlwgt` | −1 column | Census sampling weight; r = −0.008 with income |
| Drop `education` | −1 column | Perfectly redundant with `education_num` (already numeric ordinal) |
| Replace `?` with `Unknown` in `workclass` | 1,836 values changed | Non-labor-force status is a meaningful category |
| Replace `?` with `Unknown` in `occupation` | 1,843 values changed | Same rationale; same individuals |
| Leave `native_country` `?` untouched | — | Will become `0` in the `native_us` binary feature during engineering |

## What's Next

In the next lesson, you'll **engineer features**: encode the binary target, handle the capital columns, create `native_us`, one-hot encode the categorical features, scale the numeric columns, and produce the final feature matrix for a classifier.
