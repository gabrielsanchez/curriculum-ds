# Introduction to the Case Study

## Overview

You've spent the last two modules building the foundational skills of the data science pipeline: sourcing and evaluating data, cleaning it systematically, and engineering features that algorithms can use. Now you'll apply all of those skills together on a real-world dataset — without a guided checklist to follow. This case study walks you through the complete data preparation workflow on the **1994 US Census Income Dataset**, a benchmark dataset from the UCI Machine Learning Repository. By the end of the module, you'll have a clean, engineered feature matrix ready for a binary classifier.

## Learning Objectives

By the end of this case study, you will have learned how to:

- Understand the context and goals of the census income dataset.
- Outline the questions to be answered and what the data will need before a model can use it.

## What Is a Case Study?

The previous modules taught individual skills in isolation — one lesson on missing values, one on encoding, one on scaling. Real data science work doesn't look like that. A case study gives you an end-to-end problem: messy raw data and a business question. Your job is to figure out what the data needs, in what order, using the skills you've built.

This module is structured as a four-step walkthrough:

1. **Exploratory Data Analysis** — Understand the data's shape, distributions, and relationships before touching anything.
2. **Data Cleaning** — Fix quality issues discovered during EDA: handle missing values encoded as `?`, remove redundant and non-predictive columns, and drop duplicate rows.
3. **Feature Engineering** — Encode categorical variables, scale numerics, create useful derived features, and produce a final feature matrix ready for modeling.
4. **Summary** — Synthesize findings, assess the final dataset, and define the next steps toward a classifier.

## The Dataset

The **Census Income Dataset** (also known as the Adult dataset) was extracted from the 1994 US Census Bureau database by Barry Becker and donated to the UCI Machine Learning Repository. Each row describes one person from the census survey.

**Download:** The dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). The raw data file is loaded directly via URL in the starter code — no manual download required.

### Columns

| Column | Description | Type |
|--------|-------------|------|
| `age` | Age in years | Integer |
| `workclass` | Employment sector | Categorical — 8 levels + `?` |
| `fnlwgt` | Census sampling weight | Integer |
| `education` | Highest level of education attained | Categorical — 16 levels |
| `education_num` | Education encoded as years (same information, numeric) | Integer 1–16 |
| `marital_status` | Marital status | Categorical — 7 levels |
| `occupation` | Type of work | Categorical — 14 levels + `?` |
| `relationship` | Relationship within household | Categorical — 6 levels |
| `race` | Race | Categorical — 5 levels |
| `sex` | Sex | Male or Female |
| `capital_gain` | Capital gains from investments (USD) | Integer — mostly 0 |
| `capital_loss` | Capital losses from investments (USD) | Integer — mostly 0 |
| `hours_per_week` | Hours worked per week | Integer |
| `native_country` | Country of origin | Categorical — 41 countries + `?` |
| `income` | **Target**: annual income bracket | `<=50K` or `>50K` |

**Workclass values:** Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked

**Education levels (ordered by years):** Preschool, 1st-4th, 5th-6th, 7th-8th, 9th, 10th, 11th, 12th, HS-grad, Some-college, Assoc-voc, Assoc-acdm, Bachelors, Masters, Prof-school, Doctorate

### The Business Question

> **Can we predict whether a person earns more than $50,000 per year from demographic and employment information alone?**

This kind of model has applications in insurance underwriting, loan eligibility screening, and economic research. It is also a dataset frequently studied for **algorithmic fairness** — because features like race and sex are included, any model trained on this data must be examined carefully for discriminatory patterns.

## Starter Code

Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/08-census-income/01_census-income-case-study_starter.ipynb) to follow along with the case study. The notebook contains all code blocks from all four lessons.

## Loading the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=columns, sep=", ", engine="python")

print(f"Shape: {df.shape}")
print(f"\nData types:")
print(df.dtypes)
```

Output:
```
Shape: (32561, 15)

Data types:
age               int64
workclass        object
fnlwgt            int64
education        object
education_num     int64
marital_status   object
occupation       object
relationship     object
race             object
sex              object
capital_gain      int64
capital_loss      int64
hours_per_week    int64
native_country   object
income           object
dtype: object
```

32,561 individuals across 15 columns. Six numeric columns and nine string columns. No header row in the raw file — the column names were added manually from the dataset documentation.

## Initial Impressions

### Class Distribution

```python
print("Income class distribution:")
print(df["income"].value_counts())
print(f"\nHigh-income fraction: {(df['income'] == '>50K').mean():.1%}")
```

Output:
```
Income class distribution:
<=50K    24720
>50K      7841
Name: income, dtype: int64

High-income fraction: 24.1%
```

76% of the sample earns ≤$50K; only 24% earns more. This is a moderately imbalanced dataset. A trivial classifier that always predicts "≤$50K" would achieve 76% accuracy — which is why accuracy alone is a misleading metric here.

### Missing Values

```python
print(f"Standard NaN values: {df.isnull().sum().sum()}")

print("\nCount of '?' per column:")
print((df == "?").sum()[(df == "?").sum() > 0])
```

Output:
```
Standard NaN values: 0

Count of '?' per column:
workclass         1836
occupation        1843
native_country     583
```

No standard `NaN` values — but three columns use `?` to indicate missing data. `workclass` and `occupation` have nearly the same count of missing values, which makes sense: people not in the labor force would have neither an employment sector nor an occupation recorded.

### Numeric Summary

```python
df.describe()
```

Output:
```
              age        fnlwgt  education_num  capital_gain  capital_loss  hours_per_week
count   32561.000     32561.000      32561.000     32561.000     32561.000       32561.000
mean       38.582    189778.367         10.081      1077.649        87.304          40.438
std        13.641    105549.977          2.573      7385.292       402.960          12.347
min        17.000     12285.000          1.000         0.000         0.000           1.000
25%        28.000    117827.000          9.000         0.000         0.000          40.000
50%        37.000    178356.000         10.000         0.000         0.000          40.000
75%        48.000    237051.000         12.000         0.000         0.000          45.000
max        90.000   1484705.000         16.000     99999.000      4356.000          99.000
```

Three observations stand out:

- **`capital_gain` and `capital_loss`**: Median is 0 for both. Most people report no investment activity — these are sparse signals concentrated in a small minority of records.
- **`fnlwgt`**: Ranges from 12,285 to 1,484,705. This is a **census sampling weight** representing how many people in the US population each individual corresponds to. It is a survey methodology artifact, not a predictive feature, and will be dropped.
- **`education` and `education_num`**: Both encode the same information — one as a string label, one as an integer (1–16 representing years of schooling). Keeping both introduces redundancy into any model.

## What's Next

The initial scan surfaces four immediate tasks:

1. **Handle the `?` missingness** — in `workclass`, `occupation`, and `native_country`
2. **Drop `fnlwgt`** — a sampling weight, not a predictive feature
3. **Resolve the `education` / `education_num` redundancy** — one encodes the other
4. **Decide how to handle `native_country`** — 41 unique values is too many for straightforward one-hot encoding

In the next lesson, you'll dig deeper with EDA before making any of these changes — examining which features most strongly predict income level and building the intuition that will guide every downstream decision.
