# Exploratory Data Analysis

## Overview

In the previous lesson, you loaded the Census Income Dataset and found 32,561 individuals across 15 columns, a 76%/24% class imbalance, and `?` missing values in three columns. Now comes the most important step before any cleaning or modeling: **exploratory data analysis**. EDA means looking at the data carefully and with curiosity. You're trying to understand distributions, spot anomalies, identify relationships, and surface the questions that will guide everything that follows.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Examine distributions and relationships in a dataset with mixed numeric and categorical features.
- Identify which features are most associated with the target variable.

## Starter Code

Use the included [*Colaboratory notebook*](#) to run the code as you read through this lesson.

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

# Add a numeric label for easier analysis
df["label"] = (df["income"] == ">50K").astype(int)   # 1 = high income, 0 = low income
```

## Step 1: Age Distribution

Age is often one of the strongest predictors in census-based models:

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df["age"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Age Distribution (All)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

axes[1].hist(df[df["label"] == 0]["age"], bins=40, alpha=0.6,
             color="#2196f3", label="<=50K")
axes[1].hist(df[df["label"] == 1]["age"], bins=40, alpha=0.6,
             color="#e74c3c", label=">50K")
axes[1].set_title("Age Distribution by Income")
axes[1].set_xlabel("Age")
axes[1].legend()

plt.tight_layout()
plt.show()

print(df.groupby("income")["age"].agg(["mean", "median"]).round(1))
```

Output:
```
         mean  median
income
<=50K    36.8    34.0
>50K     44.3    44.0
```

High-income individuals are substantially older on average. The median age for the >50K group (44) is a full decade older than the ≤50K group (34). This makes intuitive sense: income accumulates with career progression. Age will be an important feature.

## Step 2: Income Rate by Education

```python
# Income rate per education level, sorted by education_num
edu_order = (
    df.groupby("education")["education_num"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

edu_rates = (
    df.groupby("education")["label"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "income_rate", "count": "n"})
    .loc[edu_order]
    .round(3)
)

print(edu_rates.to_string())
```

Output:
```
                 income_rate     n
education
Preschool              0.000    51
1st-4th                0.009   168
5th-6th                0.009   333
7th-8th                0.019   646
9th                    0.027   514
10th                   0.013   933
11th                   0.029  1175
12th                   0.037   433
HS-grad                0.157  9840
Some-college           0.198  7291
Assoc-voc              0.257  1382
Assoc-acdm             0.254  1067
Bachelors              0.419  5355
Masters                0.556  1723
Prof-school            0.734   576
Doctorate              0.735   413
```

Education has an extraordinarily clear gradient: from 0% high-income among preschool-educated individuals to 73% among those with doctoral or professional degrees. This monotonic relationship across 16 levels is why `education_num` (the numeric encoding) works well as a feature.

```python
plt.figure(figsize=(11, 5))
income_rates = edu_rates["income_rate"]
colors = ["#e74c3c" if r > 0.5 else "#2196f3" for r in income_rates]
plt.bar(range(len(edu_order)), income_rates, color=colors, tick_label=edu_order)
plt.xticks(rotation=45, ha="right")
plt.title("Income Rate (>$50K) by Education Level")
plt.ylabel("Fraction earning >$50K")
plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
```

## Step 3: Income Rate by Occupation

```python
occ_rates = (
    df[df["occupation"] != "?"]
    .groupby("occupation")["label"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "income_rate", "count": "n"})
    .sort_values("income_rate", ascending=False)
    .round(3)
)

print(occ_rates.to_string())
```

Output:
```
                    income_rate     n
occupation
Exec-managerial           0.470  3992
Prof-specialty            0.445  4140
Protective-serv           0.337   649
Tech-support              0.303   928
Sales                     0.268  3650
Craft-repair              0.221  4099
Transport-moving          0.178  1597
Machine-op-inspct         0.126  2002
Adm-clerical              0.120  3721
Farming-fishing           0.101   994
Other-service             0.041  3295
Handlers-cleaners         0.041  1370
Priv-house-serv           0.006   149
Armed-Forces              0.000     9
```

Executive-managerial and professional-specialty roles are the highest-earning occupations (47% and 44.5% high income respectively), while service and cleaning roles are the lowest. This strong signal is expected — occupation and income are structurally linked.

## Step 4: Hours Worked Per Week

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Distribution by class
axes[0].hist(df[df["label"] == 0]["hours_per_week"], bins=40, alpha=0.6,
             color="#2196f3", label="<=50K")
axes[0].hist(df[df["label"] == 1]["hours_per_week"], bins=40, alpha=0.6,
             color="#e74c3c", label=">50K")
axes[0].set_title("Hours/Week by Income")
axes[0].set_xlabel("Hours per week")
axes[0].legend()

# Box plot
axes[1].boxplot(
    [df[df["label"] == 0]["hours_per_week"],
     df[df["label"] == 1]["hours_per_week"]],
    labels=["<=50K", ">50K"],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
)
axes[1].set_title("Hours/Week by Income (Box Plot)")
axes[1].set_ylabel("Hours per week")

plt.tight_layout()
plt.show()

print(df.groupby("income")["hours_per_week"].agg(["mean", "median"]).round(1))
```

Output:
```
         mean  median
income
<=50K    38.8    40.0
>50K     45.4    45.0
```

High-income individuals work more hours on average (45.4 vs. 38.8), and the median is 45 vs. 40. The distribution for >50K is shifted right, with more people working more than 45 hours per week. Note that both distributions spike at 40 — a standard full-time work week — but the >50K group has a heavier tail beyond 40 hours.

## Step 5: The Capital Gain and Capital Loss Columns

```python
print("Capital gain distribution:")
print(f"  Fraction with zero: {(df['capital_gain'] == 0).mean():.1%}")
print(f"  Among non-zero — min: {df[df['capital_gain'] > 0]['capital_gain'].min()}")
print(f"  Among non-zero — median: {df[df['capital_gain'] > 0]['capital_gain'].median():.0f}")
print(f"  Among non-zero — max: {df['capital_gain'].max()}")
print()
print("Capital loss distribution:")
print(f"  Fraction with zero: {(df['capital_loss'] == 0).mean():.1%}")
print(f"  Among non-zero — max: {df['capital_loss'].max()}")
```

Output:
```
Capital gain distribution:
  Fraction with zero: 91.7%
  Among non-zero — min: 114
  Among non-zero — median: 5178
  Among non-zero — max: 99999

Capital loss distribution:
  Fraction with zero: 95.3%
  Among non-zero — max: 4356
```

Over 91% of people report zero capital gains and 95% report zero capital losses. When a feature is zero for the vast majority of the dataset, it is sometimes more informative to encode it as a binary flag (has investment activity or not) rather than as a raw number. Both representations will be considered during feature engineering.

```python
# Income rate for those with vs without capital activity
has_cap = (df["capital_gain"] > 0) | (df["capital_loss"] > 0)
print(f"Income rate — with capital activity:     {df[has_cap]['label'].mean():.1%}")
print(f"Income rate — without capital activity:  {df[~has_cap]['label'].mean():.1%}")
```

Output:
```
Income rate — with capital activity:     63.2%
Income rate — without capital activity:  17.9%
```

Despite being sparse, capital activity is a very strong signal: 63% of people with any capital activity earn >$50K, versus only 18% of those without. This is worth preserving, even though most values are zero.

## Step 6: Income Rate by Workclass

```python
wc_rates = (
    df[df["workclass"] != "?"]
    .groupby("workclass")["label"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "income_rate", "count": "n"})
    .sort_values("income_rate", ascending=False)
    .round(3)
)

print(wc_rates.to_string())
```

Output:
```
                      income_rate     n
workclass
Self-emp-inc                0.557   622
Federal-gov                 0.398   960
Local-gov                   0.293  2093
Self-emp-not-inc            0.285  2541
State-gov                   0.270  1298
Private                     0.218 22696
Without-pay                 0.000    14
Never-worked                0.000     7
```

Self-employed incorporated businesses have the highest high-income rate (55.7%), followed by federal government workers (39.8%). Private sector workers (the largest group at 22,696) have a lower rate of 21.8%. Workers with no pay or no work history report 0% high-income rate (though the sample sizes are very small).

## Step 7: Correlation Matrix

For the numeric features, compute correlations with the binary income label:

```python
numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week", "label"]

plt.figure(figsize=(7, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True, fmt=".2f", cmap="coolwarm",
    vmin=-1, vmax=1, linewidths=0.5,
)
plt.title("Correlation Matrix — Numeric Features")
plt.tight_layout()
plt.show()

print("\nCorrelation with income (label):")
print(df[numeric_cols].corr()["label"].drop("label").sort_values(ascending=False).round(3))
```

Output:
```
Correlation with income (label):
education_num    0.339
age              0.242
hours_per_week   0.229
capital_gain     0.223
capital_loss     0.150
fnlwgt          -0.008
dtype: float64
```

`education_num` is the strongest numeric predictor (r = 0.339), followed by age (0.242), hours per week (0.229), and capital gain (0.223). `fnlwgt` has essentially zero correlation with income (r = -0.008), further confirming it should be dropped.

## Step 8: Native Country

```python
print(f"Unique native countries: {df['native_country'].nunique()}")
print(f"\nTop 5 most common:")
print(df["native_country"].value_counts().head())
print(f"\nUnited-States fraction: {(df['native_country'] == 'United-States').mean():.1%}")
```

Output:
```
Unique native countries: 41

Top 5 most common:
United-States    29170
Mexico             643
?                  583
Philippines        198
Germany            137
Name: native_country, dtype: int64

United-States fraction: 89.6%
```

89.6% of the sample is from the United States. With 41 unique values and extreme skew toward a single category, one-hot encoding `native_country` directly would produce 40+ mostly-empty columns. A more practical approach is to create a binary flag: `native_us = 1` if United-States, `0` otherwise.

## EDA Summary

| Finding | Implication |
|---------|-------------|
| 76%/24% class split | Accuracy is misleading; evaluate with precision, recall, and AUC |
| Age strongly predicts income (median gap of 10 years) | Keep as a scaled numeric feature |
| Education has a near-monotonic relationship with income rate | `education_num` (numeric) is a strong feature; `education` (string) is redundant |
| Occupation and workclass have large income-rate differences | Encode these categories carefully |
| Capital gain/loss: >90% zero, but highly predictive when non-zero | Consider a binary flag `has_capital` |
| `fnlwgt` has near-zero correlation with income (r = -0.008) | Drop unconditionally |
| `native_country`: 89.6% United States, 41 unique values | Create a `native_us` binary flag |
| Hours/week is right-shifted for high-income individuals | Keep as scaled numeric feature |

## What's Next

In the next lesson, you'll **clean** the dataset: handle the `?` values in `workclass`, `occupation`, and `native_country`, drop `fnlwgt`, remove duplicate rows, and resolve the `education`/`education_num` redundancy. Every decision will be grounded in what you observed here.
