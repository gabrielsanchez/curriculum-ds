# Exploratory Data Analysis

## Overview

In the previous lesson, you loaded the House Rent Prediction Dataset and got an initial sense of its structure — 4,746 listings, 12 columns, no missing values. Now comes the most important step before any cleaning or modeling: **exploratory data analysis**. EDA means looking at the data carefully and with curiosity. You're trying to understand distributions, spot anomalies, identify relationships, and surface the questions that will guide everything that follows. In this lesson, you'll explore the rental data from multiple angles before touching a single value.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Examine distributions, missing values, and correlations within housing data.
- Generate initial insights about rental prices.

## Starter Code

Use the included [*Colaboratory notebook*](#) to run the code as you read through this lesson.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("House_Rent_Dataset.csv")
```

## Step 1: Summary Statistics

Start with the numbers:

```python
df.describe()
```

Output:
```
              BHK          Rent          Size     Bathroom
count  4746.000000   4746.000000   4746.00000  4746.000000
mean      2.083756  35003.073751    967.49262     1.965865
std       0.832607  67534.480513   1289.54848     0.884532
min       1.000000   1200.000000     10.00000     1.000000
25%       2.000000  10000.000000    550.00000     1.000000
50%       2.000000  16000.000000    800.00000     2.000000
75%       3.000000  33000.000000   1200.00000     2.000000
max       6.000000  3500000.000000  8000.00000     10.000000
```

Observations:
- **Rent**: The mean (₹35,003) is far above the median (₹16,000) — the distribution is heavily right-skewed. The max of ₹3,500,000 is extreme.
- **Size**: Similar pattern. Mean (967 sq ft) > median (800 sq ft), max is 8,000 sq ft.
- **Bathroom**: Max is 10, which warrants investigation relative to BHK counts.
- **BHK**: Ranges from 1 to 6. Most listings are 2 BHK.

## Step 2: Distribution of Rent

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
axes[0].hist(df["Rent"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Rent Distribution (Raw)")
axes[0].set_xlabel("Rent (INR)")
axes[0].set_ylabel("Count")

# Log-transformed
axes[1].hist(np.log1p(df["Rent"]), bins=50, color="steelblue", edgecolor="white")
axes[1].set_title("Rent Distribution (Log-Transformed)")
axes[1].set_xlabel("log(1 + Rent)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
```

The raw distribution is nearly unreadable — a spike near zero with a long tail. The log-transformed version reveals a roughly bell-shaped distribution centered around ₹10,000–₹20,000 (log ≈ 9.2–9.9). This tells you that **log transformation of Rent will be important before modeling**.

## Step 3: Rent by City

```python
city_order = df.groupby("City")["Rent"].median().sort_values(ascending=False).index

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="City", y="Rent", order=city_order, palette="Blues_r")
plt.yscale("log")
plt.title("Rent Distribution by City (Log Scale)")
plt.xlabel("City")
plt.ylabel("Rent (INR, log scale)")
plt.tight_layout()
plt.show()

print(df.groupby("City")["Rent"].median().sort_values(ascending=False))
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

Mumbai commands the highest rents by a wide margin — more than 3× Kolkata's median. City will almost certainly be one of the most predictive features.

## Step 4: Rent by Furnishing Status

```python
furnish_order = df.groupby("Furnishing Status")["Rent"].median().sort_values(ascending=False).index

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Furnishing Status", y="Rent", order=furnish_order, palette="Set2")
plt.yscale("log")
plt.title("Rent by Furnishing Status (Log Scale)")
plt.xlabel("Furnishing Status")
plt.ylabel("Rent (INR, log scale)")
plt.tight_layout()
plt.show()

print(df.groupby("Furnishing Status")["Rent"].median().sort_values(ascending=False))
```

Output:
```
Furnishing Status
Furnished        22000
Semi-Furnished   15000
Unfurnished      10000
Name: Rent, dtype: int64
```

Furnished properties command a 2× premium over unfurnished ones at the median. A clear, interpretable signal.

## Step 5: Rent by BHK Count

```python
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="BHK", y="Rent", palette="Blues")
plt.yscale("log")
plt.title("Rent by BHK Count (Log Scale)")
plt.xlabel("BHK (Bedrooms + Halls + Kitchens)")
plt.ylabel("Rent (INR, log scale)")
plt.tight_layout()
plt.show()

print(df.groupby("BHK")["Rent"].agg(["median", "count"]))
```

Output:
```
     median  count
BHK
1      8000    659
2     15000   2782
3     28000   1089
4     55000    174
5    100000     30
6    107500     12
```

Rent increases with BHK, as expected. The 5 and 6 BHK rows are very sparse — these will need attention during cleaning.

## Step 6: Size Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["Size"], bins=50, color="coral", edgecolor="white")
axes[0].set_title("Size Distribution (Raw)")
axes[0].set_xlabel("Size (sq ft)")

axes[1].hist(np.log1p(df["Size"]), bins=50, color="coral", edgecolor="white")
axes[1].set_title("Size Distribution (Log-Transformed)")
axes[1].set_xlabel("log(1 + Size)")

plt.tight_layout()
plt.show()

# Flag potential outliers
print("Very small sizes (< 100 sq ft):")
print(df[df["Size"] < 100][["BHK", "Rent", "Size", "City"]].head(10))

print(f"\nVery large sizes (> 4000 sq ft):")
print(df[df["Size"] > 4000][["BHK", "Rent", "Size", "City"]].head(10))
```

A 10 sq ft listing for a 2 BHK is almost certainly a data entry error — the unit may have been entered in square meters rather than square feet, or is simply wrong. Sizes above 4,000 sq ft appear to be genuine luxury or commercial properties.

## Step 7: Rent vs. Size Scatter Plot

```python
plt.figure(figsize=(9, 6))
plt.scatter(np.log1p(df["Size"]), np.log1p(df["Rent"]),
            alpha=0.3, s=10, c=df["BHK"], cmap="viridis")
plt.colorbar(label="BHK")
plt.title("log(Size) vs. log(Rent), colored by BHK")
plt.xlabel("log(1 + Size)")
plt.ylabel("log(1 + Rent)")
plt.tight_layout()
plt.show()
```

There is a positive relationship between size and rent, but with considerable scatter — location, furnishing, and BHK count all shift the cloud. A few points appear isolated from the main cluster, hinting at outliers.

## Step 8: Area Type

```python
print(df["Area Type"].value_counts())
print(f"\nMedian rent by area type:")
print(df.groupby("Area Type")["Rent"].median().sort_values(ascending=False))
```

Output:
```
Super Area      2483
Carpet Area     2024
Built Area       239
Name: Area Type, dtype: int64

Area Type
Built Area      22000
Carpet Area     18500
Super Area      14500
Name: Area Type, dtype: int64
```

Built Area and Carpet Area listings tend to be priced higher — possibly because they represent the usable space more accurately, attracting higher-priced listings. Worth keeping as a feature.

## Step 9: Area Locality — High Cardinality

```python
n_localities = df["Area Locality"].nunique()
print(f"Unique localities: {n_localities}")
print(f"\nTop 10 most common localities:")
print(df["Area Locality"].value_counts().head(10))
```

Output:
```
Unique localities: 2235

Top 10 most common localities:
Electronic City Phase II     27
Whitefield                   21
Marathahalli                 19
Indira Nagar                 18
...
```

With 2,235 unique values across 4,746 rows, `Area Locality` has extreme cardinality — most values appear only once or twice. One-hot encoding this column directly would produce thousands of near-empty columns. You'll need to make a decision about this feature during engineering (drop it, or group into city-level features).

## Step 10: The Floor Column

```python
print("Sample Floor values:")
print(df["Floor"].value_counts().head(15))
```

Output:
```
Ground out of 2       262
1 out of 3            148
Ground out of 3       143
2 out of 3            137
1 out of 2            136
Ground out of 4        97
2 out of 4             90
1 out of 4             88
...
Upper Basement          3
Lower Basement          2
```

The `Floor` column stores two pieces of information: the unit's floor and the building's total height. It will need to be parsed into two separate numeric columns — a natural data cleaning task for the next lesson.

## Step 11: Correlation Matrix

```python
numeric_df = df[["BHK", "Rent", "Size", "Bathroom"]]

plt.figure(figsize=(6, 5))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix — Numeric Features")
plt.tight_layout()
plt.show()
```

Output (approximate):
```
           BHK   Rent   Size  Bathroom
BHK       1.00   0.41   0.61      0.66
Rent      0.41   1.00   0.58      0.39
Size      0.61   0.58   1.00      0.57
Bathroom  0.66   0.39   0.57      1.00
```

`Size` is the strongest numeric predictor of `Rent` (r = 0.58). `BHK` and `Bathroom` are correlated with each other and with `Size` — they're not independent signals. `City` and `Furnishing Status`, being categorical, don't appear here but showed strong effects in the boxplots.

## EDA Summary

After exploration, here is what's known about this dataset:

| Finding | Implication |
|---------|-------------|
| `Rent` is highly right-skewed | Log-transform before modeling |
| `Size` is right-skewed with extreme low values (< 100 sq ft) | Investigate and potentially cap |
| `Floor` is a string encoding two values | Parse into `current_floor` + `total_floors` |
| `Area Locality` has 2,235 unique values | Drop or reduce cardinality |
| `City` and `Furnishing Status` strongly predict rent | Encode as features |
| Very sparse 5 and 6 BHK listings | Consider capping or treating separately |
| Bathroom max of 10 | Validate against BHK |

These findings directly drive the cleaning and feature engineering steps. EDA is not just exploration — it is a decision-making process.

## What's Next

In the next lesson, you'll tackle **Data Cleaning**: parsing the `Floor` column, handling outliers in `Rent` and `Size`, validating the `BHK`/`Bathroom` relationship, and deciding how to handle `Area Locality`. Every decision will be grounded in what you observed here.
