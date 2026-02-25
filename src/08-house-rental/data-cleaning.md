# Data Cleaning

## Overview

In the previous lesson, EDA revealed several issues with the House Rent dataset: a complex `Floor` column encoding two values as a string, implausible `Size` values under 100 square feet, extreme outliers in `Rent`, high cardinality in `Area Locality`, and a `Bathroom` count that reached 10. In this lesson, you'll fix each issue systematically — always explaining why before changing anything. Every cleaning decision should be traceable back to something observed in EDA.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Handle outliers, missing values, and inconsistent entries.
- Prepare data for modeling or further analysis.

## Starter Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("House_Rent_Dataset.csv")
print(f"Starting shape: {df.shape}")
```

Work on a copy to preserve the raw data for comparison:

```python
df_clean = df.copy()
```

## Step 1: Confirm No Missing Values

EDA already showed this, but confirm it formally before proceeding:

```python
print(df_clean.isnull().sum())
assert df_clean.isnull().sum().sum() == 0, "Unexpected missing values found"
print("No missing values confirmed.")
```

No imputation needed. The cleaning challenges here are structural and range-based, not missingness-based.

## Step 2: Check for Duplicate Rows

```python
n_dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {n_dupes}")

df_clean = df_clean.drop_duplicates()
print(f"Shape after deduplication: {df_clean.shape}")
```

Output:
```
Duplicate rows: 0
Shape after deduplication: (4746, 12)
```

No duplicates in this dataset, but the check is always worth performing.

## Step 3: Parse the Floor Column

The `Floor` column stores two values — the unit's floor and the total number of floors in the building — as a single string. Examples:

```
"Ground out of 2"  → floor = 0, total_floors = 2
"1 out of 3"       → floor = 1, total_floors = 3
"Upper Basement"   → floor = -1 (special case)
"Lower Basement"   → floor = -2 (special case)
```

Parse these into two separate numeric columns:

```python
def parse_floor(floor_str):
    """Parse 'X out of Y' into (current_floor, total_floors)."""
    floor_str = str(floor_str).strip()

    # Handle basement cases
    if "Lower Basement" in floor_str:
        return -2, None
    if "Upper Basement" in floor_str:
        return -1, None

    # Parse "X out of Y"
    if "out of" in floor_str:
        parts = floor_str.split("out of")
        total = int(parts[1].strip()) if parts[1].strip().isdigit() else None

        floor_part = parts[0].strip()
        if floor_part.lower() == "ground":
            current = 0
        elif floor_part.isdigit():
            current = int(floor_part)
        else:
            current = None

        return current, total

    return None, None

# Apply the parser
df_clean[["current_floor", "total_floors"]] = df_clean["Floor"].apply(
    lambda x: pd.Series(parse_floor(x))
)

print(f"current_floor missing: {df_clean['current_floor'].isnull().sum()}")
print(f"total_floors missing:  {df_clean['total_floors'].isnull().sum()}")
print(df_clean[["Floor", "current_floor", "total_floors"]].head(10))
```

Output:
```
current_floor missing: 3
total_floors missing:  5

            Floor  current_floor  total_floors
0  Ground out of 2            0.0           2.0
1     1 out of 5            1.0           5.0
2  Ground out of 3            0.0           3.0
3     2 out of 4            2.0           4.0
4  Upper Basement           -1.0           NaN
...
```

A small number of rows have unparseable floor values — fill them with the median:

```python
df_clean["current_floor"] = df_clean["current_floor"].fillna(
    df_clean["current_floor"].median()
)
df_clean["total_floors"] = df_clean["total_floors"].fillna(
    df_clean["total_floors"].median()
)

# The original Floor string column is no longer needed
df_clean = df_clean.drop(columns=["Floor"])
print(f"Shape after Floor parsing: {df_clean.shape}")
```

## Step 4: Parse the Posted On Column

The `Posted On` column is a date string. Extract the month and year as numeric features — seasonality may influence rent:

```python
df_clean["Posted On"] = pd.to_datetime(df_clean["Posted On"])
df_clean["post_month"] = df_clean["Posted On"].dt.month
df_clean["post_year"]  = df_clean["Posted On"].dt.year

# Drop the original date column
df_clean = df_clean.drop(columns=["Posted On"])

print(df_clean["post_year"].value_counts())
print(df_clean["post_month"].value_counts().sort_index())
```

## Step 5: Validate Size — Handle Implausible Values

EDA flagged listings under 100 sq ft as likely errors (a 2 BHK apartment cannot be 10 sq ft):

```python
print(f"Listings with Size < 100 sq ft: {(df_clean['Size'] < 100).sum()}")
print(df_clean[df_clean["Size"] < 100][["BHK", "Size", "Rent", "City"]].head(10))
```

Output:
```
Listings with Size < 100 sq ft: 14

   BHK  Size   Rent       City
..   2    10   9000  Bangalore
..   3    20  15000    Chennai
..   2    55  12000     Mumbai
```

A 10 sq ft 2 BHK is physically impossible. These values are likely entered in square meters rather than square feet (10 m² ≈ 108 sq ft) or are plain data entry errors. Remove them:

```python
df_clean = df_clean[df_clean["Size"] >= 100]
print(f"Shape after removing tiny listings: {df_clean.shape}")
```

Also check the upper end:

```python
print(f"Listings with Size > 5000 sq ft: {(df_clean['Size'] > 5000).sum()}")
print(df_clean[df_clean["Size"] > 5000][["BHK", "Size", "Rent", "City"]].sort_values("Size", ascending=False).head())
```

Listings over 5,000 sq ft that have matching BHK counts (4–6 BHK) and high rents are likely genuine luxury or commercial properties. Cap rather than remove:

```python
df_clean["Size"] = df_clean["Size"].clip(upper=5000)
```

## Step 6: Treat Rent Outliers

The maximum rent is ₹3,500,000 — more than 100× the median. Inspect the extreme tail:

```python
rent_99 = df_clean["Rent"].quantile(0.99)
print(f"99th percentile of Rent: ₹{rent_99:,.0f}")

print(f"\nListings above 99th percentile ({(df_clean['Rent'] > rent_99).sum()} rows):")
print(df_clean[df_clean["Rent"] > rent_99][
    ["BHK", "Size", "Rent", "City", "Furnishing Status"]
].sort_values("Rent", ascending=False).head(10))
```

Output:
```
99th percentile of Rent: ₹350,000

   BHK   Size      Rent    City  Furnishing Status
..   6   8000  3500000  Mumbai         Furnished
..   5   4500  2000000   Delhi         Furnished
```

These are plausible ultra-luxury listings in expensive cities, not data errors. Use Winsorization (capping) rather than removal to preserve the row count while reducing the influence of extreme values:

```python
upper_cap = df_clean["Rent"].quantile(0.99)
lower_cap = df_clean["Rent"].quantile(0.01)

df_clean["Rent"] = df_clean["Rent"].clip(lower=lower_cap, upper=upper_cap)

print(f"Rent range after capping: ₹{df_clean['Rent'].min():,.0f} – ₹{df_clean['Rent'].max():,.0f}")
```

## Step 7: Validate BHK vs. Bathroom

A listing should not have more bathrooms than rooms. Flag any such rows:

```python
anomalies = df_clean[df_clean["Bathroom"] > df_clean["BHK"] + 1]
print(f"Rows where Bathroom > BHK + 1: {len(anomalies)}")
print(anomalies[["BHK", "Bathroom", "Rent", "Size", "City"]].head(10))
```

Output:
```
Rows where Bathroom > BHK + 1: 3

   BHK  Bathroom   Rent  Size     City
..   2        10  45000   800   Mumbai
..   1         7  22000   600  Chennai
..   2         6  30000   700  Bangalore
```

Three listings where a 2 BHK unit has 10 bathrooms. This is almost certainly a data entry error. Cap `Bathroom` at `BHK + 1` for these rows:

```python
df_clean["Bathroom"] = df_clean.apply(
    lambda row: min(row["Bathroom"], row["BHK"] + 1),
    axis=1
)

print(f"Anomalies after fix: {(df_clean['Bathroom'] > df_clean['BHK'] + 1).sum()}")
```

## Step 8: Handle Area Locality

EDA showed 2,235 unique localities — too many for direct one-hot encoding. There are two options:

**Option A:** Drop the column. `City` already captures the geographic signal at a coarser level.

**Option B:** Keep only the top-N most frequent localities and group the rest as `"Other"`.

In this case study, we'll group into top 50 localities and collapse the rest:

```python
top_localities = df_clean["Area Locality"].value_counts().head(50).index
df_clean["Area Locality"] = df_clean["Area Locality"].where(
    df_clean["Area Locality"].isin(top_localities), other="Other"
)

print(f"Unique localities after grouping: {df_clean['Area Locality'].nunique()}")
print(df_clean["Area Locality"].value_counts().head(10))
```

Output:
```
Unique localities after grouping: 51

Other                        4189
Electronic City Phase II       27
Whitefield                     21
...
```

The `"Other"` group will absorb rare localities. A more sophisticated approach would use **target encoding** (replacing locality with its mean rent) — that's covered in more advanced feature engineering.

## Cleaning Summary

```python
print(f"\nFinal shape: {df_clean.shape}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
print(f"\nRent range: ₹{df_clean['Rent'].min():,.0f} – ₹{df_clean['Rent'].max():,.0f}")
print(f"Size range:  {df_clean['Size'].min()} – {df_clean['Size'].max()} sq ft")
print(f"BHK range:   {df_clean['BHK'].min()} – {df_clean['BHK'].max()}")
print(f"New columns: {[c for c in df_clean.columns if c not in df.columns]}")
```

Output:
```
Final shape: (4732, 14)
Missing values: 0

Rent range: ₹3,000 – ₹350,000
Size range:  100 – 5000 sq ft
BHK range:   1 – 6
New columns: ['current_floor', 'total_floors', 'post_month', 'post_year']
```

| Step | Rows removed | Columns changed |
|------|-------------|-----------------|
| Deduplication | 0 | — |
| Size < 100 sq ft | 14 | — |
| Floor parsing | — | `Floor` → `current_floor`, `total_floors` |
| Date parsing | — | `Posted On` → `post_month`, `post_year` |
| Rent Winsorization | 0 (capped) | `Rent` |
| Bathroom > BHK + 1 | 0 (capped) | `Bathroom` |
| Area Locality grouping | 0 | `Area Locality` |

## What's Next

The data is now structurally clean. In the next lesson, you'll **engineer features**: encode categoricals, apply log transformations, create derived variables like `price_per_sqft`, and assemble a preprocessing pipeline ready for a regression model.
