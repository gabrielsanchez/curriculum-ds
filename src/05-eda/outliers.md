# Outliers

## Overview

Throughout this module, you've been building the skills to detect anomalies in data — you've seen extreme values flagged by box plots, values that pull the mean far from the median, and correlations that look unusually strong. In this lesson, you'll bring all of that together and learn to **formally identify outliers** using both visual and statistical methods, and — crucially — learn how to decide what to do with them.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Identify outliers using visual and statistical methods.
- Decide how to treat or remove outliers based on context.

## Key terms

**Outlier:** A data point that differs markedly from the other observations — either much higher, much lower, or otherwise anomalous compared to the rest of the data.

**IQR method:** A statistical approach to detecting outliers: any value below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR is flagged as a potential outlier.

**Z-score:** A standardized measure of how many standard deviations a value is from the mean. Values with |z| > 3 are commonly flagged as outliers.

**Winsorization (capping):** A treatment that replaces outlier values with the boundary value (e.g., the 1st and 99th percentile), rather than removing the row entirely.

**Imputation:** Replacing a problematic value with a more reasonable estimate, such as the median.

**Data error:** An outlier caused by a mistake in data collection or entry — for example, a negative age or a price of $0.

**Legitimate extreme value:** An outlier that is real and accurate but represents an unusual case — for example, a billionaire in an income dataset.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Not all outliers are the same. A house price of $5,000,000 in a dataset of typical homes might be a real luxury property — or a misplaced decimal point. A temperature reading of −999°C is obviously a sentinel value used to represent missing data. A survey response of 150 on a 1–100 scale is a data entry error.

How you handle an outlier depends entirely on **why it is there**. Blindly removing all outliers can delete valuable information and bias your analysis. Blindly keeping them can distort summary statistics, correlations, and model training. The right approach is to detect, investigate, and then make an informed decision.

## Setup: Sample Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 300

df = pd.DataFrame({
    "price":     np.concatenate([
                     np.random.normal(250000, 40000, n - 5),
                     [1_500_000, 1_200_000,   # Legitimate luxury homes
                      -5000, 0, 999_999_999]  # Data errors
                 ]),
    "sqft":      np.concatenate([
                     np.random.normal(1600, 350, n - 3),
                     [8500, 9200, -100]  # Extreme values
                 ]),
    "age_years": np.concatenate([
                     np.random.normal(18, 8, n - 2),
                     [150, -5]           # Impossible values
                 ])
})
```

## Visual Methods

### Box Plot

The box plot is the fastest visual outlier detector. Points beyond the whiskers are immediately apparent:

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, col in zip(axes, ["price", "sqft", "age_years"]):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6),
               medianprops=dict(color="red", linewidth=2),
               flierprops=dict(marker="o", color="red", alpha=0.5))
    ax.set_title(col)
    ax.set_ylabel("Value")

plt.suptitle("Box Plots — Outlier Detection")
plt.tight_layout()
plt.show()
```

The `flierprops` parameter styles the outlier points (red dots beyond the whiskers).

### Scatter Plot

Scatter plots are useful when outliers exist in the **relationship** between two variables:

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["sqft"], df["price"], alpha=0.4, color="steelblue")
plt.title("Price vs. Square Footage")
plt.xlabel("Square Feet")
plt.ylabel("Price ($)")
plt.show()
```

A home with 9,200 sqft and $0 price, or one with −100 sqft, will stand out immediately on this plot as impossible combinations.

### Histogram

Extreme values appear as isolated bars far from the main distribution:

```python
plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=50, color="steelblue", edgecolor="white")
plt.title("Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.show()
```

The $999,999,999 entry will appear as a lone bar at the far right, completely disconnected from the main cluster.

## Statistical Methods

### Method 1: The IQR Method

The IQR method defines outlier boundaries as:
- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

```python
def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    print(f"\n{series.name}:")
    print(f"  Bounds: [{lower:,.1f}, {upper:,.1f}]")
    print(f"  Outliers: {mask.sum()} ({mask.mean()*100:.1f}%)")
    return mask

for col in ["price", "sqft", "age_years"]:
    iqr_outliers(df[col])
```

Output:
```
price:
  Bounds: [120,537.5, 388,462.5]
  Outliers: 5 (1.7%)

sqft:
  Bounds: [689.8, 2,556.6]
  Outliers: 3 (1.0%)

age_years:
  Bounds: [-0.9, 40.3]
  Outliers: 2 (0.7%)
```

The IQR method is robust to outliers in the boundaries themselves and works well for moderately skewed data.

### Method 2: The Z-Score Method

A Z-score measures how many standard deviations a value is from the mean. Values with |z| > 3 are commonly flagged as outliers:

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df["price"]))
outlier_mask = z_scores > 3

print(f"Z-score outliers in 'price': {outlier_mask.sum()}")
print(df["price"][outlier_mask].values)
```

Output:
```
Z-score outliers in 'price': 3
[  1500000.   1200000. 999999999.]
```

**Limitation:** Because the Z-score uses the mean and standard deviation, which are themselves affected by outliers, this method is less reliable when the data has many extreme values. Use the IQR method as a first pass, especially on skewed data.

### Comparing the Two Methods

| Feature | IQR Method | Z-Score Method |
|---------|-----------|---------------|
| Sensitive to existing outliers | Less (uses quartiles) | More (uses mean/std) |
| Works on skewed data | Better | Less reliable |
| Threshold | 1.5 × IQR beyond box | |z| > 3 |
| Typical use | General EDA | Approximately normal data |

## Investigating Before Acting

Once you've identified potential outliers, **investigate before removing**:

```python
# Look at the flagged rows
lower_bound = df["price"].quantile(0.25) - 1.5 * (df["price"].quantile(0.75) - df["price"].quantile(0.25))
upper_bound = df["price"].quantile(0.75) + 1.5 * (df["price"].quantile(0.75) - df["price"].quantile(0.25))

flagged = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
print(flagged[["price", "sqft", "age_years"]])
```

Ask yourself:
- Is the value **impossible** (negative age, price of $0)? → Data error, remove or impute.
- Is the value **implausible but possible** ($999,999,999)? → Likely a data error, investigate.
- Is the value **extreme but legitimate** ($1,500,000 luxury home)? → Keep it, note it, or handle it with capping.

## Treatment Options

### Option 1: Remove the Row

Appropriate when the row is a clear data error that cannot be recovered:

```python
# Remove impossible values
df_clean = df[df["price"] > 0]                    # No negative/zero prices
df_clean = df_clean[df_clean["age_years"] >= 0]   # No negative ages
df_clean = df_clean[df_clean["age_years"] <= 120] # No implausible ages
df_clean = df_clean[df_clean["price"] < 10_000_000] # Cap at a plausible maximum
```

### Option 2: Winsorization (Capping)

Cap outliers at the percentile boundaries instead of removing the row. This preserves the row while reducing the outlier's influence:

```python
lower_cap = df["price"].quantile(0.01)
upper_cap = df["price"].quantile(0.99)

df["price_capped"] = df["price"].clip(lower=lower_cap, upper=upper_cap)

print(f"Original max: ${df['price'].max():,.0f}")
print(f"Capped max:   ${df['price_capped'].max():,.0f}")
```

### Option 3: Log Transformation

For right-skewed data with many high-value outliers, a log transformation compresses the scale so that extreme values are less influential:

```python
df["log_price"] = np.log1p(df["price"])
```

### Option 4: Keep the Outlier

If the outlier is a legitimate extreme value that is genuinely part of the phenomenon you're studying, keeping it is correct. Removing true outliers introduces bias. For example, in a dataset of all homes in a city, a $5M mansion is real data — removing it would under-represent the high end of the market.

### Option 5: Impute

If the outlier is a data error but the row is otherwise valuable, replace the bad value with a reasonable estimate:

```python
median_price = df[df["price"] > 0]["price"].median()
df["price"] = df["price"].apply(lambda x: median_price if x <= 0 else x)
```

## Outlier Treatment Decision Framework

```
Is the value impossible (negative age, price = 0)?
  → Yes: Remove the row or impute.

Is the value possibly a data entry error (e.g., 999999999)?
  → Investigate the source. If likely an error, remove or impute.

Is the value real but extreme?
  → Does your analysis/model need to represent the full range?
      → Yes: Keep it, possibly transform (log).
      → No (e.g., you're modeling the typical case): Cap with Winsorization.
```

## Conclusion

In this lesson, you learned how to detect outliers visually using box plots, scatter plots, and histograms, and statistically using the IQR method and Z-scores. More importantly, you learned that outlier treatment requires judgment: investigate the cause before acting, distinguish data errors from legitimate extremes, and choose a treatment — removal, capping, transformation, imputation, or keeping — that fits the context. This is the final analytical skill in the EDA toolkit. In the next lesson, you'll apply the full EDA workflow — summary statistics, histograms, box plots, correlation matrix, and outlier analysis — in the **module assessment**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: Using the IQR method, what are the outlier boundaries for a column where Q1 = 100 and Q3 = 200?**
1. Lower: 50, Upper: 250
2. Lower: 0, Upper: 300
3. Lower: −50, Upper: 350
4. Lower: 75, Upper: 225

**Correct Answer:**
3. Lower: −50, Upper: 350

**Explanation:**
IQR = Q3 − Q1 = 200 − 100 = 100. Lower bound = Q1 − 1.5 × IQR = 100 − 150 = −50. Upper bound = Q3 + 1.5 × IQR = 200 + 150 = 350. Any value below −50 or above 350 is flagged as a potential outlier.

---

#### **Question 2: A dataset of employee ages contains the value 187. What is the most appropriate action?**
1. Keep the value — it might represent a very old employee.
2. Apply a log transformation to reduce its influence.
3. Remove or correct the value — 187 is an impossible age and is clearly a data error.
4. Cap it using Winsorization to the 99th percentile.

**Correct Answer:**
3. Remove or correct the value — 187 is an impossible age and is clearly a data error.

**Explanation:**
An age of 187 is impossible — the oldest verified human was 122. This is clearly a data entry error (possibly a typo of 87, or a placeholder). Impossible values should be corrected if the correct value can be determined, or removed/imputed if it cannot. Applying transformations or capping would still leave an erroneous value in the dataset.

---

#### **Question 3: What is the key difference between removing an outlier and Winsorizing it?**
1. Removing an outlier eliminates the entire row; Winsorizing replaces the extreme value with a boundary value and keeps the row.
2. Winsorizing can only be applied to positive values; removing works on any value.
3. Removing always improves model performance; Winsorizing always makes it worse.
4. There is no difference — both result in the value being excluded from analysis.

**Correct Answer:**
1. Removing an outlier eliminates the entire row; Winsorizing replaces the extreme value with a boundary value and keeps the row.

**Explanation:**
When you remove a row, all information in that row — including values in other columns that may be perfectly valid — is lost. Winsorization (capping) replaces only the extreme value with the boundary value (e.g., the 99th percentile), preserving the entire row. This is preferable when the outlier is in just one column but the rest of the row contains useful data.
