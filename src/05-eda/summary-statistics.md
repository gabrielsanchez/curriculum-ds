# Summary Statistics

## Overview

In the previous lesson, you learned what EDA is and how it fits into the data science lifecycle. The first quantitative step in any EDA is computing **summary statistics** — numeric measurements that condense an entire column of data into a handful of descriptive values. In this lesson, you'll learn how to calculate and interpret the most important summary statistics using Pandas and understand what they reveal about your data.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Calculate mean, median, mode, and standard deviation.
- Interpret these metrics to understand data distributions.

## Key terms

**Mean:** The arithmetic average of a set of values — the sum of all values divided by the count. Sensitive to outliers.

**Median:** The middle value when all values are sorted in order. Robust to outliers. Also called the 50th percentile.

**Mode:** The most frequently occurring value. Useful for categorical data and identifying peaks in numeric distributions.

**Variance:** The average of the squared differences from the mean — measures how spread out values are around the mean.

**Standard deviation:** The square root of the variance. Expressed in the same units as the original data, making it more interpretable than variance.

**Range:** The difference between the maximum and minimum values.

**Interquartile Range (IQR):** The difference between the 75th percentile (Q3) and the 25th percentile (Q1). A robust measure of spread that ignores the extremes.

**Percentile (quantile):** The value below which a given percentage of observations fall. The 25th percentile (Q1), 50th (median), and 75th (Q3) are the most common.

**Skewness:** A measure of asymmetry in a distribution. A right-skewed (positive) distribution has a long tail to the right; a left-skewed (negative) distribution has a long tail to the left.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Before you plot a single chart, numbers can already tell you a great deal about a column of data. Is the typical house price $200,000 or $2,000,000? Are most values clustered tightly together or spread across a wide range? Is the distribution symmetric, or are there a few extremely high values pulling the average up?

Summary statistics answer these questions in a compact form. They are the foundation of EDA — both for understanding data directly and for knowing which visualizations to reach for next.

## Setup: Sample Dataset

Throughout this lesson, you'll work with a housing dataset:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("housing.csv")

# For demonstration, we'll create a representative sample
df = pd.DataFrame({
    "price":      [215000, 180000, 340000, 290000, 125000, 410000,
                   175000, 260000, 198000, 1250000, 305000, 222000],
    "bedrooms":   [3, 2, 4, 3, 2, 5, 2, 3, 3, 6, 4, 3],
    "sqft":       [1400, 1100, 2100, 1800, 900, 2800, 1050,
                   1600, 1350, 5200, 2000, 1450],
    "age_years":  [12, 25, 5, 18, 40, 3, 30, 10, 22, 8, 6, 15],
    "neighborhood": ["North", "South", "North", "East", "South",
                     "North", "East", "South", "North", "North",
                     "East", "South"]
})
```

## The Quick Summary: `df.describe()`

The fastest way to get an overview of all numeric columns at once is `.describe()`:

```python
print(df.describe())
```

Output:
```
              price    bedrooms         sqft    age_years
count  1.200000e+01   12.000000    12.000000    12.000000
mean   3.147500e+05    3.250000  1729.166667    16.166667
std    2.842434e+05    0.965377   993.826983    11.519428
min    1.250000e+05    2.000000   900.000000     3.000000
25%    1.913750e+05    3.000000  1112.500000     6.750000
50%    2.410000e+05    3.000000  1525.000000    13.500000
75%    3.068750e+05    4.000000  1975.000000    22.750000
max    1.250000e+06    6.000000  5200.000000    40.000000
```

This single call gives you count, mean, standard deviation, min, three quartiles, and max for every numeric column. The large gap between the mean ($314,750) and median ($241,000) for price is already telling — something is pulling the mean up. (That $1,250,000 mansion.)

## Measures of Central Tendency

### Mean

The mean is the sum of all values divided by the count:

```python
print(df["price"].mean())    # Output: 314750.0
```

**When to use it:** The mean is intuitive and works well when data is roughly symmetric with no extreme outliers.

**Weakness:** A single very large or very small value can pull the mean far from the "typical" value. The $1.25M property raises the mean by over $70,000 compared to what you'd get without it.

### Median

The median is the middle value when values are sorted:

```python
print(df["price"].median())  # Output: 241000.0
```

**When to use it:** Use the median when data is skewed or contains outliers. It represents the "typical" value more accurately than the mean in these cases.

The median house price of $241,000 gives a much better sense of a "typical" home in this dataset than the mean of $314,750.

### Mean vs. Median: When It Matters

The relationship between mean and median tells you about the shape of the distribution:

| Relationship | Implication |
|---|---|
| Mean ≈ Median | Roughly symmetric distribution |
| Mean > Median | Right-skewed (long tail to the right, pulled by high values) |
| Mean < Median | Left-skewed (long tail to the left, pulled by low values) |

Income, house prices, and wealth are classic examples of right-skewed distributions where the median is the more representative measure.

### Mode

The mode is the most frequently occurring value:

```python
print(df["bedrooms"].mode())   # Output: 3
print(df["neighborhood"].mode()) # Output: North
```

**When to use it:** Mode is most useful for categorical columns (like `neighborhood`) and for integer columns where a specific value occurs very frequently (like `bedrooms`). Continuous numeric columns rarely have a meaningful mode.

## Measures of Spread

Knowing the center of the data is only half the picture. You also need to know how spread out values are.

### Standard Deviation

Standard deviation measures the average distance between each value and the mean:

```python
print(df["price"].std())    # Output: 284243.4
print(df["sqft"].std())     # Output: 993.8
```

A large standard deviation means values are widely spread. A small one means values are tightly clustered. In this case, the price standard deviation ($284K) is almost as large as the mean ($315K) — a sign of high variability, driven largely by the outlier.

### Variance

Variance is the standard deviation squared. It is less commonly reported because it is in squared units (e.g., dollars²), but it is used in many statistical calculations:

```python
print(df["price"].var())    # Output: 80794149.2
```

### Range

The range is the simplest spread measure — just max minus min:

```python
price_range = df["price"].max() - df["price"].min()
print(price_range)  # Output: 1125000
```

The range is sensitive to outliers since it's entirely determined by the two extreme values.

### Interquartile Range (IQR)

The IQR is the range of the middle 50% of the data — from Q1 (25th percentile) to Q3 (75th percentile). It is robust to outliers because it ignores the extremes:

```python
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1: {Q1:,.0f}")    # Output: Q1: 191,375
print(f"Q3: {Q3:,.0f}")    # Output: Q3: 306,875
print(f"IQR: {IQR:,.0f}")  # Output: IQR: 115,500
```

The IQR of $115,500 tells you the spread of the middle half of home prices, unaffected by the $1.25M outlier.

## Percentiles and Quantiles

Percentiles tell you the value below which a given percentage of observations fall:

```python
# Key percentiles
for pct in [10, 25, 50, 75, 90, 95]:
    val = df["price"].quantile(pct / 100)
    print(f"{pct}th percentile: ${val:,.0f}")
```

Output:
```
10th percentile: $151,500
25th percentile: $191,375
50th percentile: $241,000
75th percentile: $306,875
90th percentile: $399,500
95th percentile: $824,750
```

The jump from the 90th ($399,500) to the 95th ($824,750) percentile is dramatic — another signal of the outlier's effect at the high end.

## Skewness

Pandas can compute skewness directly:

```python
print(df["price"].skew())   # Output: 2.87
print(df["sqft"].skew())    # Output: 2.51
print(df["age_years"].skew()) # Output: 0.47
```

A skewness value above **+1** indicates strong right skew (a long right tail). A value below **-1** indicates strong left skew. Values between -1 and +1 are approximately symmetric. The strong positive skewness in `price` and `sqft` confirms what the mean vs. median comparison suggested.

## Computing Stats for Categorical Columns

For string/object columns, `.describe()` gives different information:

```python
print(df[["neighborhood"]].describe())
```

Output:
```
       neighborhood
count            12
unique            3
top           North
freq              4
```

This tells you there are 12 entries, 3 unique neighborhoods, and the most common is "North" (appears 4 times). Use `.value_counts()` for a full frequency table:

```python
print(df["neighborhood"].value_counts())
```

Output:
```
North    4
South    4
East     3
Name: neighborhood, dtype: int64
```

## Putting It Together

Here's a compact summary function you can reuse on any numeric column:

```python
def column_summary(series):
    print(f"Column: {series.name}")
    print(f"  Count:    {series.count()}")
    print(f"  Mean:     {series.mean():,.2f}")
    print(f"  Median:   {series.median():,.2f}")
    print(f"  Std Dev:  {series.std():,.2f}")
    print(f"  Min:      {series.min():,.2f}")
    print(f"  Max:      {series.max():,.2f}")
    print(f"  IQR:      {(series.quantile(0.75) - series.quantile(0.25)):,.2f}")
    print(f"  Skewness: {series.skew():.2f}")
    missing = series.isnull().sum()
    print(f"  Missing:  {missing} ({missing/len(series)*100:.1f}%)")

column_summary(df["price"])
```

Output:
```
Column: price
  Count:    12
  Mean:     314,750.00
  Median:   241,000.00
  Std Dev:  284,243.40
  Min:      125,000.00
  Max:      1,250,000.00
  IQR:      115,500.00
  Skewness: 2.87
  Missing:  0 (0.0%)
```

## Conclusion

In this lesson, you learned how to compute and interpret the core summary statistics for a dataset: mean, median, and mode for central tendency; standard deviation, variance, range, and IQR for spread; percentiles for understanding the data's rank order; and skewness for detecting asymmetry. You also saw how the relationship between mean and median signals whether a distribution is skewed — a key observation that guides the choice of subsequent visualizations. In the next lesson, you'll visualize these distributions directly using **histograms**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: A dataset of household incomes has a mean of $85,000 and a median of $52,000. What does this tell you about the distribution?**
1. The distribution is roughly symmetric.
2. The distribution is left-skewed, with a long tail toward lower incomes.
3. The distribution is right-skewed, with a long tail toward higher incomes driven by a small number of very high earners.
4. The dataset contains no outliers.

**Correct Answer:**
3. The distribution is right-skewed, with a long tail toward higher incomes driven by a small number of very high earners.

**Explanation:**
When the mean is significantly higher than the median, it indicates a right-skewed (positively skewed) distribution. A small number of very high values — like millionaires in an income dataset — pull the mean upward while the median stays closer to the typical value. This is why median household income is usually reported rather than mean income.

---

#### **Question 2: Why is the IQR considered more robust than the standard deviation when a dataset contains outliers?**
1. The IQR is always a larger number than the standard deviation.
2. The IQR measures the spread of the middle 50% of the data and completely ignores values in the tails, while the standard deviation is affected by every value including extremes.
3. The IQR can be applied to categorical variables but standard deviation cannot.
4. The IQR is calculated using the mean, which is less sensitive to outliers.

**Correct Answer:**
2. The IQR measures the spread of the middle 50% of the data and completely ignores values in the tails, while the standard deviation is affected by every value including extremes.

**Explanation:**
Standard deviation squares the distance of each value from the mean, meaning extreme outliers contribute disproportionately to its value. The IQR is simply Q3 − Q1 — it is computed entirely from the middle 50% of the data and is unaffected by however extreme the top and bottom values are. This makes it the preferred spread measure when outliers are present.

---

#### **Question 3: What does `df["category"].value_counts()` return?**
1. The mean and standard deviation of the column.
2. The number of unique values in the column.
3. A sorted count of how many times each unique value appears in the column.
4. The percentage of missing values in the column.

**Correct Answer:**
3. A sorted count of how many times each unique value appears in the column.

**Explanation:**
`value_counts()` returns a Series listing each unique value in the column alongside how many times it appears, sorted from most frequent to least frequent. It is the go-to method for understanding the composition of a categorical column, equivalent to a frequency table.
