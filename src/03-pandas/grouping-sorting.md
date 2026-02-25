# Grouping and Sorting

## Overview

In the previous lesson, you learned how to map and apply functions to transform individual values in a DataFrame. Now you'll take a step back and look at the bigger picture: understanding patterns across **groups** of data. In this lesson, you'll learn how to use `groupby()` to aggregate data by category and how to sort a DataFrame to surface the most important rows.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Group data by categorical values and aggregate results.
- Sort values by one or more columns.

## Key terms

**`groupby()`:** A Pandas method that splits a DataFrame into groups based on the unique values of one or more columns, allowing you to apply aggregate functions to each group independently.

**Aggregation:** The process of computing a summary statistic (e.g., sum, mean, count, min, max) across multiple rows within a group.

**`agg()`:** A flexible aggregation method that lets you apply multiple or custom aggregation functions to grouped data.

**`sort_values()`:** A DataFrame method that reorders rows based on the values in one or more columns.

**`sort_index()`:** A method that sorts rows by the DataFrame's index rather than its values.

**`ascending`:** A boolean parameter in `sort_values()` that controls whether the sort order is ascending (`True`, default) or descending (`False`).

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Analyzing individual rows only gets you so far. Real insights in data science come from understanding how groups of records compare to each other — average sales by region, total revenue per product category, number of customers per city. This kind of **split-apply-combine** workflow is at the heart of exploratory data analysis, and Pandas' `groupby()` makes it both concise and powerful.

Sorting complements grouping by helping you rank results, surface extremes (highest/lowest), and present data in a logical order for reports or visualizations.

## Setup: Sample Dataset

Throughout this lesson, you'll work with a sales dataset:

```python
import pandas as pd

df = pd.DataFrame({
    "region":   ["North", "South", "North", "East", "South", "East", "North", "South"],
    "product":  ["Laptop", "Phone", "Phone", "Laptop", "Tablet", "Phone", "Tablet", "Laptop"],
    "rep":      ["Alex", "Jordan", "Morgan", "Sam", "Taylor", "Alex", "Jordan", "Morgan"],
    "sales":    [1200, 850, 950, 1100, 600, 780, 700, 1050],
    "units":    [5, 12, 10, 4, 8, 11, 7, 6]
})
```

## Sorting with `sort_values()`

### Sorting by a Single Column

```python
# Sort by sales (ascending — lowest first)
print(df.sort_values("sales"))

# Sort by sales (descending — highest first)
print(df.sort_values("sales", ascending=False))
```

Output (descending):
```
  region product     rep  sales  units
0  North  Laptop    Alex   1200      5
3   East  Laptop     Sam   1100      4
7  South  Laptop  Morgan   1050      6
2  North   Phone  Morgan    950     10
1  South   Phone  Jordan    850     12
5   East   Phone    Alex    780     11
6  North  Tablet  Jordan    700      7
4  South  Tablet  Taylor    600      8
```

### Sorting by Multiple Columns

Pass a list of column names and a corresponding list of `ascending` booleans:

```python
# Sort by region (A-Z), then by sales within each region (highest first)
print(df.sort_values(["region", "sales"], ascending=[True, False]))
```

Output:
```
  region product     rep  sales  units
3   East  Laptop     Sam   1100      4
5   East   Phone    Alex    780     11
0  North  Laptop    Alex   1200      5
2  North   Phone  Morgan    950     10
6  North  Tablet  Jordan    700      7
7  South  Laptop  Morgan   1050      6
1  South   Phone  Jordan    850     12
4  South  Tablet  Taylor    600      8
```

### `nlargest()` and `nsmallest()`

Quickly get the top or bottom N rows by a column:

```python
print(df.nlargest(3, "sales"))    # Top 3 by sales
print(df.nsmallest(2, "units"))   # Bottom 2 by units
```

## Grouping with `groupby()`

### Basic Grouping

`groupby(column)` splits the DataFrame by unique values of that column. Chain an aggregation function to compute a summary per group:

```python
# Total sales per region
print(df.groupby("region")["sales"].sum())
```

Output:
```
region
East     1880
North    2850
South    2500
Name: sales, dtype: int64
```

```python
# Average units per product
print(df.groupby("product")["units"].mean())
```

Output:
```
product
Laptop    5.000000
Phone    11.000000
Tablet    7.500000
Name: units, dtype: float64
```

### Common Aggregation Functions

| Method | Description |
|--------|-------------|
| `.sum()` | Total of values in each group |
| `.mean()` | Average value per group |
| `.count()` | Number of non-null rows per group |
| `.min()` / `.max()` | Minimum / maximum value per group |
| `.median()` | Median value per group |
| `.std()` | Standard deviation per group |

### Grouping by Multiple Columns

Pass a list to `groupby()` to create groups from combinations of columns:

```python
# Total sales broken down by region AND product
print(df.groupby(["region", "product"])["sales"].sum())
```

Output:
```
region  product
East    Laptop     1100
        Phone       780
North   Laptop     1200
        Phone       950
        Tablet      700
South   Laptop     1050
        Phone       850
        Tablet      600
Name: sales, dtype: int64
```

### Reset the Index After Grouping

`groupby()` sets the grouping columns as the index. Use `.reset_index()` to convert it back to a regular DataFrame:

```python
region_sales = df.groupby("region")["sales"].sum().reset_index()
print(region_sales)
```

Output:
```
  region  sales
0   East   1880
1  North   2850
2  South   2500
```

## Multiple Aggregations with `.agg()`

`.agg()` lets you apply several aggregation functions at once:

```python
summary = df.groupby("region")["sales"].agg(
    total="sum",
    average="mean",
    minimum="min",
    maximum="max"
).reset_index()

print(summary)
```

Output:
```
  region  total     average  minimum  maximum
0   East   1880   940.000000     780     1100
1  North   2850   950.000000     700     1200
2  South   2500   833.333333     600     1050
```

### Aggregating Multiple Columns

Apply different functions to different columns in one call:

```python
summary = df.groupby("region").agg(
    total_sales=("sales", "sum"),
    avg_units=("units", "mean"),
    num_transactions=("sales", "count")
).reset_index()

print(summary)
```

Output:
```
  region  total_sales  avg_units  num_transactions
0   East         1880        7.5                 2
1  North         2850        7.3                 3
2  South         2500        8.7                 3
```

## Combining Grouping and Sorting

A common pattern is to group, aggregate, then sort to rank the results:

```python
# Rank products by total revenue, highest first
product_revenue = (
    df.groupby("product")["sales"]
    .sum()
    .reset_index()
    .rename(columns={"sales": "total_revenue"})
    .sort_values("total_revenue", ascending=False)
)

print(product_revenue)
```

Output:
```
   product  total_revenue
0   Laptop           3350
1    Phone           2580
2   Tablet           1300
```

## Putting It Together: A Data Science Example

Here's a complete analysis that groups, aggregates, and ranks sales reps by performance:

```python
rep_summary = (
    df.groupby("rep")
    .agg(
        total_sales=("sales", "sum"),
        total_units=("units", "sum"),
        deals=("sales", "count")
    )
    .reset_index()
    .sort_values("total_sales", ascending=False)
)

rep_summary["avg_deal_size"] = (rep_summary["total_sales"] / rep_summary["deals"]).round(0)

print(rep_summary)
```

Output:
```
      rep  total_sales  total_units  deals  avg_deal_size
0    Alex         1980           16      2          990.0
3     Sam         1100            4      1         1100.0
4  Taylor          600            8      1          600.0
2  Morgan         2000           16      2         1000.0
1  Jordan         1550           19      2          775.0
```

## Conclusion

In this lesson, you learned how to sort a DataFrame by one or more columns using `sort_values()` and how to group data by categories using `groupby()`. You applied single and multiple aggregation functions using `.agg()` and combined grouping with sorting to rank results. The group-aggregate-sort pattern is one of the most frequently used workflows in data analysis — from summarizing sales by region to computing average test scores by student group. In the next lesson, you'll apply everything from this module in a hands-on **module assessment**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What does `df.groupby("region")["sales"].mean()` compute?**
1. The mean of the entire `sales` column, ignoring `region`.
2. The average `sales` value for each unique value in the `region` column.
3. The total `sales` for each `region`.
4. The number of rows in each `region` group.

**Correct Answer:**
2. The average `sales` value for each unique value in the `region` column.

**Explanation:**
`groupby("region")` splits the DataFrame into groups based on the unique values in the `region` column. Selecting `["sales"]` narrows the result to that column. Calling `.mean()` then computes the average `sales` within each group. The result is a Series indexed by the unique region values.

---

#### **Question 2: What does `.reset_index()` do after a `groupby()` operation?**
1. It removes all rows with missing values.
2. It resets the DataFrame to its original, ungrouped form.
3. It converts the group labels (the index) back into regular columns so the result is a standard DataFrame.
4. It re-sorts the DataFrame by its original row order.

**Correct Answer:**
3. It converts the group labels (the index) back into regular columns so the result is a standard DataFrame.

**Explanation:**
After `groupby().agg()`, the grouping column(s) become the index of the result. Calling `.reset_index()` moves those index labels back into regular columns, giving you a clean DataFrame with a default integer index — which is easier to work with in most downstream operations.

---

#### **Question 3: How do you sort a DataFrame by `sales` in descending order (highest first)?**
1. `df.sort_values("sales")`
2. `df.sort_values("sales", ascending=True)`
3. `df.sort_values("sales", ascending=False)`
4. `df.sort_values("sales", order="desc")`

**Correct Answer:**
3. `df.sort_values("sales", ascending=False)`

**Explanation:**
`sort_values()` sorts in ascending order by default (`ascending=True`). To reverse this and place the largest values first, set `ascending=False`. There is no `order` parameter in Pandas — option 4 would raise an error.
