# Selecting and Assigning Data

## Overview

In the previous lesson, you learned how to create DataFrames and load data from files. Now that you have data in a DataFrame, the next skill is knowing how to retrieve specific parts of it and how to add or modify columns. In this lesson, you'll learn how to select columns and rows using Pandas' powerful indexing tools, and how to assign new or updated data to a DataFrame.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Select columns by labels or indexes.
- Filter rows using conditions.
- Assign new columns or modify existing ones.

## Key terms

**`loc`:** A label-based indexer that selects rows and columns by their **names** (index labels and column names).

**`iloc`:** An integer-based indexer that selects rows and columns by their **position** (0-based integer offsets).

**Boolean mask:** A Series of `True`/`False` values used to filter rows — rows where the value is `True` are kept.

**Boolean indexing:** The technique of passing a boolean mask into `df[...]` to select only the rows that meet a condition.

**Assignment:** Adding a new column or updating the values in an existing column using `df["column"] = ...`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Loading data is just the beginning. In a dataset with hundreds of columns and millions of rows, you need precise tools to zoom in on exactly the rows and columns that matter. Pandas gives you two primary indexers — `loc` and `iloc` — as well as intuitive boolean filtering that lets you express conditions in plain, readable code.

Equally important is the ability to **assign** new data back to a DataFrame — adding a computed column, replacing missing values, or updating specific cells. Together, selecting and assigning are the core operations of data wrangling.

## Selecting Columns

### Single Column

Access a single column with bracket notation. This returns a **Series**:

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alex", "Jordan", "Morgan", "Sam", "Taylor"],
    "age":    [25, 30, 22, 28, 35],
    "city":   ["Chicago", "New York", "Austin", "Denver", "Seattle"],
    "score":  [88, 92, 75, 95, 81],
    "passed": [True, True, False, True, True]
})

print(df["name"])
```

Output:
```
0      Alex
1    Jordan
2    Morgan
3       Sam
4    Taylor
Name: name, dtype: object
```

### Multiple Columns

Pass a list of column names to get a **DataFrame** back:

```python
print(df[["name", "score"]])
```

Output:
```
     name  score
0    Alex     88
1  Jordan     92
2  Morgan     75
3     Sam     95
4  Taylor     81
```

## Selecting Rows

### `loc` — Label-Based Selection

`loc[row_label, column_label]` selects by **name**:

```python
# Select a single row by index label
print(df.loc[2])

# Select a range of rows (inclusive on both ends with loc)
print(df.loc[1:3])

# Select specific rows and columns
print(df.loc[0:2, ["name", "score"]])
```

Output of `df.loc[0:2, ["name", "score"]]`:
```
     name  score
0    Alex     88
1  Jordan     92
2  Morgan     75
```

### `iloc` — Position-Based Selection

`iloc[row_position, column_position]` selects by **integer position** (0-based):

```python
# Select the first row
print(df.iloc[0])

# Select rows 1 to 3 (exclusive end, like Python slicing)
print(df.iloc[1:3])

# Select rows 0-2, columns 0 and 2
print(df.iloc[0:3, [0, 2]])
```

Output of `df.iloc[0:3, [0, 2]]`:
```
     name      city
0    Alex   Chicago
1  Jordan  New York
2  Morgan    Austin
```

### `loc` vs `iloc` — Quick Comparison

| Feature | `loc` | `iloc` |
|---------|-------|--------|
| Selection by | Label / name | Integer position |
| Row range end | **Inclusive** | **Exclusive** |
| Use when | You know column/index names | You know positions |

## Filtering Rows with Boolean Indexing

Boolean indexing is one of the most powerful features in Pandas. You create a condition that returns a Series of `True`/`False` values (a **boolean mask**), then use it to filter the DataFrame.

### Single Condition

```python
# Keep only rows where score is greater than 85
high_scorers = df[df["score"] > 85]
print(high_scorers)
```

Output:
```
     name  age      city  score  passed
0    Alex   25   Chicago     88    True
1  Jordan   30  New York     92    True
3     Sam   28    Denver     95    True
```

### Multiple Conditions

Use `&` (and) or `|` (or) to combine conditions. **Always wrap each condition in parentheses:**

```python
# Score > 80 AND age < 30
result = df[(df["score"] > 80) & (df["age"] < 30)]
print(result)
```

Output:
```
   name  age     city  score  passed
0  Alex   25  Chicago     88    True
3   Sam   28   Denver     95    True
```

```python
# Age < 25 OR score > 90
result = df[(df["age"] < 25) | (df["score"] > 90)]
print(result)
```

### Filtering with `.isin()`

Check whether a column's values belong to a list:

```python
selected_cities = ["Chicago", "Denver"]
result = df[df["city"].isin(selected_cities)]
print(result)
```

Output:
```
   name  age     city  score  passed
0  Alex   25  Chicago     88    True
3   Sam   28   Denver     95    True
```

### Filtering with `.str` Methods

For string columns, Pandas provides a `.str` accessor:

```python
# Names that start with a specific letter
result = df[df["name"].str.startswith("A")]
print(result)   # Returns only Alex's row
```

## Assigning Data

### Adding a New Column

Assign a new column by simply specifying a name that doesn't yet exist:

```python
# Add a bonus column: 10% of score
df["bonus"] = df["score"] * 0.10
print(df[["name", "score", "bonus"]])
```

Output:
```
     name  score  bonus
0    Alex     88    8.8
1  Jordan     92    9.2
2  Morgan     75    7.5
3     Sam     95    9.5
4  Taylor     81    8.1
```

### Modifying an Existing Column

Assign to an existing column name to overwrite its values:

```python
# Round bonus to the nearest integer
df["bonus"] = df["bonus"].round(0).astype(int)
print(df["bonus"])
```

### Derived Columns with Conditions

Combine assignment with `np.where()` or boolean logic to create categorical columns:

```python
import numpy as np

df["result"] = np.where(df["score"] >= 80, "Pass", "Fail")
print(df[["name", "score", "result"]])
```

Output:
```
     name  score result
0    Alex     88   Pass
1  Jordan     92   Pass
2  Morgan     75   Fail
3     Sam     95   Pass
4  Taylor     81   Pass
```

### Using `.assign()` for Method Chaining

`.assign()` creates new columns without modifying the original DataFrame, making it ideal for building transformation pipelines:

```python
df_updated = (
    df
    .assign(score_pct=df["score"] / 100)
    .assign(grade=np.where(df["score"] >= 90, "A",
                  np.where(df["score"] >= 80, "B", "C")))
)
print(df_updated[["name", "score", "score_pct", "grade"]])
```

Output:
```
     name  score  score_pct grade
0    Alex     88       0.88     B
1  Jordan     92       0.92     A
2  Morgan     75       0.75     C
3     Sam     95       0.95     A
4  Taylor     81       0.81     B
```

## Conclusion

In this lesson, you learned how to select data from a DataFrame using column names, `loc` (label-based), and `iloc` (position-based) indexers. You also learned how to filter rows using boolean conditions with single and multiple criteria, and how to assign new or updated columns. These are the most frequently used operations in any data science workflow — you'll reach for them every time you need to examine, clean, or engineer features from a dataset. In the next lesson, you'll learn how to **map** and **apply** functions across rows and columns to transform data at scale.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What is the key difference between `loc` and `iloc`?**
1. `loc` works only on numeric columns; `iloc` works only on string columns.
2. `loc` selects by label (name); `iloc` selects by integer position.
3. `loc` returns a DataFrame; `iloc` always returns a Series.
4. `loc` is faster than `iloc` for large datasets.

**Correct Answer:**
2. `loc` selects by label (name); `iloc` selects by integer position.

**Explanation:**
`loc` uses the actual index labels and column names — so `df.loc[0, "score"]` finds the row labeled `0` and the column named `"score"`. `iloc` uses raw integer positions — `df.iloc[0, 3]` means the first row and fourth column, regardless of their names.

---

#### **Question 2: Why must you use parentheses around each condition when combining filters in Pandas?**
1. Parentheses are optional but improve readability.
2. Python's operator precedence evaluates `&` and `|` before comparison operators, so without parentheses the expression produces an error or wrong result.
3. Parentheses tell Pandas to use `loc` instead of `iloc`.
4. Pandas requires parentheses to distinguish column names from Python keywords.

**Correct Answer:**
2. Python's operator precedence evaluates `&` and `|` before comparison operators, so without parentheses the expression produces an error or wrong result.

**Explanation:**
In Python, bitwise operators `&` and `|` have higher precedence than comparison operators like `>` or `==`. Writing `df[df["score"] > 80 & df["age"] < 30]` is parsed as `df[df["score"] > (80 & df["age"]) < 30]`, which is incorrect. Parentheses force each condition to be evaluated first: `df[(df["score"] > 80) & (df["age"] < 30)]`.

---

#### **Question 3: What does the following code do?**
```python
df["category"] = np.where(df["score"] >= 90, "High", "Low")
```
1. Filters the DataFrame to keep only rows where `score >= 90`.
2. Sorts the DataFrame by the `score` column in descending order.
3. Creates a new column `"category"` with the value `"High"` where score is 90 or above, and `"Low"` otherwise.
4. Deletes all rows where `score < 90`.

**Correct Answer:**
3. Creates a new column `"category"` with the value `"High"` where score is 90 or above, and `"Low"` otherwise.

**Explanation:**
`np.where(condition, value_if_true, value_if_false)` is the NumPy equivalent of a vectorized if-else. It evaluates the condition row by row and assigns `"High"` or `"Low"` accordingly. The result is assigned to a new column `"category"` in the DataFrame.
