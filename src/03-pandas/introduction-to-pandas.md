# Introduction to Pandas

## Overview

Welcome to the Manipulating Data with Pandas module! In the previous module, you built a solid foundation in Python — variables, functions, data structures, and objects. Now you'll put those skills to work with one of the most important libraries in data science: **Pandas**. In this lesson, you'll learn what Pandas is, why it's the go-to tool for working with tabular data, and how to use its two core data structures.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Understand what Pandas is and why it's used in data analysis.
- Import Pandas and explore core data structures (Series, DataFrame).

## Key terms

**Pandas:** An open-source Python library that provides fast, flexible data structures and analysis tools for working with structured (tabular) data.

**Series:** A one-dimensional labeled array in Pandas, similar to a single column in a spreadsheet. Each value has an associated index label.

**DataFrame:** A two-dimensional labeled data structure in Pandas, similar to a spreadsheet or SQL table. It consists of rows and columns, where each column is a Series.

**Index:** The labels that identify each row in a Series or DataFrame. By default, Pandas assigns integer indices starting at 0.

**`shape`:** A DataFrame attribute that returns a tuple `(rows, columns)` describing the dimensions of the DataFrame.

**`dtypes`:** A DataFrame attribute that shows the data type of each column.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

When you work with real-world data — sales records, patient data, sports statistics — that data almost always comes in a tabular format: rows and columns, like a spreadsheet. Python's built-in lists and dictionaries can technically store this kind of data, but they weren't designed for it. Querying, filtering, aggregating, and transforming rows and columns quickly becomes cumbersome.

[*Pandas*](https://pandas.pydata.org/) solves this problem. Built on top of NumPy, Pandas gives you powerful, expressive tools to load, explore, clean, and transform tabular data with minimal code. Whether you're reading a CSV file, filtering rows based on a condition, or computing summary statistics across groups, Pandas handles it elegantly. It is the foundation of nearly every data science workflow in Python.

## Installing and Importing Pandas

Pandas comes pre-installed in **Google Colab**, so you can start using it right away. If you're working in a local environment, install it with:

```bash
pip install pandas
```

Import Pandas at the top of your notebook or script. The conventional alias is `pd`:

```python
import pandas as pd
import numpy as np  # Often used alongside Pandas
```

After this import, you access Pandas features using the `pd.` prefix.

## The Series

A **Series** is the simplest Pandas data structure — a one-dimensional array with labels (an index) for each value.

### Creating a Series

```python
# From a list
temperatures = pd.Series([72, 75, 68, 80, 77])
print(temperatures)
```

Output:
```
0    72
1    75
2    68
3    80
4    77
dtype: int64
```

The left column is the **index**, the right column is the **data**.

### Custom Index Labels

```python
scores = pd.Series(
    [88, 92, 75, 95],
    index=["Alex", "Jordan", "Morgan", "Sam"]
)
print(scores)
```

Output:
```
Alex      88
Jordan    92
Morgan    75
Sam       95
dtype: int64
```

### Accessing Series Values

```python
print(scores["Alex"])     # Output: 88
print(scores[0])          # Output: 88 (positional access)
print(scores[scores > 85])  # Filter: values greater than 85
```

Output of the filter:
```
Alex      88
Jordan    92
Sam       95
dtype: int64
```

## The DataFrame

A **DataFrame** is the centerpiece of Pandas — a two-dimensional table where each column is a Series sharing the same index.

### Creating a DataFrame from a Dictionary

The most common way to build a DataFrame manually is from a dictionary of lists, where each key becomes a column name:

```python
data = {
    "name":    ["Alex", "Jordan", "Morgan", "Sam", "Taylor"],
    "age":     [25, 30, 22, 28, 35],
    "city":    ["Chicago", "New York", "Austin", "Denver", "Seattle"],
    "score":   [88, 92, 75, 95, 81]
}

df = pd.DataFrame(data)
print(df)
```

Output:
```
     name  age      city  score
0    Alex   25   Chicago     88
1  Jordan   30  New York     92
2  Morgan   22    Austin     75
3     Sam   28    Denver     95
4  Taylor   35   Seattle     81
```

Each row has a numeric index (0–4) by default, and each column has a name from the dictionary key.

## Exploring a DataFrame

Once you have a DataFrame, these methods and attributes help you understand what's inside.

### `.head()` and `.tail()`

View the first or last few rows (default: 5):

```python
print(df.head(3))   # First 3 rows
print(df.tail(2))   # Last 2 rows
```

### `.shape`

Returns `(number_of_rows, number_of_columns)`:

```python
print(df.shape)   # Output: (5, 4)
```

### `.dtypes`

Shows the data type of each column:

```python
print(df.dtypes)
```

Output:
```
name     object
age       int64
city     object
score     int64
dtype: object
```

### `.info()`

A concise summary including column names, data types, and non-null counts:

```python
df.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   name    5 non-null      object
 1   age     5 non-null      int64
 2   city    5 non-null      object
 3   score   5 non-null      int64
dtypes: int64(2), object(2)
memory usage: 292.0+ bytes
```

### `.describe()`

Computes summary statistics for all numeric columns:

```python
print(df.describe())
```

Output:
```
             age      score
count   5.000000   5.000000
mean   28.000000  86.200000
std     4.949747   7.791020
min    22.000000  75.000000
25%    25.000000  81.000000
50%    28.000000  88.000000
75%    30.000000  92.000000
max    35.000000  95.000000
```

### Accessing Columns

Access a single column (returns a Series) using bracket or dot notation:

```python
print(df["name"])      # Bracket notation — works for all column names
print(df.score)        # Dot notation — only works when column name has no spaces
```

Access multiple columns by passing a list:

```python
print(df[["name", "score"]])
```

## Conclusion

In this lesson, you were introduced to Pandas and its two core data structures: the **Series** (a labeled one-dimensional array) and the **DataFrame** (a labeled two-dimensional table). You learned how to create each structure, access values, and use essential exploratory methods like `.head()`, `.shape`, `.dtypes`, `.info()`, and `.describe()`. These tools form the foundation of every Pandas workflow. In the next lesson, you'll learn how to create DataFrames from real data by reading files, and how to write processed data back to disk.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="A **Series** is like a single spreadsheet column — one sequence of values with labels. A **DataFrame** is like a full spreadsheet — multiple columns (each a Series) sharing the same row index. In practice, a DataFrame column is a Series.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the difference between a Pandas Series and a DataFrame?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>A Series stores text data; a DataFrame stores numeric data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>A Series is one-dimensional with an index; a DataFrame is two-dimensional with labeled rows and columns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>A Series can hold multiple data types; a DataFrame can only hold one.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>There is no difference — they are interchangeable.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="`df.shape` returns a tuple like `(100, 5)`, meaning the DataFrame has 100 rows and 5 columns. This is a quick way to understand the size of your dataset before diving into analysis.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does `df.shape` return?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The data types of each column.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The first five rows of the DataFrame.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>A tuple of (number of rows, number of columns).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The total number of non-null values.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="3" data-explanation="`df.describe()` computes summary statistics — count, mean, standard deviation, min, quartiles, and max — for every numeric column. It&#039;s one of the first methods you should call when exploring a new dataset. `df.info()` shows column types and null counts, not statistical summaries.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Which method provides a quick statistical summary (mean, min, max, std) of all numeric columns in a DataFrame?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>`df.info()`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>`df.head()`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>`df.dtypes`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>`df.describe()`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

