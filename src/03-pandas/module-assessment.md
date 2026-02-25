# Module Assessment

## Overview

Congratulations on completing the Manipulating Data with Pandas module! You've learned how to load data from files, explore DataFrames, select and filter rows and columns, transform values with mapping functions, and summarize data with grouping and sorting. This assessment brings all of those skills together in a realistic data wrangling project.

## Learning Objective

By the end of this assessment, you will have demonstrated your ability to:

- Perform data wrangling tasks on a sample dataset.
- Demonstrate ability to read, transform, and output results.

## Key terms

**Data wrangling:** The process of cleaning, reshaping, and transforming raw data into a format suitable for analysis. Also called data munging.

**Pipeline:** A sequence of data transformation steps applied in order, where the output of each step becomes the input of the next.

**Aggregated output:** A summary result derived from grouping raw records, such as totals, averages, or counts per category.

## Assessment Overview

This module assessment is a coding project completed in a Google Colaboratory notebook. You will be given a raw CSV dataset and complete a series of tasks that exercise the full Pandas workflow — from loading and exploring the data to cleaning, transforming, and summarizing it.

### Skills Assessed

| Skill | Lesson |
|-------|--------|
| Loading data from a CSV file | Creating, Reading, and Writing |
| Exploring DataFrame structure | Introduction to Pandas |
| Selecting columns and filtering rows | Selecting and Assigning |
| Creating derived columns | Selecting and Assigning |
| Transforming values with `.map()` and `.apply()` | Mapping Data |
| Grouping and aggregating | Grouping and Sorting |
| Sorting and ranking results | Grouping and Sorting |
| Writing output to a CSV file | Creating, Reading, and Writing |

## Coding Assessment

Complete the project in this [notebook](#). The notebook contains a series of guided tasks. After completing all tasks, save your notebook to GitHub and submit the link for grading.

### Task Overview

The notebook will walk you through the following steps using a provided retail sales dataset:

1. **Load and inspect** — Read the CSV file into a DataFrame. Display the shape, column types, and first 10 rows. Identify any missing values with `.isnull().sum()`.

2. **Select and explore** — Select only the columns relevant to the analysis. Filter rows to keep only records from the last two years. Use `.describe()` to summarize the numeric columns.

3. **Clean and assign** — Standardize a string column (trim whitespace and apply `.title()` case). Use `.map()` or `np.where()` to create a new categorical column based on a numeric threshold.

4. **Transform with `.apply()`** — Write a function and apply it row-wise (`axis=1`) to derive a new column that depends on values from two or more existing columns.

5. **Group and aggregate** — Use `groupby()` and `.agg()` to produce a summary table showing total sales, average order value, and transaction count for each product category.

6. **Sort and rank** — Sort the summary table to identify the top-performing categories. Use `.nlargest()` to extract the top 5 individual transactions.

7. **Write output** — Save the summary table and the cleaned full dataset to separate CSV files using `to_csv()`.

### Grading Rubric

| Criteria | Points |
|----------|--------|
| Data loaded correctly and explored with appropriate methods | 10 |
| Correct columns selected; rows filtered without errors | 15 |
| String cleaning and new categorical column created correctly | 15 |
| Row-wise `.apply()` function produces correct results | 15 |
| `groupby().agg()` summary is accurate and readable | 20 |
| Sort and ranking steps are correct | 10 |
| Output files saved with `index=False` | 5 |
| Code is clean, readable, and uses descriptive variable names | 10 |
| **Total** | **100** |

## Knowledge Check

#### **Question 1: Which Pandas method is most appropriate for quickly detecting missing values across all columns?**
1. `df.describe()`
2. `df.isnull().sum()`
3. `df.dropna()`
4. `df.info()`

**Correct Answer:**
2. `df.isnull().sum()`

**Explanation:**
`df.isnull()` returns a boolean DataFrame where `True` indicates a missing value. Chaining `.sum()` counts the `True` values per column, giving you the number of missing entries in each column at a glance. `df.info()` also shows non-null counts, but `isnull().sum()` is more direct for a column-by-column missing value count.

---

#### **Question 2: A teammate wrote the following code. What is the most likely problem?**
```python
df["discount"] = df.apply(lambda x: x["price"] * 0.1)
```
1. `lambda` functions cannot be used with `.apply()`.
2. The `axis` parameter is missing — without `axis=1`, Pandas applies the function column-by-column, not row-by-row, causing an error.
3. `.apply()` cannot access dictionary-style keys like `x["price"]`.
4. The result of the lambda should use `return` instead of a direct expression.

**Correct Answer:**
2. The `axis` parameter is missing — without `axis=1`, Pandas applies the function column-by-column, not row-by-row, causing an error.

**Explanation:**
When using `.apply()` on a DataFrame to access individual column values within each row (using `row["column_name"]`), you must specify `axis=1`. Without it, the default `axis=0` passes each **column** as a Series to the function, meaning `x["price"]` tries to index a column Series by the string `"price"`, which will raise a `KeyError`. The correct call is `df.apply(lambda row: row["price"] * 0.1, axis=1)`.

---

#### **Question 3: After running `df.groupby("category")["revenue"].sum()`, the result has `category` as the index. What is the standard next step to make the result easier to work with?**
1. Call `.sort_values()` to sort alphabetically.
2. Call `.reset_index()` to convert the index back into a regular column.
3. Call `.transpose()` to swap rows and columns.
4. Call `.rename()` to give the index a new name.

**Correct Answer:**
2. Call `.reset_index()` to convert the index back into a regular column.

**Explanation:**
After a `groupby()` operation, the grouping column(s) become the index. `.reset_index()` promotes the index back to a regular column and assigns a default integer index (0, 1, 2, ...). This makes the result a clean, standard DataFrame that can be used seamlessly in further operations like merging, sorting, or saving to CSV.
