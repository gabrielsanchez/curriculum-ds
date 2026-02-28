# Module Assessment

## Overview

Congratulations on completing the Manipulating Data with Pandas module! You've learned how to load data from files, explore DataFrames, select and filter rows and columns, transform values with mapping functions, and summarize data with grouping and sorting. This assessment brings all of those skills together in a realistic data wrangling project.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/03-pandas/03-pandas-module-assessment_starter.ipynb).

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

Complete the project in this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/03-pandas/03-pandas-module-assessment_starter.ipynb). The notebook contains a series of guided tasks. After completing all tasks, save your notebook to GitHub and [submit the link for grading](https://ai-grader-pql9.onrender.com/).

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

<div class="quiz-container" data-correct="1" data-explanation="`df.isnull()` returns a boolean DataFrame where `True` indicates a missing value. Chaining `.sum()` counts the `True` values per column, giving you the number of missing entries in each column at a glance. `df.info()` also shows non-null counts, but `isnull().sum()` is more direct for a column-by-column missing value count.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Which Pandas method is most appropriate for quickly detecting missing values across all columns?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`df.describe()`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`df.isnull().sum()`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`df.dropna()`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>`df.info()`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When using `.apply()` on a DataFrame to access individual column values within each row (using `row[&quot;column_name&quot;]`), you must specify `axis=1`. Without it, the default `axis=0` passes each **column** as a Series to the function, meaning `x[&quot;price&quot;]` tries to index a column Series by the string `&quot;price&quot;`, which will raise a `KeyError`. The correct call is `df.apply(lambda row: row[&quot;price&quot;] * 0.1, axis=1)`.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A teammate wrote the following code. What is the most likely problem?
  </div>
  <div class="quiz-subquestion">
    <pre><code>df[&quot;discount&quot;] = df.apply(lambda x: x[&quot;price&quot;] * 0.1)</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`lambda` functions cannot be used with `.apply()`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The `axis` parameter is missing — without `axis=1`, Pandas applies the function column-by-column, not row-by-row, causing an error.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`.apply()` cannot access dictionary-style keys like `x["price"]`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The result of the lambda should use `return` instead of a direct expression.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="After a `groupby()` operation, the grouping column(s) become the index. `.reset_index()` promotes the index back to a regular column and assigns a default integer index (0, 1, 2, ...). This makes the result a clean, standard DataFrame that can be used seamlessly in further operations like merging, sorting, or saving to CSV.">
  <div class="quiz-question">
    <strong>Question 3:</strong> After running `df.groupby("category")["revenue"].sum()`, the result has `category` as the index. What is the standard next step to make the result easier to work with?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Call `.sort_values()` to sort alphabetically.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Call `.reset_index()` to convert the index back into a regular column.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Call `.transpose()` to swap rows and columns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Call `.rename()` to give the index a new name.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

