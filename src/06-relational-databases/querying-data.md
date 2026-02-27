# Querying Data

## Overview

In the previous lesson, you defined the structure of a database — tables, data types, and constraints. Now it's time to read data back out. In this lesson, you'll learn the `SELECT` statement, the most important command in SQL, and how to use `WHERE` clauses to retrieve exactly the rows you need.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Write `SELECT` queries to retrieve data.
- Filter results with `WHERE` clauses.

## Key terms

**`SELECT`:** The SQL clause that specifies which columns to retrieve from a query.

**`FROM`:** The SQL clause that specifies the table to query.

**`WHERE`:** The SQL clause that filters rows, returning only those that satisfy a given condition.

**`*` (wildcard):** Used in `SELECT *` to retrieve all columns from a table.

**`DISTINCT`:** A keyword that removes duplicate rows from query results.

**`LIMIT`:** A clause that restricts the number of rows returned.

**`LIKE`:** A comparison operator that matches text patterns using `%` (any sequence of characters) and `_` (any single character) as wildcards.

**`IN`:** A comparison operator that checks whether a value matches any value in a list.

**`BETWEEN`:** A comparison operator that checks whether a value falls within an inclusive range.

**`IS NULL` / `IS NOT NULL`:** Operators for checking whether a value is missing.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/04_querying-data_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

The `SELECT` statement is the workhorse of SQL. In data science, the vast majority of your SQL work will involve reading data — selecting specific columns, filtering rows by conditions, and combining results across tables. Mastering `SELECT` and `WHERE` gives you the foundation for all of the more advanced queries you'll learn in subsequent lessons.

All examples in this lesson use the bookstore database built in the previous lesson.

## Setup

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()
```

## The `SELECT` Statement

### Selecting All Columns

Use `SELECT *` to return every column from a table:

```sql
SELECT * FROM customers;
```

```python
cursor.execute("SELECT * FROM customers;")
rows = cursor.fetchall()
for row in rows:
    print(row)
```

Output:
```
(1, 'Alex Carter',   'alex@email.com',   'Chicago',  '2022-03-10')
(2, 'Jordan Lee',    'jordan@email.com',  'New York', '2021-11-05')
(3, 'Morgan Singh',  'morgan@email.com',  'Austin',   '2023-01-20')
(4, 'Sam Torres',    'sam@email.com',     'Chicago',  '2022-07-15')
(5, 'Taylor Reyes',  'taylor@email.com',  'Denver',   '2023-06-01')
```

### Selecting Specific Columns

List the column names you want, separated by commas:

```sql
SELECT name, city FROM customers;
```

Selecting only the columns you need is good practice — it reduces the data transferred and makes results easier to read.

### Column Aliases with `AS`

Rename a column in the result using `AS`:

```sql
SELECT name AS customer_name, city AS location FROM customers;
```

Output:
```
customer_name  | location
---------------+----------
Alex Carter    | Chicago
Jordan Lee     | New York
...
```

### `DISTINCT` — Remove Duplicates

Return only unique values:

```sql
SELECT DISTINCT city FROM customers;
```

Output:
```
Chicago
New York
Austin
Denver
```

### `LIMIT` — Restrict Row Count

Return only the first N rows:

```sql
SELECT * FROM books LIMIT 3;
```

`LIMIT` is especially useful when exploring an unfamiliar table to see its structure without loading thousands of rows.

## Filtering with `WHERE`

The `WHERE` clause filters rows based on a condition. Only rows where the condition is `TRUE` are included.

### Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equal to | `city = 'Chicago'` |
| `!=` or `<>` | Not equal to | `genre != 'Statistics'` |
| `>` | Greater than | `price > 20` |
| `<` | Less than | `stock < 10` |
| `>=` | Greater than or equal | `price >= 29.99` |
| `<=` | Less than or equal | `rating <= 3.0` |

```sql
SELECT title, price FROM books WHERE price > 20;
```

Output:
```
title                  | price
-----------------------+-------
Python Crash Course    | 35.99
Storytelling with Data | 29.99
Deep Learning          | 74.99
```

### `AND` and `OR`

Combine multiple conditions:

```sql
-- Books that cost more than $20 AND are in the Programming genre
SELECT title, genre, price
FROM books
WHERE price > 20 AND genre = 'Programming';
```

```sql
-- Books that are either Statistics or Data Science
SELECT title, genre
FROM books
WHERE genre = 'Statistics' OR genre = 'Data Science';
```

### `NOT`

Exclude rows matching a condition:

```sql
SELECT title, genre FROM books WHERE NOT genre = 'Statistics';
```

## Pattern Matching with `LIKE`

`LIKE` matches text patterns. Use `%` to match any sequence of characters and `_` to match exactly one character:

```sql
-- Books whose title contains the word "Data"
SELECT title FROM books WHERE title LIKE '%Data%';
```

Output:
```
The Data Detective
Storytelling with Data
```

```sql
-- Customers whose email ends in "@email.com"
SELECT name, email FROM customers WHERE email LIKE '%@email.com';
```

`LIKE` is case-insensitive in SQLite by default for ASCII characters.

## Membership with `IN`

Check whether a value matches any item in a list — cleaner than multiple `OR` conditions:

```sql
-- Customers in Chicago or Austin
SELECT name, city FROM customers WHERE city IN ('Chicago', 'Austin');
```

Output:
```
name         | city
-------------+---------
Alex Carter  | Chicago
Morgan Singh | Austin
Sam Torres   | Chicago
```

```sql
-- Books NOT in these genres
SELECT title, genre FROM books WHERE genre NOT IN ('Statistics', 'Programming');
```

## Range Filtering with `BETWEEN`

Check whether a value falls within an inclusive range:

```sql
-- Books priced between $15 and $40
SELECT title, price FROM books WHERE price BETWEEN 15 AND 40;
```

`BETWEEN a AND b` is equivalent to `>= a AND <= b`.

## Handling Missing Values with `IS NULL`

You cannot use `= NULL` to find missing values — you must use `IS NULL` or `IS NOT NULL`:

```sql
-- Customers with no city recorded
SELECT name FROM customers WHERE city IS NULL;

-- Books where genre has been filled in
SELECT title, genre FROM books WHERE genre IS NOT NULL;
```

## Loading Query Results into Pandas

In a data science workflow, you'll often want to run a SQL query and then continue working in Pandas:

```python
query = """
    SELECT title, genre, price, stock
    FROM books
    WHERE price < 30
    ORDER BY price DESC;
"""

df = pd.read_sql_query(query, conn)
print(df)
```

Output:
```
                     title        genre  price  stock
0  Storytelling with Data   Data Science  29.99      8
1       Python Crash Course  Programming  35.99     12  ← wait, > 30
2         The Data Detective   Data Science  18.50     25
3             Naked Statistics  Statistics  16.99     30
4  How to Lie with Statistics  Statistics  12.99     50
```

`pd.read_sql_query()` executes the SQL and returns a DataFrame directly — no manual row-fetching needed. This is the most common pattern for SQL + Python in data science.

## The Logical Order of Clauses

A `SELECT` statement's clauses must appear in this order:

```sql
SELECT   columns
FROM     table
WHERE    row_filter
ORDER BY sort_column     -- covered in Grouping and Sorting lesson
LIMIT    n;
```

## Conclusion

In this lesson, you learned how to write `SELECT` queries to retrieve specific columns, use `WHERE` to filter rows with comparison operators, combine conditions with `AND`/`OR`/`NOT`, and match patterns with `LIKE`, `IN`, `BETWEEN`, and `IS NULL`. You also saw how `pd.read_sql_query()` bridges SQL and Pandas for a smooth data science workflow. In the next lesson, you'll step up to **advanced queries** — using subqueries, aliases, `CASE` expressions, and SQL functions to answer more complex questions.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/04_querying-data_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What is the difference between `SELECT *` and `SELECT title, price`?**
1. `SELECT *` is faster; `SELECT title, price` is slower.
2. `SELECT *` returns all columns; `SELECT title, price` returns only those two named columns.
3. `SELECT *` is only valid without a `WHERE` clause.
4. There is no difference — SQL ignores column names in the `SELECT` clause.

**Correct Answer:**
2. `SELECT *` returns all columns; `SELECT title, price` returns only those two named columns.

**Explanation:**
`SELECT *` is a shorthand that returns every column from the specified table. `SELECT title, price` returns only those two named columns. In practice, naming specific columns is preferred — it's explicit, returns less data, and is more resilient to schema changes that add new columns.

---

<div class="quiz-container" data-correct="1" data-explanation="In SQL, `NULL` means &quot;unknown.&quot; Any comparison involving `NULL` — including `NULL = NULL` — evaluates to `NULL`, which is treated as false in a `WHERE` clause. This is why SQL provides the special `IS NULL` and `IS NOT NULL` operators specifically for testing for missing values.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Why can't you use `WHERE city = NULL` to find rows with a missing city?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`NULL` is a reserved keyword that cannot appear after `=`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`NULL` represents the absence of a value — comparing anything to `NULL` with `=` always evaluates to `NULL` (not `TRUE`), so no rows are returned. Use `IS NULL` instead.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>You need to use `WHERE city == NULL` (double equals) to find NULL values.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>SQLite automatically converts `NULL` to an empty string, so you should use `WHERE city = ''`.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`IN` checks whether the column&#039;s value matches **any** item in the list. `WHERE genre IN (&#039;Statistics&#039;, &#039;Data Science&#039;)` is equivalent to `WHERE genre = &#039;Statistics&#039; OR genre = &#039;Data Science&#039;`. It returns rows matching either value — not both at once. To exclude these genres, you would use `NOT IN`.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does `WHERE genre IN ('Statistics', 'Data Science')` do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It filters rows where genre equals both 'Statistics' and 'Data Science' simultaneously.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It filters rows where genre is either 'Statistics' or 'Data Science'.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It excludes rows where genre is 'Statistics' or 'Data Science'.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It creates a new column containing both genre values.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

