# Grouping and Sorting

## Overview

In the previous lesson, you learned how to transform data within a query using `CASE`, `COALESCE`, and SQL functions. Now you'll learn how to **summarize** data across groups — counting records, totaling sales, and averaging values — using `GROUP BY`. You'll also learn how to control the order of results with `ORDER BY` and filter aggregated groups with `HAVING`.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Use `GROUP BY` for aggregations (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`).
- Use `ORDER BY` for sorting query results.
- Filter aggregated results with `HAVING`.

## Key terms

**Aggregate function:** A function that computes a single summary value from multiple rows — for example, `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`.

**`GROUP BY`:** A clause that splits rows into groups based on unique values of one or more columns, then applies an aggregate function to each group.

**`ORDER BY`:** A clause that sorts query results by one or more columns. `ASC` (ascending) is the default; `DESC` sorts in descending order.

**`HAVING`:** A clause that filters groups **after** aggregation — the equivalent of `WHERE` for grouped results. Use `WHERE` to filter rows before grouping, and `HAVING` to filter groups after.

**`COUNT(*)`:** Counts all rows in a group, including those with `NULL` values.

**`COUNT(column)`:** Counts only rows where the specified column is not `NULL`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Individual rows tell you about specific events. Groups tell you about patterns. How many orders did each customer place? Which genre generates the most revenue? Which month had the highest sales? These questions require **aggregation** — collapsing many rows into a single summary value per group.

The `GROUP BY` clause is how SQL performs these split-apply-combine operations, and `ORDER BY` lets you rank the results so the most important groups appear first.

## `ORDER BY` — Sorting Results

`ORDER BY` sorts query results by one or more columns:

```sql
-- All books sorted by price, cheapest first (ASC is the default)
SELECT title, price FROM books ORDER BY price ASC;

-- Most expensive first
SELECT title, price FROM books ORDER BY price DESC;
```

### Sorting by Multiple Columns

```sql
-- Sort by genre alphabetically, then by price within each genre (highest first)
SELECT title, genre, price
FROM books
ORDER BY genre ASC, price DESC;
```

Output:
```
title                       | genre        | price
----------------------------+--------------+-------
The Data Detective          | Data Science | 18.50
Storytelling with Data      | Data Science | 29.99
Deep Learning               | Programming  | 74.99
Python Crash Course         | Programming  | 35.99
How to Lie with Statistics  | Statistics   | 12.99
Naked Statistics            | Statistics   | 16.99
```

### `ORDER BY` with `LIMIT`

Combining `ORDER BY` and `LIMIT` gives you the "top N" pattern:

```sql
-- The 3 most expensive books
SELECT title, price FROM books ORDER BY price DESC LIMIT 3;
```

```sql
-- The most recent order
SELECT * FROM orders ORDER BY order_date DESC LIMIT 1;
```

## Aggregate Functions

Before using `GROUP BY`, it's useful to see aggregate functions applied to an entire table (without grouping):

```sql
-- Summary of the books table
SELECT
    COUNT(*)        AS total_books,
    COUNT(genre)    AS books_with_genre,
    AVG(price)      AS avg_price,
    MIN(price)      AS cheapest,
    MAX(price)      AS most_expensive,
    SUM(stock)      AS total_stock
FROM books;
```

Output:
```
total_books | books_with_genre | avg_price | cheapest | most_expensive | total_stock
------------+------------------+-----------+----------+----------------+------------
6           | 6                | 31.24     | 12.99    | 74.99          | 130
```

## `GROUP BY` — Aggregating by Group

`GROUP BY` divides the rows into groups and applies an aggregate function to each:

```sql
-- Total orders and average quantity per customer
SELECT
    customer_id,
    COUNT(*)        AS order_count,
    SUM(quantity)   AS total_units,
    AVG(quantity)   AS avg_quantity
FROM orders
GROUP BY customer_id;
```

Output:
```
customer_id | order_count | total_units | avg_quantity
------------+-------------+-------------+-------------
1           | 3           | 3           | 1.0
2           | 2           | 2           | 1.0
3           | 2           | 3           | 1.5
4           | 2           | 4           | 2.0
5           | 1           | 1           | 1.0
```

### Including Non-Aggregated Columns

Every column in the `SELECT` list must either be in the `GROUP BY` clause or wrapped in an aggregate function:

```sql
-- Revenue per genre
SELECT
    genre,
    COUNT(*)            AS book_count,
    ROUND(AVG(price), 2) AS avg_price,
    SUM(stock)          AS total_stock
FROM books
GROUP BY genre
ORDER BY avg_price DESC;
```

Output:
```
genre        | book_count | avg_price | total_stock
-------------+------------+-----------+------------
Programming  | 2          | 55.49     | 17
Data Science | 2          | 24.25     | 33
Statistics   | 2          | 14.99     | 80
```

### Grouping by Multiple Columns

```sql
-- Orders per customer per month
SELECT
    customer_id,
    strftime('%Y-%m', order_date) AS month,
    COUNT(*)                      AS orders_that_month
FROM orders
GROUP BY customer_id, month
ORDER BY customer_id, month;
```

## `HAVING` — Filtering Groups

`WHERE` filters rows **before** grouping. `HAVING` filters groups **after** aggregation:

```sql
-- Only show customers who placed more than 1 order
SELECT
    customer_id,
    COUNT(*) AS order_count
FROM orders
GROUP BY customer_id
HAVING order_count > 1
ORDER BY order_count DESC;
```

Output:
```
customer_id | order_count
------------+-------------
1           | 3
2           | 2
3           | 2
4           | 2
```

Customer 5 (only 1 order) is excluded.

### `WHERE` vs `HAVING`

```sql
-- WHERE filters rows BEFORE grouping (e.g., exclude small orders first)
-- HAVING filters groups AFTER aggregation (e.g., only groups with high totals)

SELECT
    customer_id,
    SUM(quantity)   AS total_units
FROM orders
WHERE quantity >= 2                     -- Row filter: only include orders of 2+ units
GROUP BY customer_id
HAVING total_units >= 3                 -- Group filter: only customers with 3+ total units
ORDER BY total_units DESC;
```

## Putting It Together: Sales Analysis

Here's a full analysis query that combines joins (covered in the next lesson), grouping, and sorting:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bookstore.db")

query = """
    SELECT
        b.genre,
        COUNT(o.order_id)               AS num_orders,
        SUM(o.quantity)                 AS units_sold,
        ROUND(SUM(o.quantity * b.price), 2) AS total_revenue,
        ROUND(AVG(b.price), 2)          AS avg_book_price
    FROM orders AS o
    JOIN books AS b ON o.book_id = b.book_id
    GROUP BY b.genre
    ORDER BY total_revenue DESC;
"""

df = pd.read_sql_query(query, conn)
print(df)
conn.close()
```

Output:
```
        genre  num_orders  units_sold  total_revenue  avg_book_price
0  Programming           3           3         146.97           55.49
1  Data Science           4           5         113.48           24.25
2   Statistics            3           6          74.94           14.99
```

## The Full `SELECT` Clause Order

SQL clauses must appear in this order:

```sql
SELECT   columns / aggregates
FROM     table
WHERE    row_filter            -- runs before grouping
GROUP BY grouping_columns
HAVING   group_filter          -- runs after grouping
ORDER BY sort_columns
LIMIT    n;
```

## Conclusion

In this lesson, you learned how to sort query results with `ORDER BY`, compute summary statistics per group using `GROUP BY` with `COUNT`, `SUM`, `AVG`, `MIN`, and `MAX`, and filter aggregated groups with `HAVING`. The distinction between `WHERE` (row-level filter) and `HAVING` (group-level filter) is one of the most important concepts in SQL. In the next lesson, you'll learn how to combine data from multiple tables using **joins** — the feature that makes the relational model truly powerful.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What is the difference between `WHERE` and `HAVING`?**
1. `WHERE` filters columns; `HAVING` filters rows.
2. `WHERE` filters rows before grouping; `HAVING` filters groups after aggregation.
3. `WHERE` is used with `GROUP BY`; `HAVING` is used without `GROUP BY`.
4. There is no difference — they are interchangeable.

**Correct Answer:**
2. `WHERE` filters rows before grouping; `HAVING` filters groups after aggregation.

**Explanation:**
`WHERE` applies to individual rows before the `GROUP BY` step — rows that don't match are excluded from grouping entirely. `HAVING` applies to the groups produced by `GROUP BY` and can reference aggregate functions like `COUNT(*)` or `SUM(price)`. You cannot use aggregate functions in a `WHERE` clause.

---

#### **Question 2: What does `COUNT(*)` count versus `COUNT(column_name)`?**
1. `COUNT(*)` counts only non-`NULL` rows; `COUNT(column)` counts all rows including `NULL`s.
2. `COUNT(*)` counts all rows in the group including `NULL` values; `COUNT(column)` counts only rows where that specific column is not `NULL`.
3. They are identical — both count all rows.
4. `COUNT(*)` is only valid without `GROUP BY`; `COUNT(column)` is only valid with `GROUP BY`.

**Correct Answer:**
2. `COUNT(*)` counts all rows in the group including `NULL` values; `COUNT(column)` counts only rows where that specific column is not `NULL`.

**Explanation:**
`COUNT(*)` counts every row in the group, regardless of `NULL` values. `COUNT(column_name)` skips rows where that column is `NULL`. This distinction matters when a column has missing values — `COUNT(*)` and `COUNT(column)` will return different numbers, which can be informative about data completeness.

---

#### **Question 3: The following query produces an error. Why?**
```sql
SELECT customer_id, COUNT(*) AS order_count
FROM orders
WHERE order_count > 2
GROUP BY customer_id;
```
1. `COUNT(*)` cannot be used with `GROUP BY`.
2. `WHERE` cannot reference an alias defined in `SELECT`.
3. Aggregate functions like `COUNT(*)` cannot be used in a `WHERE` clause — use `HAVING` instead.
4. Both B and C are correct reasons.

**Correct Answer:**
4. Both B and C are correct reasons.

**Explanation:**
There are two problems. First, `WHERE order_count > 2` references the alias `order_count`, but aliases defined in `SELECT` are not available in `WHERE` (the `WHERE` clause is evaluated before `SELECT`). Second, aggregate functions cannot appear in `WHERE` — `WHERE COUNT(*) > 2` is invalid SQL. The correct clause to filter aggregated results is `HAVING`: `GROUP BY customer_id HAVING COUNT(*) > 2`.
