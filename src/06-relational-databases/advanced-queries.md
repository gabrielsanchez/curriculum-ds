# Advanced Queries

## Overview

In the previous lesson, you learned how to retrieve and filter data using `SELECT` and `WHERE`. Those queries answer straightforward questions: "Show me all books under $20." But real data questions are rarely that simple. In this lesson, you'll learn how to write more expressive queries using subqueries, column aliases, `CASE` expressions, `COALESCE`, and SQL's built-in functions to transform and classify data directly in the database.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Use subqueries and aliases to structure complex queries.
- Perform transformations and conditional logic using `CASE` and `COALESCE`.

## Key terms

**Subquery:** A `SELECT` statement nested inside another SQL statement. The inner query runs first and its result is used by the outer query.

**Alias (`AS`):** A temporary name given to a table or column in a query to make results more readable or to reference a subquery.

**`CASE WHEN`:** SQL's conditional expression — equivalent to an if-elif-else block. Evaluates conditions in order and returns the first matching result.

**`COALESCE()`:** A function that returns the first non-`NULL` value from a list of arguments. Used to replace `NULL` with a default value.

**`ROUND()`:** A function that rounds a numeric value to a specified number of decimal places.

**`LENGTH()`:** A function that returns the number of characters in a text string.

**`UPPER()` / `LOWER()`:** Functions that convert text to uppercase or lowercase.

**`strftime()`:** SQLite's function for formatting and extracting parts of dates stored as text.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

The `SELECT` and `WHERE` clauses you learned in the previous lesson are powerful, but they only retrieve data as-is. Advanced queries let you **transform** data while querying it — classifying values into categories, replacing missing data with defaults, computing derived values, and filtering based on aggregated results. These techniques reduce the amount of post-processing you need to do in Python, keeping your data pipelines clean and efficient.

## Column Aliases

You've already seen `AS` used to rename a column in results. Aliases also make complex expressions readable:

```sql
SELECT
    title,
    price,
    price * 0.9         AS discounted_price,
    stock * price       AS inventory_value
FROM books;
```

Output:
```
title                      | price | discounted_price | inventory_value
---------------------------+-------+------------------+----------------
Python Crash Course        | 35.99 |           32.391 |          431.88
The Data Detective         | 18.50 |           16.650 |          462.50
...
```

Aliases only exist within the query — they are not stored in the table.

## `CASE WHEN` — Conditional Logic

`CASE WHEN` is SQL's if-elif-else. It evaluates conditions in order and returns the value of the first matching branch:

```sql
SELECT
    title,
    price,
    CASE
        WHEN price < 15         THEN 'Budget'
        WHEN price BETWEEN 15 AND 30 THEN 'Mid-range'
        ELSE                          'Premium'
    END AS price_tier
FROM books;
```

Output:
```
title                       | price | price_tier
----------------------------+-------+-----------
Python Crash Course         | 35.99 | Premium
The Data Detective          | 18.50 | Mid-range
Naked Statistics            | 16.99 | Mid-range
Storytelling with Data      | 29.99 | Mid-range
How to Lie with Statistics  | 12.99 | Budget
Deep Learning               | 74.99 | Premium
```

### Simple `CASE` (Equality Check)

When all branches check equality against one column, a simpler syntax is available:

```sql
SELECT
    name,
    city,
    CASE city
        WHEN 'Chicago'  THEN 'Midwest'
        WHEN 'New York' THEN 'Northeast'
        WHEN 'Austin'   THEN 'South'
        WHEN 'Denver'   THEN 'West'
        ELSE 'Other'
    END AS region
FROM customers;
```

## `COALESCE` — Handling `NULL` Values

`COALESCE(val1, val2, ...)` returns the first non-`NULL` value from its arguments. It is the clean way to replace `NULL` with a default:

```sql
-- Replace NULL city with 'Unknown'
SELECT name, COALESCE(city, 'Unknown') AS city
FROM customers;
```

```sql
-- Use discount if available, otherwise use standard price
SELECT
    title,
    price,
    COALESCE(discount_price, price) AS effective_price
FROM books;
```

`COALESCE` is especially useful when joining tables where some rows have no match — the joined columns will be `NULL` and you may want a sensible default.

## Built-In Functions

### Numeric Functions

```sql
SELECT
    title,
    price,
    ROUND(price * 1.08, 2)  AS price_with_tax,  -- Round to 2 decimal places
    ABS(stock - 10)         AS distance_from_10
FROM books;
```

### Text Functions

```sql
SELECT
    UPPER(name)             AS name_upper,
    LOWER(email)            AS email_lower,
    LENGTH(title)           AS title_length,
    SUBSTR(title, 1, 10)    AS title_preview  -- First 10 characters
FROM customers
JOIN books ON 1=1
LIMIT 5;
```

```sql
-- Find books with long titles
SELECT title, LENGTH(title) AS char_count
FROM books
WHERE LENGTH(title) > 20
ORDER BY char_count DESC;
```

### Date Functions

SQLite stores dates as text. Use `strftime()` to extract parts or format them:

```sql
-- Extract the year and month from order_date
SELECT
    order_id,
    order_date,
    strftime('%Y', order_date)       AS order_year,
    strftime('%m', order_date)       AS order_month,
    strftime('%Y-%m', order_date)    AS year_month
FROM orders;
```

```sql
-- Orders placed in February 2024
SELECT * FROM orders
WHERE strftime('%Y-%m', order_date) = '2024-02';
```

## Subqueries

A subquery is a `SELECT` embedded inside another statement. The inner query executes first and its result is passed to the outer query.

### Subquery in `WHERE`

```sql
-- Find customers who have placed at least one order
SELECT name, city
FROM customers
WHERE customer_id IN (
    SELECT DISTINCT customer_id FROM orders
);
```

The inner query returns the list of customer IDs that appear in `orders`. The outer query then returns customers whose ID is in that list.

```sql
-- Find books that cost more than the average price
SELECT title, price
FROM books
WHERE price > (SELECT AVG(price) FROM books);
```

Output:
```
title               | price
--------------------+-------
Python Crash Course | 35.99
Deep Learning       | 74.99
```

### Subquery in `FROM` (Derived Table)

A subquery in the `FROM` clause acts as a temporary table. It must be given an alias:

```sql
-- Average order value per customer, then filter for high-value customers
SELECT customer_id, avg_spent
FROM (
    SELECT customer_id, AVG(quantity * price) AS avg_spent
    FROM orders
    JOIN books USING (book_id)
    GROUP BY customer_id
) AS customer_summary
WHERE avg_spent > 30;
```

### Subquery in `SELECT` (Scalar Subquery)

A subquery that returns a single value can appear in the `SELECT` list:

```sql
-- Show each book's price alongside the overall average
SELECT
    title,
    price,
    ROUND((SELECT AVG(price) FROM books), 2)    AS avg_price,
    ROUND(price - (SELECT AVG(price) FROM books), 2) AS diff_from_avg
FROM books
ORDER BY diff_from_avg DESC;
```

## Combining Techniques

Here's a realistic query that uses aliases, `CASE`, `COALESCE`, and a subquery together:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bookstore.db")

query = """
    SELECT
        b.title,
        b.genre,
        b.price,
        CASE
            WHEN b.price < 15         THEN 'Budget'
            WHEN b.price BETWEEN 15 AND 30 THEN 'Mid-range'
            ELSE                          'Premium'
        END AS price_tier,
        COALESCE(b.genre, 'Uncategorized') AS genre_clean,
        ROUND(b.price - avg.mean_price, 2) AS vs_average
    FROM books AS b,
         (SELECT ROUND(AVG(price), 2) AS mean_price FROM books) AS avg
    ORDER BY b.price DESC;
"""

df = pd.read_sql_query(query, conn)
print(df)
conn.close()
```

## Conclusion

In this lesson, you learned how to write expressive SQL using column aliases for readability, `CASE WHEN` for conditional classification, `COALESCE` for handling `NULL` values, SQL's built-in numeric and text functions, and subqueries for filtering against derived results or computed aggregates. These tools let you do significant data transformation directly in SQL before data even reaches Python. In the next lesson, you'll learn how to summarize data across groups using `GROUP BY` and `ORDER BY`.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="The subquery `(SELECT AVG(price) FROM books)` runs first and returns a single number — the average price. The outer query then filters `books` to return only rows where `price` is greater than that computed average. This is a scalar subquery: it returns one value used as a comparison threshold.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What does the following query return?
  </div>
  <div class="quiz-subquestion">
    <pre><code>SELECT title, price
FROM books
WHERE price &gt; (SELECT AVG(price) FROM books);</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>All books, with each book's price compared to the average price.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The single book with the highest price.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Only the books whose price is above the average price across all books.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The average price of all books as a single value.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="`COALESCE(city, &#039;Unknown&#039;)` evaluates its arguments left to right and returns the first one that is not `NULL`. If `city` is `NULL`, it moves to the next argument, `&#039;Unknown&#039;`, and returns that. If `city` has a value like `&#039;Chicago&#039;`, it returns `&#039;Chicago&#039;` immediately. This makes `COALESCE` the standard SQL pattern for replacing `NULL` with a default.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A `city` column contains some `NULL` values. What does `COALESCE(city, 'Unknown')` return for those rows?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`NULL` — `COALESCE` does not modify `NULL` values.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>An empty string `''`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`'Unknown'` — `COALESCE` returns the first non-`NULL` argument.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>An error, because `NULL` cannot be compared to a string.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="If a `CASE WHEN` expression evaluates all conditions and none is `TRUE`, and there is no `ELSE` clause, the expression returns `NULL`. This is standard SQL behavior. Adding `ELSE &#039;Other&#039;` (or another default) is a best practice to ensure you always get a meaningful value rather than a silent `NULL`.">
  <div class="quiz-question">
    <strong>Question 3:</strong> In a `CASE WHEN` expression, what happens if no condition matches and there is no `ELSE` clause?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>SQL raises a syntax error because `ELSE` is required.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The expression returns `NULL`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The expression returns `0` for numeric columns and `''` for text columns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The expression returns the value from the last `WHEN` branch.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

