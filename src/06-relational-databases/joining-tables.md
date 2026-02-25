# Joining Tables

## Overview

In the previous lesson, you learned how to aggregate data within a single table using `GROUP BY` and `ORDER BY`. The true power of the relational model, however, comes from combining data **across** tables. In this lesson, you'll learn the different types of SQL joins — `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, and `FULL OUTER JOIN` — and when to use each one.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Perform inner, left, and outer joins.
- Combine data from multiple tables for comprehensive analysis.

## Key terms

**`JOIN`:** A SQL operation that combines rows from two tables based on a matching condition — typically a shared key column.

**`ON`:** The clause that specifies the join condition — which columns in each table must match.

**`INNER JOIN`:** Returns only rows where there is a match in **both** tables. Rows with no match are excluded.

**`LEFT JOIN` (LEFT OUTER JOIN):** Returns all rows from the left table and matching rows from the right table. Rows in the left table with no match get `NULL` for all right-table columns.

**`RIGHT JOIN` (RIGHT OUTER JOIN):** Returns all rows from the right table and matching rows from the left table. Note: SQLite supports `RIGHT JOIN` as of version 3.39.0 (2022).

**`FULL OUTER JOIN`:** Returns all rows from both tables, with `NULL` where there is no match on either side. Not natively supported in SQLite — can be emulated with `LEFT JOIN UNION LEFT JOIN`.

**`CROSS JOIN`:** Returns every combination of rows from both tables (the Cartesian product). Use sparingly.

**`USING`:** A shorthand for `ON table1.col = table2.col` when both tables share the same column name.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

The bookstore database stores customers in one table and orders in another. To answer "What did Alex buy?", you need to match Alex's `customer_id` in `customers` with the corresponding rows in `orders`, and then match those `book_id` values in `books`. This is what joins do — they stitch tables together along matching keys so you can ask questions that span multiple tables.

Understanding which join type to use is critical: the wrong join silently changes which rows appear in your results and can lead to incorrect conclusions.

## Visual Intuition

Imagine two tables, A and B, joined on a shared key:

```
Table A          Table B
--------         --------
 1  Alex          1  Order #101
 2  Jordan        2  Order #102
 3  Morgan        4  Order #103  ← no matching customer

INNER JOIN  → rows 1, 2 only (both sides match)
LEFT JOIN   → rows 1, 2, 3 (all of A; Morgan gets NULL from B)
RIGHT JOIN  → rows 1, 2, 4 (all of B; Order #103 gets NULL from A)
FULL OUTER  → rows 1, 2, 3, 4 (everything; NULLs where no match)
```

## Setup

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bookstore.db")
```

## `INNER JOIN`

Returns only rows where both tables have a matching value. Non-matching rows from either table are silently dropped.

```sql
-- Which customers placed which orders?
SELECT
    c.name          AS customer,
    o.order_id,
    o.order_date,
    o.quantity
FROM customers AS c
INNER JOIN orders AS o ON c.customer_id = o.customer_id
ORDER BY o.order_date;
```

Output (first few rows):
```
customer     | order_id | order_date | quantity
-------------+----------+------------+---------
Alex Carter  | 1        | 2024-01-10 | 1
Jordan Lee   | 2        | 2024-01-14 | 2
Alex Carter  | 3        | 2024-01-20 | 1
Morgan Singh | 4        | 2024-02-03 | 1
...
```

Taylor (customer 5) placed one order so they appear once. Any customer with no orders at all would not appear — `INNER JOIN` excludes them.

### Joining Three Tables

Chain multiple joins to combine data from three or more tables:

```sql
-- Full order details: customer name, book title, price, date
SELECT
    c.name          AS customer,
    b.title         AS book,
    b.genre,
    b.price,
    o.quantity,
    ROUND(o.quantity * b.price, 2) AS line_total,
    o.order_date
FROM orders AS o
INNER JOIN customers AS c ON o.customer_id = c.customer_id
INNER JOIN books     AS b ON o.book_id     = b.book_id
ORDER BY o.order_date;
```

### `USING` Shorthand

When both tables share the exact column name, `USING` is a cleaner alternative to `ON`:

```sql
SELECT c.name, o.order_date
FROM customers AS c
INNER JOIN orders USING (customer_id);
```

## `LEFT JOIN`

Returns **all rows from the left table**, plus matching rows from the right table. Where there is no match, right-table columns are `NULL`.

```sql
-- All customers, with their order count (0 if they've never ordered)
SELECT
    c.name,
    c.city,
    COUNT(o.order_id) AS order_count
FROM customers AS c
LEFT JOIN orders AS o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.city
ORDER BY order_count DESC;
```

Output:
```
name         | city     | order_count
-------------+----------+-------------
Alex Carter  | Chicago  | 3
Sam Torres   | Chicago  | 2
Jordan Lee   | New York | 2
Morgan Singh | Austin   | 2
Taylor Reyes | Denver   | 1
```

Because we have sample orders for all five customers, nobody has zero here. But in a real dataset with unmatched customers, they would show `order_count = 0` — a key difference from `INNER JOIN`, which would drop them entirely.

```sql
-- Find customers who have NEVER placed an order
SELECT c.name, c.city
FROM customers AS c
LEFT JOIN orders AS o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

This pattern — `LEFT JOIN ... WHERE right_table.key IS NULL` — is the standard SQL idiom for finding "rows in A with no match in B."

## `RIGHT JOIN`

Returns **all rows from the right table**, plus matching rows from the left table. It is the mirror of `LEFT JOIN` and is supported in SQLite 3.39.0+.

```sql
-- All books, with their order count (0 if never ordered)
SELECT
    b.title,
    b.genre,
    COUNT(o.order_id) AS times_ordered
FROM orders AS o
RIGHT JOIN books AS b ON o.book_id = b.book_id
GROUP BY b.book_id, b.title, b.genre
ORDER BY times_ordered DESC;
```

In practice, a `RIGHT JOIN` can always be rewritten as a `LEFT JOIN` by swapping the table order — many developers prefer `LEFT JOIN` for consistency.

## `FULL OUTER JOIN` (Emulated in SQLite)

SQLite does not support `FULL OUTER JOIN` natively. Emulate it by combining a `LEFT JOIN` and a reversed `LEFT JOIN` using `UNION`:

```sql
-- All customers and all books, matched where orders exist
SELECT c.name AS customer, b.title AS book
FROM customers AS c
LEFT JOIN orders  AS o ON c.customer_id = o.customer_id
LEFT JOIN books   AS b ON o.book_id     = b.book_id

UNION

SELECT c.name AS customer, b.title AS book
FROM books AS b
LEFT JOIN orders      AS o ON b.book_id    = o.book_id
LEFT JOIN customers   AS c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

## `CROSS JOIN`

Returns the Cartesian product — every row from the left table paired with every row from the right table. Use sparingly; it grows exponentially.

```sql
-- Every possible customer-book combination (useful for recommendation matrices)
SELECT c.name, b.title
FROM customers AS c
CROSS JOIN books AS b
ORDER BY c.name, b.title;
-- Returns 5 customers × 6 books = 30 rows
```

## Join Type Quick Reference

| Join Type | Returns |
|-----------|---------|
| `INNER JOIN` | Only rows with a match in both tables |
| `LEFT JOIN` | All rows from the left table; `NULL` where no match in right |
| `RIGHT JOIN` | All rows from the right table; `NULL` where no match in left |
| `FULL OUTER JOIN` | All rows from both tables; `NULL` where no match on either side |
| `CROSS JOIN` | Every combination of rows from both tables |

## Putting It Together: Full Sales Report

```python
query = """
    SELECT
        c.name          AS customer,
        c.city,
        b.title         AS book,
        b.genre,
        o.quantity,
        ROUND(o.quantity * b.price, 2)  AS line_total,
        o.order_date
    FROM orders AS o
    INNER JOIN customers AS c  ON o.customer_id = c.customer_id
    INNER JOIN books     AS b  ON o.book_id     = b.book_id
    ORDER BY line_total DESC;
"""

df = pd.read_sql_query(query, conn)
print(df.to_string(index=False))
conn.close()
```

## Conclusion

In this lesson, you learned how to combine rows from multiple tables using `INNER JOIN` (matching rows only), `LEFT JOIN` (all rows from the left, `NULL` for unmatched right rows), `RIGHT JOIN` (all rows from the right), and how to emulate `FULL OUTER JOIN` in SQLite. You also learned the "left join where right key is null" idiom for finding non-matching rows. Joins are the most powerful feature of relational databases — they let you answer questions that span the entire data model. In the next lesson, you'll learn how to connect to and query SQLite databases directly from Python using the `sqlite3` module and Pandas.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: You want to list all customers, including those who have never placed an order (showing 0 for their order count). Which join type should you use?**
1. `INNER JOIN` — it returns all customers.
2. `LEFT JOIN` with `customers` as the left table — it returns all customers, with `NULL` for unmatched orders.
3. `RIGHT JOIN` with `orders` as the right table — it returns all orders.
4. `CROSS JOIN` — it returns every customer-order combination.

**Correct Answer:**
2. `LEFT JOIN` with `customers` as the left table — it returns all customers, with `NULL` for unmatched orders.

**Explanation:**
`LEFT JOIN` keeps every row from the left table regardless of whether a match exists in the right table. By placing `customers` on the left, every customer is included. Those with no matching orders will have `NULL` in the order columns. Wrapping the join in a `GROUP BY` with `COUNT(order_id)` then produces `0` for customers with no orders, since `COUNT` skips `NULL` values.

---

#### **Question 2: What does the following pattern find?**
```sql
SELECT c.name
FROM customers AS c
LEFT JOIN orders AS o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```
1. All customers who have placed at least one order.
2. All orders where the customer's record is missing.
3. All customers who have never placed an order.
4. All customers and orders, including unmatched ones.

**Correct Answer:**
3. All customers who have never placed an order.

**Explanation:**
The `LEFT JOIN` keeps all customers, filling `NULL` into the order columns for customers with no orders. The `WHERE o.order_id IS NULL` then filters to keep only those customers where no matching order was found — i.e., customers who have never ordered. This is the canonical SQL pattern for finding "rows in table A with no match in table B."

---

#### **Question 3: What is the key difference between `INNER JOIN` and `LEFT JOIN`?**
1. `INNER JOIN` is faster; `LEFT JOIN` is used for large datasets only.
2. `INNER JOIN` returns only rows with matches in both tables; `LEFT JOIN` also returns rows from the left table that have no match in the right table.
3. `LEFT JOIN` can only be used when joining exactly two tables; `INNER JOIN` works with any number.
4. `INNER JOIN` requires the tables to have identical schemas; `LEFT JOIN` does not.

**Correct Answer:**
2. `INNER JOIN` returns only rows with matches in both tables; `LEFT JOIN` also returns rows from the left table that have no match in the right table.

**Explanation:**
The critical difference is how non-matching rows are handled. `INNER JOIN` silently drops any row from either table that has no counterpart — this can cause data to disappear without an obvious error. `LEFT JOIN` preserves all rows from the left table, using `NULL` to represent missing right-table data. Choosing the wrong join type is a common source of incorrect query results.
