# SQL and Python

## Overview

In the previous lesson, you learned how to join tables to answer questions that span multiple parts of a database. So far, you've been running SQL queries directly against your SQLite database. In this lesson, you'll learn how to integrate SQL into a Python data science workflow — executing queries, retrieving results, and loading them into Pandas DataFrames for analysis and visualization.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Connect to a SQLite database using Python's `sqlite3` module.
- Integrate SQL queries into data science workflows with Pandas.

## Key terms

**`sqlite3`:** Python's built-in module for working with SQLite databases. No installation required.

**`conn.cursor()`:** Creates a cursor object used to execute SQL statements.

**`cursor.execute()`:** Runs a single SQL statement.

**`cursor.executemany()`:** Runs a parameterized SQL statement once for each item in a list — efficient for bulk inserts.

**`cursor.fetchone()`:** Returns the next single row from a query result as a tuple, or `None` if no rows remain.

**`cursor.fetchall()`:** Returns all remaining rows from a query result as a list of tuples.

**`pd.read_sql_query()`:** A Pandas function that executes a SQL query against a database connection and returns the result as a DataFrame.

**`df.to_sql()`:** A Pandas DataFrame method that writes the DataFrame to a database table.

**Parameterized query:** A query using `?` placeholders for values, which are supplied separately to prevent SQL injection.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/08_sql-and-python_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

SQL and Python are complementary tools. SQL excels at storing, querying, and aggregating large datasets efficiently. Python excels at statistical analysis, machine learning, and visualization. The ability to move data fluidly between the two — querying in SQL, then analyzing in Pandas — is one of the most practical skills in a data scientist's toolkit.

Since `sqlite3` is part of Python's standard library, no installation is required. You can open a database, run queries, and pull results into Pandas in just a few lines.

## The `sqlite3` Workflow

### Connecting and Querying

```python
import sqlite3

# Open a connection (creates the file if it doesn't exist)
conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# Execute a query
cursor.execute("SELECT name, city FROM customers WHERE city = 'Chicago';")

# Retrieve results
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
```

Output:
```
('Alex Carter', 'Chicago')
('Sam Torres', 'Chicago')
```

### Using a Context Manager

The `with` statement ensures the connection is properly closed (and changes committed) even if an error occurs:

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM orders;")
    count = cursor.fetchone()[0]
    print(f"Total orders: {count}")
```

### Fetching Results

| Method | Returns | Use when |
|--------|---------|----------|
| `fetchone()` | One row as a tuple, or `None` | You expect or want one row |
| `fetchmany(n)` | List of n rows | Processing in batches |
| `fetchall()` | All rows as a list of tuples | Small to medium result sets |

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT title, price FROM books ORDER BY price DESC;")

    # Get only the most expensive book
    top = cursor.fetchone()
    print(f"Most expensive: {top[0]} — ${top[1]}")

    # Get the next two
    next_two = cursor.fetchmany(2)
    for row in next_two:
        print(row)
```

## Parameterized Queries

Always use `?` placeholders when incorporating variables into a query — never use f-strings or string concatenation for SQL values:

```python
city = "Chicago"

# UNSAFE — never do this:
# cursor.execute(f"SELECT * FROM customers WHERE city = '{city}'")

# SAFE — always do this:
cursor.execute("SELECT * FROM customers WHERE city = ?", (city,))
```

The `?` approach prevents **SQL injection** — a security vulnerability where malicious input could execute unintended SQL.

### Multiple Parameters

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()

    min_price = 15
    max_price = 40
    cursor.execute(
        "SELECT title, price FROM books WHERE price BETWEEN ? AND ? ORDER BY price;",
        (min_price, max_price)
    )
    for row in cursor.fetchall():
        print(row)
```

## Inserting and Modifying Data

### Single Insert

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO customers (name, email, city, member_since) VALUES (?, ?, ?, ?)",
        ("Casey Wright", "casey@email.com", "Boston", "2024-04-01")
    )
    # conn.commit() is called automatically by the context manager
    print(f"New customer ID: {cursor.lastrowid}")
```

`cursor.lastrowid` returns the primary key assigned to the most recently inserted row.

### Bulk Insert with `executemany()`

```python
new_books = [
    ("Clean Code",   "Robert Martin", "Programming", 42.99, 15),
    ("The Art of Statistics", "David Spiegelhalter", "Statistics", 24.99, 20),
]

with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO books (title, author, genre, price, stock) VALUES (?, ?, ?, ?, ?)",
        new_books
    )
    print(f"{cursor.rowcount} rows inserted.")
```

### Update and Delete

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()

    # Reduce stock after a sale
    cursor.execute(
        "UPDATE books SET stock = stock - ? WHERE book_id = ?",
        (1, 3)
    )

    # Delete a record
    cursor.execute(
        "DELETE FROM orders WHERE order_id = ?",
        (10,)
    )
```

## SQL + Pandas: The Core Workflow

### `pd.read_sql_query()` — Query Directly into a DataFrame

This is the most common pattern in data science — run a SQL query and load the result directly into a Pandas DataFrame for further analysis:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bookstore.db")

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
    INNER JOIN customers AS c ON o.customer_id = c.customer_id
    INNER JOIN books     AS b ON o.book_id     = b.book_id
    ORDER BY o.order_date;
"""

df = pd.read_sql_query(query, conn)
print(df.head())
print(f"\nShape: {df.shape}")
conn.close()
```

The column names in the DataFrame come from the `AS` aliases in the `SELECT` clause — another reason to name your columns descriptively.

### Analyzing the Query Result in Pandas

Once the data is in a DataFrame, apply any Pandas operation:

```python
conn = sqlite3.connect("bookstore.db")

df = pd.read_sql_query("""
    SELECT c.name, b.genre, o.quantity, b.price,
           ROUND(o.quantity * b.price, 2) AS line_total
    FROM orders o
    JOIN customers c USING (customer_id)
    JOIN books     b USING (book_id)
""", conn)

# Total spend per customer
print(df.groupby("name")["line_total"].sum().sort_values(ascending=False))

# Average line total by genre
print(df.groupby("genre")["line_total"].mean().round(2))

conn.close()
```

### `df.to_sql()` — Write a DataFrame to a Database Table

You can also go the other direction — writing a Pandas DataFrame (e.g., from a CSV or API) into a SQLite table:

```python
import pandas as pd
import sqlite3

# Load external data
external_df = pd.read_csv("new_customers.csv")

with sqlite3.connect("bookstore.db") as conn:
    external_df.to_sql(
        name="customers",
        con=conn,
        if_exists="append",   # 'replace' to overwrite, 'fail' to error if exists
        index=False
    )
    print(f"Wrote {len(external_df)} rows to 'customers' table.")
```

### Getting Column Names from a Query

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books LIMIT 1;")
    column_names = [description[0] for description in cursor.description]
    print(column_names)
```

Output:
```
['book_id', 'title', 'author', 'genre', 'price', 'stock']
```

## Putting It Together: A Full Pipeline

Here's a realistic end-to-end workflow: load a CSV into SQLite, query it with SQL, and analyze it with Pandas:

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load CSV into SQLite
df_raw = pd.read_csv("monthly_sales.csv")
with sqlite3.connect("sales.db") as conn:
    df_raw.to_sql("sales", conn, if_exists="replace", index=False)

# Step 2: Query with SQL
with sqlite3.connect("sales.db") as conn:
    query = """
        SELECT
            strftime('%Y-%m', order_date)  AS month,
            SUM(revenue)                   AS total_revenue,
            COUNT(*)                       AS num_orders
        FROM sales
        GROUP BY month
        ORDER BY month;
    """
    monthly = pd.read_sql_query(query, conn)

# Step 3: Analyze and visualize in Pandas / Matplotlib
monthly.plot(x="month", y="total_revenue", kind="bar",
             title="Monthly Revenue", figsize=(10, 5), legend=False)
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.show()
```

## Conclusion

In this lesson, you learned how to use Python's `sqlite3` module to connect to a database, execute queries with `cursor.execute()`, use parameterized `?` placeholders for safety, insert and update data, and load query results directly into Pandas DataFrames with `pd.read_sql_query()`. You also saw how to write Pandas DataFrames to SQLite tables using `df.to_sql()`. This SQL-to-Pandas bridge is a core pattern in data science — it lets you leverage the best of both tools. In the next lesson, you'll apply everything from this module in the **module assessment**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/08_sql-and-python_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="`pd.read_sql_query()` executes the SQL string against the provided database connection and automatically wraps the results in a Pandas DataFrame. The column names come from the `SELECT` clause — either the original column names or `AS` aliases. This is the primary bridge between SQL and the Pandas analysis environment.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What does `pd.read_sql_query(query, conn)` return?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>A list of tuples containing the query results.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>A Pandas DataFrame with columns matching the query's `SELECT` aliases.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>A cursor object that can be iterated to retrieve rows.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>A JSON string representing the query results.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Only option 2 uses a parameterized query with a `?` placeholder, which is the safe approach. Options 1, 3, and 4 all embed the user input directly into the SQL string, creating a SQL injection vulnerability. For example, if `user_input` is `&quot;&#039;; DROP TABLE users; --&quot;`, options 1, 3, and 4 would execute that destructive command. The `?` placeholder causes `sqlite3` to treat the value as pure data, never as SQL syntax.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You want to insert a user-supplied value into a SQL query. Which approach is correct?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`cursor.execute(f"INSERT INTO users (name) VALUES ('{user_input}')")`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>`cursor.execute("INSERT INTO users (name) VALUES (?)", (user_input,))`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`cursor.execute("INSERT INTO users (name) VALUES (" + user_input + ")")`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>`cursor.execute("INSERT INTO users (name) VALUES (%s)" % user_input)`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="The `if_exists` parameter controls what happens when the target table already exists: `&quot;fail&quot;` raises an error, `&quot;replace&quot;` drops and recreates the table with the new data, and `&quot;append&quot;` adds the DataFrame rows to the end of the existing table without changing its structure. Use `&quot;replace&quot;` carefully — it permanently deletes the existing table and all its data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> When using `df.to_sql(name="sales", con=conn, if_exists="replace", index=False)`, what does `if_exists="replace"` do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>It skips the write if the table already exists.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>It appends the DataFrame rows to the existing table without changing its structure.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>It drops the existing table and recreates it with the DataFrame's schema and data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It raises an error if any row in the DataFrame already exists in the table.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

