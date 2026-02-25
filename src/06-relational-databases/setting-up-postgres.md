# Setting Up SQLite

## Overview

In the previous lesson, you learned what relational databases and SQL are and why they matter for data science. Now it's time to get hands-on. In this lesson, you'll set up **SQLite** — a lightweight, serverless database that requires zero installation — create your first database, and build the sample schema you'll use throughout this module.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Create and connect to a SQLite database using Python.
- Access and inspect a SQLite database using a GUI tool.

## Key terms

**SQLite:** A lightweight, file-based relational database engine that is built into Python's standard library. A SQLite database is stored as a single `.db` file.

**`sqlite3`:** Python's built-in module for creating, connecting to, and querying SQLite databases. No installation required.

**Connection:** A live link between a Python script and a SQLite database file, created with `sqlite3.connect()`.

**Cursor:** An object used to execute SQL statements and retrieve results from a database connection.

**`commit()`:** A method that saves all pending changes to the database permanently.

**`fetchall()`:** A cursor method that retrieves all rows from the result of a query as a list of tuples.

**DB Browser for SQLite:** A free, open-source GUI application for viewing, editing, and querying SQLite database files without writing code.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

One of SQLite's greatest advantages is that it requires absolutely no setup. There is no server to start, no username or password to configure, and no software to install beyond Python itself. A SQLite database is simply a file on your computer (or in memory). The `sqlite3` module is part of Python's standard library, so it is available in every Python environment including Google Colab.

This makes SQLite the ideal tool for learning SQL — you can go from zero to running queries in under a minute.

## Connecting to a Database

### Creating (or Opening) a Database

Use `sqlite3.connect()` to open a connection. If the file doesn't exist, SQLite creates it automatically:

```python
import sqlite3

# Create (or open) a database file called bookstore.db
conn = sqlite3.connect("bookstore.db")
print("Connected to bookstore.db")
```

A file named `bookstore.db` will appear in your working directory. To create an **in-memory** database that disappears when your script ends (useful for testing):

```python
conn = sqlite3.connect(":memory:")
```

### The Cursor

To execute SQL statements, you need a **cursor** — think of it as the pen you use to write queries against the connection:

```python
cursor = conn.cursor()
```

### The Basic Workflow

Every interaction with a SQLite database follows the same four-step pattern:

```python
import sqlite3

# 1. Connect
conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# 2. Execute SQL
cursor.execute("SELECT sqlite_version();")

# 3. Fetch results
version = cursor.fetchone()
print(f"SQLite version: {version[0]}")

# 4. Close when done
conn.close()
```

## Building the Sample Database

Throughout this module, you'll work with a simple bookstore database containing three tables:

- **`customers`** — people who buy books
- **`books`** — the items in the store's catalog
- **`orders`** — purchase records linking customers to books

Let's build it:

```python
import sqlite3

conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# Create tables
cursor.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id   INTEGER PRIMARY KEY,
        name          TEXT    NOT NULL,
        email         TEXT    UNIQUE NOT NULL,
        city          TEXT,
        member_since  TEXT
    );

    CREATE TABLE IF NOT EXISTS books (
        book_id       INTEGER PRIMARY KEY,
        title         TEXT    NOT NULL,
        author        TEXT    NOT NULL,
        genre         TEXT,
        price         REAL    NOT NULL,
        stock         INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id      INTEGER PRIMARY KEY,
        customer_id   INTEGER NOT NULL,
        book_id       INTEGER NOT NULL,
        order_date    TEXT    NOT NULL,
        quantity      INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (book_id)     REFERENCES books(book_id)
    );
""")

conn.commit()
print("Tables created successfully.")
conn.close()
```

`executescript()` runs multiple SQL statements separated by semicolons in one call. `CREATE TABLE IF NOT EXISTS` avoids an error if the table already exists.

### Inserting Sample Data

```python
conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# Insert customers
cursor.executemany(
    "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?)",
    [
        (1, "Alex Carter",   "alex@email.com",   "Chicago",  "2022-03-10"),
        (2, "Jordan Lee",    "jordan@email.com",  "New York", "2021-11-05"),
        (3, "Morgan Singh",  "morgan@email.com",  "Austin",   "2023-01-20"),
        (4, "Sam Torres",    "sam@email.com",     "Chicago",  "2022-07-15"),
        (5, "Taylor Reyes",  "taylor@email.com",  "Denver",   "2023-06-01"),
    ]
)

# Insert books
cursor.executemany(
    "INSERT OR IGNORE INTO books VALUES (?, ?, ?, ?, ?, ?)",
    [
        (1, "Python Crash Course",      "Eric Matthes",     "Programming", 35.99, 12),
        (2, "The Data Detective",        "Tim Harford",      "Data Science", 18.50, 25),
        (3, "Naked Statistics",          "Charles Wheelan",  "Statistics",  16.99, 30),
        (4, "Storytelling with Data",    "Cole Knaflic",     "Data Science", 29.99, 8),
        (5, "How to Lie with Statistics","Darrell Huff",     "Statistics",  12.99, 50),
        (6, "Deep Learning",             "Goodfellow et al.","Programming", 74.99, 5),
    ]
)

# Insert orders
cursor.executemany(
    "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?)",
    [
        (1,  1, 1, "2024-01-10", 1),
        (2,  2, 3, "2024-01-14", 2),
        (3,  1, 4, "2024-01-20", 1),
        (4,  3, 2, "2024-02-03", 1),
        (5,  4, 5, "2024-02-10", 3),
        (6,  2, 6, "2024-02-15", 1),
        (7,  5, 1, "2024-02-20", 1),
        (8,  3, 4, "2024-03-05", 2),
        (9,  1, 5, "2024-03-12", 1),
        (10, 4, 2, "2024-03-18", 1),
    ]
)

conn.commit()
print("Sample data inserted.")
conn.close()
```

The `?` placeholders prevent SQL injection by safely binding values — always use these instead of formatting values directly into query strings.

## Verifying the Database

Check that everything was created correctly:

```python
conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Count rows in each table
for table in ["customers", "books", "orders"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    print(f"  {table}: {cursor.fetchone()[0]} rows")

conn.close()
```

Output:
```
Tables: [('customers',), ('books',), ('orders',)]
  customers: 5 rows
  books: 6 rows
  orders: 10 rows
```

## Using a Context Manager

Python's `with` statement automatically commits and closes the connection, even if an error occurs — this is the recommended pattern for production code:

```python
with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT name, city FROM customers;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
```

## Optional: DB Browser for SQLite (GUI)

If you prefer a visual interface, [**DB Browser for SQLite**](https://sqlitebrowser.org/) is a free desktop application that lets you:

- Browse tables and their data in a spreadsheet-like view
- Run SQL queries interactively
- View table schemas and relationships
- Export query results to CSV

To use it: download and install from [sqlitebrowser.org](https://sqlitebrowser.org/), then open your `.db` file using **File → Open Database**.

## Conclusion

In this lesson, you set up a SQLite database using Python's built-in `sqlite3` module with no installation required, built a three-table bookstore schema using `CREATE TABLE` and `executescript()`, inserted sample data using `executemany()` with `?` placeholders, and verified the setup. You also learned the recommended `with` context manager pattern for safe database access. In the next lesson, you'll define table structures in depth — data types, primary keys, foreign keys, and constraints.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What does `sqlite3.connect("store.db")` do if the file `store.db` does not yet exist?**
1. It raises a `FileNotFoundError`.
2. It creates a new, empty database file called `store.db` and returns a connection to it.
3. It connects to an in-memory database and ignores the filename.
4. It prompts you to create a new database interactively.

**Correct Answer:**
2. It creates a new, empty database file called `store.db` and returns a connection to it.

**Explanation:**
SQLite's behavior is to create the database file if it doesn't exist. This is by design — it makes getting started extremely simple. If you pass `":memory:"` instead of a filename, it creates a temporary database in RAM instead.

---

#### **Question 2: Why should you use `?` placeholders instead of f-strings when inserting data into a SQLite query?**
1. F-strings are slower than `?` placeholders for large datasets.
2. `?` placeholders prevent SQL injection by letting the database driver safely handle the values, while f-strings insert them as raw strings.
3. SQLite does not support f-strings in Python 3.
4. `?` placeholders automatically convert Python types to SQL types.

**Correct Answer:**
2. `?` placeholders prevent SQL injection by letting the database driver safely handle the values, while f-strings insert them as raw strings.

**Explanation:**
SQL injection is a security vulnerability where malicious input in a value (like `'; DROP TABLE customers; --`) is interpreted as SQL code. Using `?` placeholders tells the `sqlite3` driver to treat the value as data — not as SQL — no matter what it contains. Always use parameterized queries (`?`) instead of string formatting when inserting user-supplied or external data.

---

#### **Question 3: What is the purpose of calling `conn.commit()` after executing INSERT statements?**
1. It closes the database connection safely.
2. It permanently saves the changes to the database file. Without it, changes are lost when the connection closes.
3. It verifies that the inserted data passes all constraints.
4. It refreshes the connection to apply the new schema.

**Correct Answer:**
2. It permanently saves the changes to the database file. Without it, changes are lost when the connection closes.

**Explanation:**
SQLite (like most databases) uses **transactions** — a group of changes that are applied together. Changes made within a transaction are held in memory until you call `commit()`, which writes them permanently to disk. If you close the connection without committing, the changes are rolled back and lost. The `with` context manager handles this automatically.
