# Tables, Data Types, and Fields

## Overview

In the previous lesson, you set up a SQLite database and created the bookstore schema using `CREATE TABLE`. In this lesson, you'll go deeper into how tables are structured — the data types SQLite supports, how to define constraints that enforce data quality, and how primary and foreign keys create relationships between tables.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Create tables with appropriate data types.
- Define primary keys, foreign keys, and constraints.

## Key terms

**`CREATE TABLE`:** The SQL statement used to define a new table, specifying column names, data types, and constraints.

**Data type:** A declaration that tells SQLite what kind of value a column holds — integer, real number, text, or binary data.

**`INTEGER`:** SQLite's data type for whole numbers.

**`REAL`:** SQLite's data type for floating-point (decimal) numbers.

**`TEXT`:** SQLite's data type for strings of any length.

**`NULL`:** A special marker representing the absence of a value. Different from zero or an empty string.

**`NOT NULL`:** A constraint that prevents a column from accepting `NULL` — every row must supply a value.

**`UNIQUE`:** A constraint that ensures no two rows can have the same value in a column.

**`DEFAULT`:** A clause that supplies a value automatically when a row is inserted without specifying that column.

**`PRIMARY KEY`:** A constraint that uniquely identifies each row. Combining `INTEGER` and `PRIMARY KEY` in SQLite creates an auto-incrementing identifier.

**`FOREIGN KEY`:** A constraint that ensures a column's value matches an existing primary key in another table, enforcing referential integrity.

**`CHECK`:** A constraint that requires column values to satisfy a custom condition.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

A table's structure — its column names, data types, and constraints — is called its **schema**. Designing a good schema is one of the most important decisions in database work. A well-designed schema enforces data quality automatically, making it impossible (or at least difficult) to store invalid data. A poorly designed schema allows inconsistencies that cause subtle errors in every downstream query and analysis.

## SQLite Data Types

SQLite uses a flexible type system called **type affinity**. Unlike most databases, SQLite does not strictly enforce types — a column declared as `INTEGER` will still accept text. However, declaring the intended type is important for clarity, sorting behavior, and compatibility with other SQL systems.

The five SQLite storage classes are:

| Type | Description | Example values |
|------|-------------|----------------|
| `INTEGER` | Whole numbers | `1`, `42`, `-5` |
| `REAL` | Floating-point numbers | `3.14`, `99.95` |
| `TEXT` | Strings (any length) | `'Chicago'`, `'2024-01-15'` |
| `BLOB` | Binary data (images, files) | Raw bytes |
| `NULL` | Absence of a value | `NULL` |

**Dates and booleans in SQLite:**
- SQLite has no dedicated `DATE` or `BOOLEAN` type.
- **Dates** are stored as `TEXT` in ISO format (`'YYYY-MM-DD'`) — SQLite's date functions work with this format.
- **Booleans** are stored as `INTEGER`: `0` = false, `1` = true.

## Creating Tables

### Basic `CREATE TABLE` Syntax

```sql
CREATE TABLE table_name (
    column1  datatype  constraints,
    column2  datatype  constraints,
    ...
);
```

### A Simple Example

```sql
CREATE TABLE products (
    product_id  INTEGER  PRIMARY KEY,
    name        TEXT     NOT NULL,
    price       REAL     NOT NULL,
    in_stock    INTEGER  DEFAULT 1
);
```

- `product_id` is the primary key — SQLite auto-increments it when you insert a row without specifying it.
- `name` and `price` are required (`NOT NULL`).
- `in_stock` defaults to `1` (true) if not specified.

### `IF NOT EXISTS`

Always include `IF NOT EXISTS` to avoid an error if the table already exists:

```sql
CREATE TABLE IF NOT EXISTS products (
    product_id  INTEGER  PRIMARY KEY,
    name        TEXT     NOT NULL,
    price       REAL     NOT NULL,
    in_stock    INTEGER  DEFAULT 1
);
```

## Constraints

Constraints are rules applied to columns (or combinations of columns) that SQLite enforces on every `INSERT` and `UPDATE`. They are your first line of defense against bad data.

### `NOT NULL`

Requires that a column always have a value:

```sql
CREATE TABLE customers (
    customer_id  INTEGER  PRIMARY KEY,
    name         TEXT     NOT NULL,   -- Name is required
    email        TEXT     NOT NULL,   -- Email is required
    city         TEXT                 -- City is optional (NULL allowed)
);
```

### `UNIQUE`

Prevents duplicate values in a column across all rows:

```sql
CREATE TABLE customers (
    customer_id  INTEGER  PRIMARY KEY,
    name         TEXT     NOT NULL,
    email        TEXT     NOT NULL  UNIQUE   -- No two customers can share an email
);
```

### `DEFAULT`

Supplies a value when none is provided during insert:

```sql
CREATE TABLE books (
    book_id    INTEGER  PRIMARY KEY,
    title      TEXT     NOT NULL,
    stock      INTEGER  DEFAULT 0,          -- Defaults to 0 if not specified
    available  INTEGER  DEFAULT 1           -- Defaults to 1 (true)
);
```

### `CHECK`

Validates that a value meets a condition — like a data entry rule built into the schema:

```sql
CREATE TABLE books (
    book_id  INTEGER  PRIMARY KEY,
    title    TEXT     NOT NULL,
    price    REAL     NOT NULL  CHECK (price > 0),       -- Price must be positive
    stock    INTEGER  DEFAULT 0  CHECK (stock >= 0),     -- Stock can't go negative
    rating   REAL     CHECK (rating BETWEEN 1.0 AND 5.0) -- Rating range enforced
);
```

Attempting to insert a book with `price = -5` will raise an error.

## Primary Keys

A primary key uniquely identifies each row. In SQLite, declaring a column as `INTEGER PRIMARY KEY` creates an **auto-incrementing** ID — SQLite assigns the next available integer automatically when you insert without specifying it:

```sql
-- Explicitly providing the ID:
INSERT INTO customers (customer_id, name, email) VALUES (1, 'Alex', 'alex@email.com');

-- Letting SQLite auto-assign the ID:
INSERT INTO customers (name, email) VALUES ('Jordan', 'jordan@email.com');
-- SQLite assigns customer_id = 2 automatically
```

### Composite Primary Keys

A primary key can span multiple columns when no single column is unique on its own:

```sql
CREATE TABLE order_items (
    order_id   INTEGER,
    product_id INTEGER,
    quantity   INTEGER  NOT NULL  DEFAULT 1,
    PRIMARY KEY (order_id, product_id)   -- The combination must be unique
);
```

## Foreign Keys

A foreign key enforces a relationship between tables. It ensures that a value in one table's column matches an existing value in another table's primary key — you can't create an order for a customer who doesn't exist.

```sql
CREATE TABLE orders (
    order_id     INTEGER  PRIMARY KEY,
    customer_id  INTEGER  NOT NULL,
    book_id      INTEGER  NOT NULL,
    order_date   TEXT     NOT NULL,
    quantity     INTEGER  NOT NULL  DEFAULT 1,
    FOREIGN KEY (customer_id)  REFERENCES customers(customer_id),
    FOREIGN KEY (book_id)      REFERENCES books(book_id)
);
```

**Important:** SQLite does not enforce foreign keys by default. You must enable enforcement at the start of each connection:

```python
import sqlite3

conn = sqlite3.connect("bookstore.db")
conn.execute("PRAGMA foreign_keys = ON;")   # Enable FK enforcement
cursor = conn.cursor()
```

## Modifying Tables

### Adding a Column

```sql
ALTER TABLE customers ADD COLUMN phone TEXT;
```

### Dropping a Table

```sql
DROP TABLE IF EXISTS old_table;
```

## Inspecting a Table's Schema

To see how a table was defined in SQLite, query the `sqlite_master` table:

```python
import sqlite3

with sqlite3.connect("bookstore.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE name = 'orders';")
    print(cursor.fetchone()[0])
```

Output:
```sql
CREATE TABLE orders (
    order_id     INTEGER  PRIMARY KEY,
    customer_id  INTEGER  NOT NULL,
    book_id      INTEGER  NOT NULL,
    order_date   TEXT     NOT NULL,
    quantity     INTEGER  NOT NULL  DEFAULT 1,
    FOREIGN KEY (customer_id)  REFERENCES customers(customer_id),
    FOREIGN KEY (book_id)      REFERENCES books(book_id)
)
```

## Putting It Together: Rebuilding the Bookstore Schema

Here's the complete, annotated schema for the bookstore database used throughout this module:

```python
import sqlite3

with sqlite3.connect("bookstore.db") as conn:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript("""
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id   INTEGER  PRIMARY KEY,
            name          TEXT     NOT NULL,
            email         TEXT     NOT NULL  UNIQUE,
            city          TEXT,
            member_since  TEXT
        );

        CREATE TABLE books (
            book_id   INTEGER  PRIMARY KEY,
            title     TEXT     NOT NULL,
            author    TEXT     NOT NULL,
            genre     TEXT,
            price     REAL     NOT NULL  CHECK (price > 0),
            stock     INTEGER  DEFAULT 0  CHECK (stock >= 0)
        );

        CREATE TABLE orders (
            order_id     INTEGER  PRIMARY KEY,
            customer_id  INTEGER  NOT NULL,
            book_id      INTEGER  NOT NULL,
            order_date   TEXT     NOT NULL,
            quantity     INTEGER  NOT NULL  DEFAULT 1  CHECK (quantity > 0),
            FOREIGN KEY (customer_id)  REFERENCES customers(customer_id),
            FOREIGN KEY (book_id)      REFERENCES books(book_id)
        );
    """)
    print("Schema created successfully.")
```

## Conclusion

In this lesson, you learned how to design a table using `CREATE TABLE` with SQLite's data types (`INTEGER`, `REAL`, `TEXT`, `NULL`) and constraints (`NOT NULL`, `UNIQUE`, `DEFAULT`, `CHECK`). You defined primary keys for unique row identification and foreign keys for inter-table relationships, and you enabled foreign key enforcement with `PRAGMA foreign_keys = ON`. A well-constrained schema is the foundation of reliable data — it prevents bad data from ever entering the database. In the next lesson, you'll learn how to read data out of these tables using `SELECT` queries.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="SQLite has no built-in `DATE` or `DATETIME` type. The standard convention is to store dates as `TEXT` in ISO 8601 format (`&#039;YYYY-MM-DD&#039;`), which sorts correctly as a string and works with SQLite&#039;s built-in date functions like `date()`, `strftime()`, and `datetime()`.">
  <div class="quiz-question">
    <strong>Question 1:</strong> How does SQLite store dates?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>In a dedicated `DATE` column type that formats values automatically.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>As `TEXT` in `'YYYY-MM-DD'` format, `REAL` as Julian day numbers, or `INTEGER` as Unix timestamps.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Only as integers representing the number of days since January 1, 1970.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>SQLite cannot store dates — you must use a separate string-formatting library.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="In SQLite, `INTEGER PRIMARY KEY` is a special declaration that aliases the column to SQLite&#039;s internal row ID. When you insert a row without providing a value for this column, SQLite assigns the next available integer automatically. This is SQLite&#039;s equivalent of `SERIAL` in PostgreSQL or `AUTO_INCREMENT` in MySQL.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does `INTEGER PRIMARY KEY` do in SQLite specifically?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It ensures the column only stores positive integers.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It makes the column an auto-incrementing unique identifier — SQLite automatically assigns the next available integer when you insert a row without specifying the value.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It creates an index on the column to speed up queries.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It prevents the column from being updated after the row is inserted.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When `PRAGMA foreign_keys = ON` is set, SQLite enforces referential integrity. Any attempt to insert a row with a foreign key value that doesn&#039;t match an existing primary key in the referenced table will raise an `IntegrityError`. This is the correct behavior — it prevents orphaned records (orders with no customer) from entering the database.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You try to insert an order with `customer_id = 99` but no customer with that ID exists. Foreign keys are enabled. What happens?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The insert succeeds and a new customer with ID 99 is created automatically.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>SQLite raises an error because the foreign key constraint is violated — `customer_id = 99` does not exist in the customers table.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>SQLite silently ignores the invalid `customer_id` and stores `NULL` instead.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>SQLite disables the foreign key check for this insert only.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

