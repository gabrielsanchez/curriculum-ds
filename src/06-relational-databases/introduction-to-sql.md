# Introduction to SQL

## Overview

Welcome to the Relational Databases module! In the previous modules, you worked with data stored in files — CSVs, Excel sheets, and JSON. In professional settings, most structured data lives in **relational databases**, and the language used to interact with them is **SQL**. In this lesson, you'll learn what relational databases are, why they matter for data science, and the fundamentals of SQL syntax.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Understand the basics of SQL syntax and relational databases.

## Key terms

**Relational database:** A structured system for storing data organized into tables with rows and columns, where relationships between tables are defined using keys.

**SQL (Structured Query Language):** The standard language used to create, read, update, and delete data in a relational database. Pronounced "sequel" or "S-Q-L."

**Table:** The fundamental storage unit in a relational database — similar to a spreadsheet tab. Each table has a fixed set of columns and any number of rows.

**Row (record):** A single entry in a table, representing one instance of the entity the table describes (e.g., one customer or one order).

**Column (field):** A single attribute of the table (e.g., `customer_name`, `order_date`). Every row has a value for each column.

**Query:** A SQL statement sent to a database that retrieves or manipulates data. The most common is the `SELECT` query for reading data.

**Schema:** The structure of a database — the tables it contains, the columns in each table, their data types, and the relationships between tables.

**Primary key:** A column (or combination of columns) whose value uniquely identifies each row in a table. No two rows can share the same primary key value.

**Foreign key:** A column in one table that references the primary key of another table, creating a link between the two tables.

## Introduction

When data grows beyond what comfortably fits in a spreadsheet — thousands of customers, millions of transactions, dozens of interconnected categories — a **relational database** becomes essential. Databases store data reliably, allow multiple users to query it simultaneously, enforce data integrity rules, and enable you to answer complex questions by combining information from multiple tables in a single query.

SQL is the universal language for working with relational databases. Whether you're querying a small SQLite file or a multi-terabyte data warehouse, the core SQL syntax is the same. For data scientists, SQL is one of the most practical skills you can develop — in many organizations, SQL is the primary tool for extracting the data that gets loaded into Python for analysis.

## The Relational Model

The foundation of relational databases is the **relational model**, introduced by Edgar F. Codd in 1970. The core idea is simple: represent data as a set of tables (called **relations**), and express relationships between entities by storing matching key values rather than embedding one table's data inside another.

Consider a simple e-commerce system:

**`customers` table:**
| customer_id | name    | city      |
|-------------|---------|-----------|
| 1           | Alex    | Chicago   |
| 2           | Jordan  | New York  |
| 3           | Morgan  | Austin    |

**`orders` table:**
| order_id | customer_id | order_date | total |
|----------|-------------|------------|-------|
| 101      | 1           | 2024-01-15 | 89.99 |
| 102      | 2           | 2024-01-16 | 45.00 |
| 103      | 1           | 2024-02-01 | 120.50 |

Rather than repeating Alex's name and city in every order row, we store Alex's information once in `customers` and reference her `customer_id` in each `orders` row. This avoids duplication, reduces storage, and ensures that updating Alex's address requires changing exactly one row.

## Why SQL for Data Science?

- **Data lives in databases.** Most production data in companies — sales, users, events, inventory — is stored in relational databases. SQL is the key to accessing it.
- **Scale.** SQL databases are optimized for queries over millions or billions of rows — far beyond what Pandas handles comfortably in memory.
- **Expressiveness.** Complex questions that would take dozens of lines of Python can often be expressed in a single SQL query.
- **Universality.** The core SQL you learn here transfers directly to PostgreSQL, MySQL, Microsoft SQL Server, BigQuery, Snowflake, and more.

## SQL Statement Types

SQL statements fall into four main categories:

| Category | Name | Common statements | Purpose |
|----------|------|-------------------|---------|
| **DDL** | Data Definition Language | `CREATE`, `DROP`, `ALTER` | Define table structure |
| **DML** | Data Manipulation Language | `INSERT`, `UPDATE`, `DELETE` | Modify data |
| **DQL** | Data Query Language | `SELECT` | Read data |
| **DCL** | Data Control Language | `GRANT`, `REVOKE` | Manage permissions |

As a data scientist, you'll primarily use **DQL** (reading data with `SELECT`) and occasionally **DDL** (creating tables) and **DML** (inserting data for testing).

## Basic SQL Syntax

SQL statements are written in plain English-like phrases and end with a semicolon. Keywords are conventionally written in uppercase (though SQL is case-insensitive for keywords):

```sql
SELECT column1, column2
FROM table_name
WHERE condition;
```

A complete example — "Show me the name and city of every customer in Chicago":

```sql
SELECT name, city
FROM customers
WHERE city = 'Chicago';
```

Result:
```
name  | city
------+---------
Alex  | Chicago
```

SQL reads naturally from top to bottom:
1. **SELECT** — which columns do you want?
2. **FROM** — from which table?
3. **WHERE** — what condition must rows meet to be included?

You'll learn each clause in depth in the lessons that follow. For now, notice that SQL describes **what** you want, not **how** to find it — the database engine figures out the most efficient way to retrieve the data.

## The Tool: SQLite

Throughout this module, you'll use **SQLite** — a lightweight, serverless relational database engine that is built directly into Python's standard library. Unlike PostgreSQL or MySQL, SQLite requires no installation, no server process, and no configuration. A database is simply a single `.db` file on your computer.

SQLite is:
- **Zero-configuration** — works out of the box in any Python environment, including Google Colab.
- **Industry-standard SQL** — the SQL you write here transfers directly to other databases.
- **Widely used** — SQLite is the most deployed database in the world, built into iOS, Android, Firefox, and many more.

In the next lesson, you'll set up your SQLite environment and create your first database.

## Conclusion

In this lesson, you learned what relational databases are, why they're foundational to data science, how the relational model links tables through keys, and the basic structure of SQL statements. You were introduced to SQLite as the database engine for this module. In the next lesson, you'll set up SQLite, create a database, and build your first tables.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="A foreign key is a column in one table that holds a value matching the primary key of a related table. This relationship allows you to connect information across tables — for example, linking each `order` row to the `customer` who placed it — without duplicating data. It is the mechanism that makes the relational model powerful.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the purpose of a foreign key in a relational database?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>It provides a unique identifier for each row in a table.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>It stores encrypted passwords for database access.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>It references the primary key of another table, creating a link between the two tables.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>It defines which columns are required to have a value.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`SELECT` reads data. `FROM customers` specifies the source table. `WHERE city = &#039;Austin&#039;` filters to include only rows where the city column equals &#039;Austin&#039;. The result is a set of rows showing `name` and `city` for every Austin customer — no data is modified.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does the following SQL query do?
  </div>
  <div class="quiz-subquestion">
    <pre><code>SELECT name, city
FROM customers
WHERE city = &#039;Austin&#039;;</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It updates the city column to 'Austin' for all customers named 'name'.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It retrieves the name and city columns for all rows in the customers table where city is 'Austin'.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It deletes all customers who do not live in Austin.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It creates a new table containing only customers from Austin.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="SQLite&#039;s key advantage for learning is its zero-configuration setup — you can create and query a database with a single line of Python without installing anything. The SQL syntax is standard, so skills learned in SQLite transfer directly to PostgreSQL, MySQL, and cloud data warehouses. And contrary to option 4, SQLite is extensively used in production — it powers the storage layer of iOS apps, Android apps, web browsers, and many more applications.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Why is SQLite a good choice for learning SQL in a data science context?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>SQLite is the only database that supports Python.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>SQLite requires no server installation or configuration, is built into Python's standard library, and supports standard SQL that transfers to other databases.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>SQLite is faster than all other database systems for large datasets.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>SQLite is only suitable for learning — it is never used in production applications.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

