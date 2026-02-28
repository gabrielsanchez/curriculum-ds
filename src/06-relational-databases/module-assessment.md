# Module Assessment

## Overview

Congratulations on completing the Relational Databases module! You've built a SQLite database from scratch, mastered `SELECT` queries with filtering, aggregation, and joins, used advanced techniques like `CASE`, `COALESCE`, and subqueries, and integrated SQL with Python and Pandas. This assessment brings all of those skills together in a single end-to-end project.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/06-relational-databases-module-assessment_starter.ipynb).

## Learning Objective

By the end of this assessment, you will have demonstrated your ability to:

- Create tables, write queries, and join data in a SQLite database.
- Execute SQL queries within Python and integrate results into a Pandas workflow.

## Key terms

**Schema design:** The process of deciding what tables to create, what columns each needs, what data types to use, and how tables relate to each other.

**Data pipeline:** A sequence of steps that loads raw data, transforms it through queries, and produces analysis-ready output.

**Analytical query:** A SQL query designed to answer a specific business or analytical question, often involving joins, aggregations, and filtering.

## Assessment Overview

This module assessment is a coding project completed in a Google Colaboratory notebook. You will design and build a SQLite database for a fictional movie rental service, load it with sample data, and write a series of SQL queries to answer analytical questions — all from within Python.

### Skills Assessed

| Skill | Lesson |
|-------|--------|
| Create a SQLite database and connect with Python | Setting Up SQLite |
| Define tables with appropriate types and constraints | Tables, Data Types, and Fields |
| Write `SELECT` queries with `WHERE` filtering | Querying Data |
| Use `CASE`, `COALESCE`, and SQL functions | Advanced Queries |
| Aggregate with `GROUP BY`, `HAVING`, `ORDER BY` | Grouping and Sorting |
| Combine tables with `JOIN` | Joining Tables |
| Load results into Pandas and visualize | SQL and Python |

## Coding Assessment

Complete the project in this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/06-relational-databases/06-relational-databases-module-assessment_starter.ipynb). After completing all tasks, save your notebook to GitHub and [submit the link for grading](https://ai-grader-pql9.onrender.com/).

### Task Overview

The notebook will guide you through building and analyzing a **movie rental database** with four tables: `customers`, `movies`, `rentals`, and `genres`.

**Task 1 — Schema design and creation:**
Create all four tables using `CREATE TABLE` with appropriate SQLite data types. Apply at minimum: one `PRIMARY KEY` per table, at least two `NOT NULL` constraints, one `UNIQUE` constraint, one `CHECK` constraint, and two `FOREIGN KEY` relationships. Enable foreign key enforcement with `PRAGMA foreign_keys = ON`.

**Task 2 — Load sample data:**
Insert at least 8 customers, 12 movies (across at least 3 genres), and 20 rental records using `executemany()` with parameterized `?` placeholders.

**Task 3 — Basic queries:**
Write five `SELECT` queries that demonstrate: selecting specific columns with aliases, filtering with `WHERE` and at least one of `LIKE`, `IN`, or `BETWEEN`, using `IS NULL` to find incomplete records, and limiting results with `LIMIT`.

**Task 4 — Advanced queries:**
Write three queries using: a `CASE WHEN` expression to classify movies into tiers (e.g., by rental price or rating), `COALESCE` to replace `NULL` values with a default, and a subquery to find movies rented more than the average number of times.

**Task 5 — Aggregation:**
Write queries that use `GROUP BY` to compute: total rentals and revenue per genre, the customer with the highest total spending, and monthly rental counts. Use `HAVING` to filter at least one group result.

**Task 6 — Joins:**
Write three join queries: an `INNER JOIN` showing full rental details (customer name, movie title, genre, rental date, price), a `LEFT JOIN` to find customers who have never rented a movie, and a three-table join that computes total spending per customer per genre.

**Task 7 — Python + Pandas integration:**
Use `pd.read_sql_query()` to load the results of your three-table join into a DataFrame. Then use Pandas to: compute summary statistics on the result, create a bar chart of revenue by genre, and save the DataFrame to a CSV file.

### Grading Rubric

| Criteria | Points |
|----------|--------|
| Schema created with correct types, constraints, and foreign keys | 15 |
| Data inserted using `executemany()` and parameterized queries | 10 |
| Basic queries correct and clearly commented | 10 |
| Advanced queries use `CASE`, `COALESCE`, and subquery correctly | 15 |
| Aggregation queries use `GROUP BY` and `HAVING` correctly | 15 |
| Join queries produce correct results; `LEFT JOIN` finds non-renters | 20 |
| Pandas integration: DataFrame created, chart plotted, CSV saved | 10 |
| Code is clean, readable, and uses descriptive names and aliases | 5 |
| **Total** | **100** |

## Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="The `LEFT JOIN ... WHERE right_key IS NULL` pattern is the standard SQL idiom for finding rows with no match in the related table. Option 1 assumes a `rental_count` column that doesn&#039;t exist. Option 2 uses `INNER JOIN`, which would only return movies that have been rented, making the `IS NULL` condition impossible. Option 4 is close but the `OR movie_id IS NULL` adds an unnecessary condition. Option 3 is the clean, correct approach.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You need to find all movies that have never been rented. Write the correct SQL approach:
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`SELECT * FROM movies WHERE rental_count = 0`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`SELECT m.title FROM movies m INNER JOIN rentals r ON m.movie_id = r.movie_id WHERE r.rental_id IS NULL`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`SELECT m.title FROM movies m LEFT JOIN rentals r ON m.movie_id = r.movie_id WHERE r.rental_id IS NULL`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>`SELECT * FROM movies WHERE movie_id NOT IN (SELECT movie_id FROM rentals) OR movie_id IS NULL`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="If a user types `&#039;); DROP TABLE customers; --` as their name, the f-string produces `INSERT INTO customers (name) VALUES (&#039;&#039;); DROP TABLE customers; --&#039;)`, which could destroy the entire table. The safe version is: `cursor.execute(&quot;INSERT INTO customers (name) VALUES (?)&quot;, (name,))`. The `sqlite3` driver then treats `name` as data, never as executable SQL.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What is wrong with the following Python code that inserts a customer?
  </div>
  <div class="quiz-subquestion">
    <pre><code>name = input(&quot;Enter customer name: &quot;)
cursor.execute(f&quot;INSERT INTO customers (name) VALUES (&#039;{name}&#039;)&quot;)</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`input()` cannot be used in a Python database script.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The f-string embeds user input directly into the SQL string, creating a SQL injection vulnerability. Use `?` placeholders instead.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`cursor.execute()` does not support `INSERT` statements.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The `VALUES` clause needs parentheses around the column name.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`HAVING` is the correct clause for filtering based on aggregate results. It executes after `GROUP BY` has collapsed rows into groups. `WHERE` executes before grouping and cannot reference aggregate functions like `COUNT(*)` — using it would cause a SQL error. The `HAVING` clause is placed after `GROUP BY` and before `ORDER BY` in the query.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A query using `GROUP BY genre` returns one row per genre. You then want to filter to show only genres with more than 5 rentals. Which clause do you add and where?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Add `WHERE rental_count > 5` before `GROUP BY`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Add `HAVING COUNT(*) > 5` after `GROUP BY`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Add `WHERE COUNT(*) > 5` after `GROUP BY`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Add `FILTER (WHERE COUNT(*) > 5)` at the end of the query.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

