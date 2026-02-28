# Relational Databases

Not all data lives in CSV files. Much of the data you will encounter in practice is stored in relational databases and accessed through SQL — the standard language for querying structured data. This module introduces relational database concepts and the SQL skills needed to extract, filter, aggregate, and join data from tables.

The module uses SQLite, which runs entirely within Python with no server setup required, and concludes by connecting SQL queries back to Pandas — the workflow you will use in real data pipelines.

## Lessons

1. **Introduction to SQL** — What relational databases are, when to use SQL versus Pandas, and your first SELECT queries.
2. **Setting Up SQLite** — Connecting to a SQLite database in Python using `sqlite3` and loading data into tables.
3. **Tables, Data Types, and Fields** — How relational data is structured: tables, rows, columns, primary keys, and data types.
4. **Querying Data** — Filtering rows with WHERE, working with NULL values, and combining conditions.
5. **Advanced Queries** — Subqueries, CASE expressions, and LIMIT for exploratory work.
6. **Grouping and Sorting** — Aggregating with GROUP BY and HAVING; ordering results with ORDER BY.
7. **Joining Tables** — INNER, LEFT, RIGHT, and FULL JOIN for combining data across multiple tables.
8. **SQL and Python** — Executing SQL queries from Python, loading results into Pandas DataFrames, and building end-to-end data pipelines.
9. **Module Assessment** — A SQL assessment requiring you to query a multi-table database and extract specific analytical results.
