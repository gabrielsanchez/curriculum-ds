# Creating, Reading, and Writing Data

## Overview

In the previous lesson, you learned what Pandas is and explored its core data structures — the Series and the DataFrame. You created DataFrames by typing data directly into your code. In practice, data science work almost always starts with **loading data from a file**. In this lesson, you'll learn how to create DataFrames from multiple sources and how to save your work back to disk.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Create DataFrames manually or from dictionaries and lists.
- Read and write CSV, Excel, and JSON files.

## Key terms

**CSV (Comma-Separated Values):** A plain-text file format where each row is a line and values within a row are separated by commas. It is the most common format for sharing tabular data.

**`pd.read_csv()`:** The Pandas function used to load a CSV file into a DataFrame.

**`pd.read_excel()`:** The Pandas function used to load an Excel spreadsheet into a DataFrame.

**`pd.read_json()`:** The Pandas function used to load a JSON file into a DataFrame.

**`df.to_csv()`:** A DataFrame method that saves the DataFrame to a CSV file.

**`index=False`:** A parameter used when writing files to prevent Pandas from writing the DataFrame's row index as an extra column.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

In real-world data science, you rarely create data from scratch — you load it from files that were collected, exported, or shared by others. Pandas makes it straightforward to read data from the most common file formats. With a single line of code you can load a CSV with millions of rows, an Excel report, or a JSON response from a web API into a fully-featured DataFrame, ready for analysis.

Equally important is the ability to **write** processed data back to a file. After cleaning, transforming, or aggregating a dataset, you'll want to save your results so they can be shared or used in a downstream step.

## Creating DataFrames

### From a Dictionary

The most direct way to build a DataFrame in code is from a dictionary of lists:

```python
import pandas as pd

data = {
    "product":  ["Laptop", "Phone", "Tablet", "Monitor"],
    "price":    [999, 699, 499, 349],
    "in_stock": [True, True, False, True]
}

df = pd.DataFrame(data)
print(df)
```

Output:
```
  product  price  in_stock
0  Laptop    999      True
1   Phone    699      True
2  Tablet    499     False
3 Monitor    349      True
```

### From a List of Dictionaries

A list of dictionaries — where each dictionary is one row — is another natural format, especially when you've built records one at a time:

```python
records = [
    {"name": "Alex",   "score": 88, "grade": "B"},
    {"name": "Jordan", "score": 95, "grade": "A"},
    {"name": "Morgan", "score": 72, "grade": "C"},
]

df = pd.DataFrame(records)
print(df)
```

Output:
```
     name  score grade
0    Alex     88     B
1  Jordan     95     A
2  Morgan     72     C
```

### Setting a Custom Index

By default, Pandas assigns a 0-based integer index. You can set a meaningful column as the index:

```python
df = df.set_index("name")
print(df)
```

Output:
```
        score grade
name
Alex       88     B
Jordan     95     A
Morgan     72     C
```

## Reading Files

### Reading a CSV File

`pd.read_csv()` is the most commonly used data-loading function in Pandas:

```python
df = pd.read_csv("students.csv")
print(df.head())
```

**Common parameters:**

```python
# Specify a different delimiter (e.g., tab-separated)
df = pd.read_csv("data.tsv", sep="\t")

# Use a specific column as the index
df = pd.read_csv("students.csv", index_col="student_id")

# Read only specific columns
df = pd.read_csv("students.csv", usecols=["name", "score", "grade"])

# Skip the first 2 rows
df = pd.read_csv("students.csv", skiprows=2)

# Limit to 100 rows (useful for previewing large files)
df = pd.read_csv("students.csv", nrows=100)
```

### Reading from a URL

`pd.read_csv()` also accepts a URL directly, making it easy to load public datasets:

```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(df.shape)   # Output: (891, 12)
print(df.head())
```

### Reading an Excel File

```python
df = pd.read_excel("report.xlsx")

# Read a specific sheet by name
df = pd.read_excel("report.xlsx", sheet_name="Q1 Sales")
```

> **Note:** Reading Excel files requires the `openpyxl` package. Install it with `pip install openpyxl` if needed.

### Reading a JSON File

JSON is a common format for data from web APIs:

```python
df = pd.read_json("data.json")
```

For nested JSON (e.g., API responses), you may need `pd.json_normalize()`:

```python
import json

with open("api_response.json") as f:
    raw = json.load(f)

df = pd.json_normalize(raw["results"])
```

## Writing Files

### Writing to CSV

```python
df.to_csv("output.csv", index=False)
```

The `index=False` parameter prevents Pandas from writing the row index as an extra column in the file — you almost always want this.

```python
# Write only specific columns
df[["name", "score"]].to_csv("scores_only.csv", index=False)

# Append to an existing file
df.to_csv("output.csv", mode="a", header=False, index=False)
```

### Writing to Excel

```python
df.to_excel("output.xlsx", index=False, sheet_name="Results")
```

### Writing to JSON

```python
# Each row becomes a JSON object
df.to_json("output.json", orient="records", indent=2)
```

## Putting It Together: A Realistic Workflow

Here's what a typical data loading and inspection workflow looks like:

```python
import pandas as pd

# 1. Load the data
df = pd.read_csv("sales_data.csv")

# 2. Inspect the data
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nSummary statistics:\n{df.describe()}")

# 3. Check for missing values
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# 4. Save a cleaned subset
clean_df = df.dropna()
clean_df.to_csv("sales_data_clean.csv", index=False)
print(f"\nSaved {len(clean_df)} rows to sales_data_clean.csv")
```

## Conclusion

In this lesson, you learned how to create DataFrames from dictionaries and lists, and how to load data from CSV, Excel, and JSON files using `pd.read_csv()`, `pd.read_excel()`, and `pd.read_json()`. You also learned how to write processed data back to files using `to_csv()` and `to_excel()`. These input/output skills are the entry and exit points of almost every data science project. In the next lesson, you'll learn how to **select** specific rows and columns from a DataFrame and **assign** new data to it.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: Which function do you use to load a CSV file into a Pandas DataFrame?**
1. `pd.load_csv()`
2. `pd.open_csv()`
3. `pd.read_csv()`
4. `pd.import_csv()`

**Correct Answer:**
3. `pd.read_csv()`

**Explanation:**
`pd.read_csv()` is the standard Pandas function for loading CSV files. It accepts a file path or a URL and returns a DataFrame. The other options (`load_csv`, `open_csv`, `import_csv`) do not exist in Pandas.

---

#### **Question 2: Why is `index=False` commonly used when calling `df.to_csv()`?**
1. It speeds up the file-writing process.
2. It prevents the row index from being written as an extra column in the output file.
3. It removes all column headers from the output file.
4. It compresses the CSV file to save disk space.

**Correct Answer:**
2. It prevents the row index from being written as an extra column in the output file.

**Explanation:**
By default, Pandas includes the DataFrame's row index as the first column when writing to CSV. This creates an unnamed column of numbers (0, 1, 2, ...) that is usually unwanted. Passing `index=False` skips writing the index, keeping the output file clean and consistent with the original column structure.

---

#### **Question 3: Which Pandas function would you use to create a DataFrame from a list of dictionaries?**
1. `pd.Series(records)`
2. `pd.DataFrame(records)`
3. `pd.read_json(records)`
4. `pd.from_records(records)`

**Correct Answer:**
2. `pd.DataFrame(records)`

**Explanation:**
`pd.DataFrame()` accepts a list of dictionaries directly, treating each dictionary as one row and using the dictionary keys as column names. This is a very natural way to build a DataFrame when you've collected records one by one, such as from an API response or a loop.
