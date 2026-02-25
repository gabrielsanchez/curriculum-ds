# Dictionaries

## Overview

In the previous lesson, you learned how to use lists to store ordered collections of values. Lists work well when the position of items matters, but sometimes you need to look data up by a meaningful name rather than a number. In this lesson, you'll learn about **dictionaries** — Python's structure for storing data as labeled key-value pairs.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Store and retrieve data using key-value pairs.
- Update, delete, and iterate through dictionaries.

## Key terms

**Dictionary:** An unordered, mutable collection of key-value pairs enclosed in curly braces `{}`.

**Key:** A unique identifier used to look up a value in a dictionary. Keys are typically strings or integers.

**Value:** The data associated with a key. Values can be any Python object, including lists or other dictionaries.

**Key-value pair:** A single entry in a dictionary, written as `key: value`.

**`get()`:** A dictionary method that retrieves the value for a key, returning `None` (or a default) if the key does not exist — avoiding errors.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Imagine you have a spreadsheet where each row represents a student. Instead of tracking which column number holds the student's name versus their score, you could label each piece of data — `"name"`, `"score"`, `"grade"` — and look things up by label. That's exactly what a **dictionary** does.

Dictionaries are one of the most widely used data structures in Python. In data science, they appear naturally when loading JSON data from APIs, representing rows in a dataset, or building lookup tables. Because dictionaries use meaningful keys instead of numeric indices, they make code easier to read and maintain.

## Creating a Dictionary

Create a dictionary with curly braces `{}`, listing key-value pairs separated by commas:

```python
student = {
    "name": "Alex",
    "age": 25,
    "major": "Data Science",
    "gpa": 3.8
}

print(student)
# Output: {'name': 'Alex', 'age': 25, 'major': 'Data Science', 'gpa': 3.8}
```

Keys must be **unique** within a dictionary. If you repeat a key, the later value overwrites the earlier one.

## Accessing Values

### Using Square Bracket Notation

Retrieve a value by its key:

```python
print(student["name"])   # Output: Alex
print(student["gpa"])    # Output: 3.8
```

If the key does not exist, Python raises a `KeyError`. To avoid this, use the `get()` method:

### Using `get()`

```python
print(student.get("name"))          # Output: Alex
print(student.get("scholarship"))   # Output: None  (key doesn't exist, no error)
print(student.get("scholarship", "Not awarded"))  # Output: Not awarded
```

The optional second argument to `get()` is the **default value** returned when the key is missing.

## Modifying a Dictionary

### Adding or Updating a Key-Value Pair

Assign a value to a key — if the key exists, it's updated; if not, it's added:

```python
student = {"name": "Alex", "age": 25}

# Add a new key
student["major"] = "Data Science"

# Update an existing key
student["age"] = 26

print(student)
# Output: {'name': 'Alex', 'age': 26, 'major': 'Data Science'}
```

### Deleting a Key-Value Pair

Use the `del` keyword or the `pop()` method:

```python
student = {"name": "Alex", "age": 26, "major": "Data Science", "gpa": 3.8}

# del removes the key-value pair
del student["gpa"]
print(student)
# Output: {'name': 'Alex', 'age': 26, 'major': 'Data Science'}

# pop() removes and returns the value
age = student.pop("age")
print(age)      # Output: 26
print(student)  # Output: {'name': 'Alex', 'major': 'Data Science'}
```

## Checking Keys

Use the `in` operator to check whether a key exists:

```python
student = {"name": "Alex", "major": "Data Science"}

print("name" in student)    # Output: True
print("gpa" in student)     # Output: False
```

## Iterating Through a Dictionary

### Iterating Over Keys

```python
scores = {"math": 90, "science": 85, "english": 78}

for subject in scores:
    print(subject)
# Output:
# math
# science
# english
```

### Iterating Over Values

```python
for score in scores.values():
    print(score)
# Output:
# 90
# 85
# 78
```

### Iterating Over Key-Value Pairs

The `.items()` method returns each key-value pair as a tuple, which you can unpack:

```python
for subject, score in scores.items():
    print(f"{subject}: {score}")
# Output:
# math: 90
# science: 85
# english: 78
```

## Useful Dictionary Methods

```python
scores = {"math": 90, "science": 85, "english": 78}

print(scores.keys())    # Output: dict_keys(['math', 'science', 'english'])
print(scores.values())  # Output: dict_values([90, 85, 78])
print(len(scores))      # Output: 3
```

### `update()` — Merging Dictionaries

```python
student = {"name": "Alex", "age": 25}
extra_info = {"major": "Data Science", "gpa": 3.8}

student.update(extra_info)
print(student)
# Output: {'name': 'Alex', 'age': 25, 'major': 'Data Science', 'gpa': 3.8}
```

## Nested Dictionaries

Dictionary values can themselves be dictionaries, allowing you to represent hierarchical data:

```python
class_roster = {
    "student_1": {"name": "Alex", "grade": "A"},
    "student_2": {"name": "Jordan", "grade": "B"},
    "student_3": {"name": "Morgan", "grade": "A"}
}

# Access nested data
print(class_roster["student_2"]["name"])   # Output: Jordan
print(class_roster["student_1"]["grade"])  # Output: A
```

## Putting It Together: A Data Science Example

Here's a practical example that uses a list of dictionaries — a common way to represent tabular data before loading it into a DataFrame:

```python
dataset = [
    {"city": "Chicago",     "population": 2696000, "avg_temp_f": 49},
    {"city": "Los Angeles", "population": 3979000, "avg_temp_f": 66},
    {"city": "New York",    "population": 8336000, "avg_temp_f": 55},
    {"city": "Houston",     "population": 2304000, "avg_temp_f": 68},
]

# Find the city with the highest population
largest = max(dataset, key=lambda row: row["population"])
print(f"Largest city: {largest['city']} ({largest['population']:,} people)")
# Output: Largest city: New York (8,336,000 people)

# Calculate average temperature across all cities
avg_temp = sum(row["avg_temp_f"] for row in dataset) / len(dataset)
print(f"Average temperature: {avg_temp:.1f}°F")
# Output: Average temperature: 59.5°F
```

## Conclusion

In this lesson, you learned how to create dictionaries and access their values using keys, how to add, update, and delete entries, and how to iterate through keys, values, and key-value pairs. Dictionaries are indispensable in data science — you'll encounter them when working with JSON data from APIs, building feature mappings, and representing structured records. In the next lesson, you'll learn about **objects**, which take the idea of bundling related data to the next level with Python's object-oriented programming model.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: How do you safely retrieve a value from a dictionary without raising an error if the key doesn't exist?**
1. Use square bracket notation: `d["key"]`
2. Use the `get()` method: `d.get("key")`
3. Use the `find()` method: `d.find("key")`
4. Use the `fetch()` method: `d.fetch("key")`

**Correct Answer:**
2. Use the `get()` method: `d.get("key")`

**Explanation:**
Using `d["key"]` raises a `KeyError` if the key is not present. The `get()` method avoids this by returning `None` (or a custom default value) when the key doesn't exist, making your code more robust and error-tolerant.

---

#### **Question 2: What does the `.items()` method return when iterating over a dictionary?**
1. Only the keys of the dictionary.
2. Only the values of the dictionary.
3. Both the keys and values as key-value tuples.
4. The number of items in the dictionary.

**Correct Answer:**
3. Both the keys and values as key-value tuples.

**Explanation:**
The `.items()` method returns each entry as a `(key, value)` tuple. When used in a `for` loop — `for key, value in d.items()` — you can unpack both the key and value in each iteration, making it easy to process all entries.

---

#### **Question 3: What happens when you assign a value to a key that already exists in a dictionary?**
1. Python raises a `KeyError`.
2. A new key is created and both values are kept.
3. The existing value is overwritten with the new value.
4. The new value is added to a list alongside the old value.

**Correct Answer:**
3. The existing value is overwritten with the new value.

**Explanation:**
Dictionary keys must be unique. If you write `d["age"] = 30` and `"age"` already exists, Python silently replaces the old value with `30`. No error is raised, and no duplicate key is created.
