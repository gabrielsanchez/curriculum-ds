# Module Assessment

## Overview

Congratulations on completing the Introduction to Python for Data Science module! You've covered a lot of ground — from variables and data types, through functions, strings, conditional logic, lists, and dictionaries, to the fundamentals of object-oriented programming. This assessment is your opportunity to demonstrate that you can apply these skills together to solve realistic data problems.

## Learning Objective

By the end of this assessment, you will have demonstrated your ability to:

- Validate Python fluency (variables, functions, data structures).
- Practice writing clean, modular code.

## Key terms

**Modular code:** Code that is organized into small, focused functions and classes, making it easier to read, test, and reuse.

**Data pipeline:** A series of processing steps applied to raw data to produce a clean, analysis-ready result.

## Assessment Overview

This module assessment is a coding project completed in a Google Colaboratory notebook. You will work with a small dataset and complete a series of tasks that exercise each skill area from the module.

### Skills Assessed

| Skill                  | Lesson          |
|------------------------|-----------------|
| Variables & data types | Intro to Python |
| Functions              | Functions       |
| String manipulation    | Strings         |
| Conditional logic      | Conditional Logic |
| List operations        | Lists           |
| Dictionary operations  | Dictionaries    |
| Classes & objects      | Objects         |

## Coding Assessment

Complete the project in this [notebook](#). The notebook contains a series of guided tasks. After completing all tasks, save your notebook to GitHub and submit the link for grading.

### Task Overview

The notebook will walk you through:

1. **Loading and inspecting raw data** — You will be given a list of dictionaries representing student records. Use variables, `print()`, and `len()` to explore the dataset.

2. **Cleaning string data** — Write a function that accepts a student record and returns a cleaned version with names properly capitalized and extra whitespace removed.

3. **Classifying records with conditional logic** — Write a function that accepts a GPA value and returns the corresponding letter grade using `if/elif/else` logic.

4. **Filtering and transforming with lists** — Use list comprehensions to extract only the students who are passing (GPA ≥ 2.0) and to compute a curved score list.

5. **Summarizing with dictionaries** — Build a summary dictionary that counts how many students received each letter grade.

6. **Building a class** — Define a `StudentRoster` class with an `__init__()` method that stores a list of student records, and at least two methods: one that prints a summary report and one that returns the student with the highest GPA.

### Grading Rubric

| Criteria                                               | Points |
|--------------------------------------------------------|--------|
| Variables and data types used correctly                | 10     |
| Cleaning function works correctly with edge cases      | 15     |
| Grade classification function handles all ranges       | 15     |
| List comprehensions produce correct results            | 15     |
| Summary dictionary is accurate                        | 15     |
| `StudentRoster` class is correctly defined and working | 20     |
| Code is readable, well-named, and modular             | 10     |
| **Total**                                              | **100** |

## Knowledge Check

#### **Question 1: Which of the following best describes "modular code"?**
1. Code that is written in a single long script without functions.
2. Code organized into small, focused functions and classes that each do one thing well.
3. Code that only uses built-in Python functions and no custom logic.
4. Code that is compressed to use as few lines as possible.

**Correct Answer:**
2. Code organized into small, focused functions and classes that each do one thing well.

**Explanation:**
Modular code is broken into reusable pieces — functions, classes, or modules — where each piece has a single, clear responsibility. This makes it easier to read, debug, and reuse. The opposite — putting all logic in one long script — is sometimes called "spaghetti code."

---

#### **Question 2: A teammate's function uses a variable named `x` to store a student's GPA. What is a better variable name and why?**
1. `g` — shorter is always better.
2. `gpa` — it is descriptive and makes the code's intent immediately clear.
3. `GPA_VALUE_123` — more characters means more descriptive.
4. `var1` — generic names prevent naming conflicts.

**Correct Answer:**
2. `gpa` — it is descriptive and makes the code's intent immediately clear.

**Explanation:**
Descriptive variable names are a key part of writing readable, maintainable code. `x` or `var1` give the reader no information about what the variable holds. `gpa` immediately communicates the meaning, making the code self-documenting.

---

#### **Question 3: You have a list of student records (each a dictionary) and want to collect only those with a GPA above 3.5. Which approach is most Pythonic?**
1. Write a `for` loop that manually appends matching records to a new list.
2. Use a list comprehension: `[s for s in students if s["gpa"] > 3.5]`
3. Delete all non-matching records from the original list using `remove()`.
4. Sort the list by GPA and slice the top records.

**Correct Answer:**
2. Use a list comprehension: `[s for s in students if s["gpa"] > 3.5]`

**Explanation:**
List comprehensions are the idiomatic Python way to filter and transform collections. They are concise, readable, and create a new list without modifying the original. While a `for` loop also works, the list comprehension expresses the intent more cleanly in a single line.
