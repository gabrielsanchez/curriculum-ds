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

<div class="quiz-container" data-correct="1" data-explanation="Modular code is broken into reusable pieces — functions, classes, or modules — where each piece has a single, clear responsibility. This makes it easier to read, debug, and reuse. The opposite — putting all logic in one long script — is sometimes called &quot;spaghetti code.&quot;">
  <div class="quiz-question">
    <strong>Question 1:</strong> Which of the following best describes "modular code"?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Code that is written in a single long script without functions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Code organized into small, focused functions and classes that each do one thing well.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Code that only uses built-in Python functions and no custom logic.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Code that is compressed to use as few lines as possible.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Descriptive variable names are a key part of writing readable, maintainable code. `x` or `var1` give the reader no information about what the variable holds. `gpa` immediately communicates the meaning, making the code self-documenting.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A teammate's function uses a variable named `x` to store a student's GPA. What is a better variable name and why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`g` — shorter is always better.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>`gpa` — it is descriptive and makes the code's intent immediately clear.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`GPA_VALUE_123` — more characters means more descriptive.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>`var1` — generic names prevent naming conflicts.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="List comprehensions are the idiomatic Python way to filter and transform collections. They are concise, readable, and create a new list without modifying the original. While a `for` loop also works, the list comprehension expresses the intent more cleanly in a single line.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You have a list of student records (each a dictionary) and want to collect only those with a GPA above 3.5. Which approach is most Pythonic?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Write a `for` loop that manually appends matching records to a new list.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Use a list comprehension: `[s for s in students if s["gpa"] > 3.5]`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Delete all non-matching records from the original list using `remove()`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Sort the list by GPA and slice the top records.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

