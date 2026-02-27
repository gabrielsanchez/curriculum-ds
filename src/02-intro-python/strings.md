# Strings

## Overview

In the previous lesson, you learned how to write functions to organize and reuse code. Now you'll take a closer look at one of Python's most important data types: **strings**. Text data is everywhere in data science — from survey responses and product reviews to social media posts and patient notes. In this lesson, you'll learn how to work with and transform string data in Python.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Manipulate strings (concatenation, slicing).
- Understand common string methods (replace, split, strip).

## Key terms

**String:** A sequence of characters enclosed in single or double quotes, used to represent text.

**Concatenation:** Joining two or more strings together end-to-end using the `+` operator.

**Slicing:** Extracting a portion of a string by specifying a start and end index.

**Index:** The numeric position of a character in a string, starting at `0` for the first character.

**String method:** A built-in function that operates on a string, such as `.upper()`, `.replace()`, or `.split()`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/02-intro-python/02_strings_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

In data science, a significant portion of real-world data comes in the form of text. Before you can analyze text — whether you're counting words, cleaning survey responses, or preparing data for a machine learning model — you need to be comfortable working with strings. Python provides a rich set of tools to create, combine, and transform string data with minimal code.

## Creating Strings

You can create a string by enclosing characters in single quotes, double quotes, or triple quotes (for multi-line strings):

```python
name = "Alex"
city = 'Chicago'
bio = """Alex is a data science student
who lives in Chicago."""

print(name)  # Output: Alex
print(bio)
# Output:
# Alex is a data science student
# who lives in Chicago.
```

## String Concatenation

### Joining Strings with `+`

Concatenation lets you combine strings into one:

```python
first_name = "Data"
last_name = "Science"
full_name = first_name + " " + last_name
print(full_name)  # Output: Data Science
```

### f-Strings (Formatted String Literals)

A cleaner and more readable way to embed variables inside strings is using **f-strings**, introduced in Python 3.6:

```python
name = "Alex"
age = 25
message = f"My name is {name} and I am {age} years old."
print(message)  # Output: My name is Alex and I am 25 years old.
```

Wrapping a string with `f` before the opening quote lets you insert any expression inside `{}`.

## String Indexing and Slicing

### Accessing Individual Characters

Each character in a string has an **index** starting at `0`. You can access a character by placing its index in square brackets:

```python
word = "Python"
print(word[0])   # Output: P
print(word[1])   # Output: y
print(word[-1])  # Output: n  (negative index counts from the end)
```

### Slicing a String

Use slicing to extract a substring. The syntax is `string[start:end]`, where `start` is inclusive and `end` is exclusive:

```python
word = "Python"
print(word[0:3])   # Output: Pyt  (characters at index 0, 1, 2)
print(word[2:])    # Output: thon (from index 2 to the end)
print(word[:4])    # Output: Pyth (from start to index 3)
print(word[::2])   # Output: Pto  (every 2nd character)
```

A practical example with text data:

```python
date = "2024-03-15"
year = date[:4]
month = date[5:7]
day = date[8:]
print(year, month, day)  # Output: 2024 03 15
```

## Common String Methods

Python strings come with many built-in methods. A **method** is called using dot notation: `string.method()`.

### Changing Case

```python
text = "Hello, World!"
print(text.upper())   # Output: HELLO, WORLD!
print(text.lower())   # Output: hello, world!
print(text.title())   # Output: Hello, World!
```

### `strip()` — Removing Whitespace

The `strip()` method removes leading and trailing whitespace (spaces, tabs, newlines). This is extremely useful when cleaning raw data:

```python
raw = "   data science   "
clean = raw.strip()
print(clean)           # Output: data science
print(len(raw))        # Output: 18
print(len(clean))      # Output: 12
```

Use `lstrip()` to strip only the left side, or `rstrip()` for only the right side.

### `replace()` — Substituting Text

`replace(old, new)` returns a new string with all occurrences of `old` replaced by `new`:

```python
sentence = "I love cats. Cats are great."
updated = sentence.replace("cats", "dogs")
print(updated)  # Output: I love dogs. Cats are great.
```

Note that `replace()` is case-sensitive — `"cats"` was replaced but `"Cats"` was not.

### `split()` — Breaking a String into Parts

`split(separator)` divides a string at every occurrence of the separator and returns a **list** of the resulting pieces:

```python
csv_row = "Alex,25,Chicago,Data Science"
fields = csv_row.split(",")
print(fields)
# Output: ['Alex', '25', 'Chicago', 'Data Science']

print(fields[0])  # Output: Alex
print(fields[2])  # Output: Chicago
```

Splitting on whitespace (the default) is useful for tokenizing sentences:

```python
sentence = "Python is great for data science"
words = sentence.split()
print(words)
# Output: ['Python', 'is', 'great', 'for', 'data', 'science']
print(len(words))  # Output: 6
```

### Checking String Contents

```python
text = "Hello123"
print(text.startswith("Hello"))  # Output: True
print(text.endswith("456"))      # Output: False
print("123" in text)             # Output: True
print(text.isdigit())            # Output: False
```

## Putting It Together: Cleaning Text Data

Here's a realistic example that combines several string techniques to clean a messy data entry:

```python
raw_entry = "  john DOE , data scientist , new york  "

# Strip whitespace, then split on comma
fields = raw_entry.strip().split(",")

# Clean each field
name = fields[0].strip().title()
role = fields[1].strip().title()
location = fields[2].strip().title()

print(f"Name: {name}")        # Output: Name: John Doe
print(f"Role: {role}")        # Output: Role: Data Scientist
print(f"Location: {location}")# Output: Location: New York
```

## Conclusion

In this lesson, you learned how to work with strings in Python — how to combine them with concatenation and f-strings, extract portions using indexing and slicing, and transform them using methods like `strip()`, `replace()`, and `split()`. These skills are fundamental to cleaning and preparing text data, a task you'll encounter constantly in data science. In the next lesson, you'll learn how to control the flow of your programs using conditional logic.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/02-intro-python/02_strings_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="String slicing with `[1:4]` extracts characters at indices 1, 2, and 3 (the end index 4 is **exclusive**). In `&quot;Python&quot;`, index 1 is `y`, index 2 is `t`, and index 3 is `h`, giving `&quot;yth&quot;`.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What does the following code output?
  </div>
  <div class="quiz-subquestion">
    <pre><code>word = &quot;Python&quot;
print(word[1:4])</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`Pyt`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`yth`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`ytho`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>`ython`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="The `strip()` method removes spaces, tabs, and newlines from the beginning and end of a string. It does not affect whitespace in the middle. This is especially useful when cleaning data loaded from CSV files or user input.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does the `strip()` method do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Removes all vowels from a string.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Splits a string into a list of words.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Removes leading and trailing whitespace from a string.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Replaces all spaces inside a string with underscores.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="`split(&quot;,&quot;)` divides the string at each comma, producing the list `[&#039;apple&#039;, &#039;banana&#039;, &#039;cherry&#039;]`. Indexing with `[2]` accesses the third element (index starts at 0), which is `&#039;cherry&#039;`.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What is the output of the following code?
  </div>
  <div class="quiz-subquestion">
    <pre><code>text = &quot;apple,banana,cherry&quot;
items = text.split(&quot;,&quot;)
print(items[2])</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>`apple`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>`banana`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>`cherry`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>`apple,banana`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

