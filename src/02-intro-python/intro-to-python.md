# Intro to Python

## Overview

Welcome to the Introduction to Python for Data Science module! Python is the most widely used programming language in data science, valued for its simplicity and powerful libraries. In this first lesson, you'll learn the building blocks of Python: how to store data in variables and how Python classifies different kinds of data using types.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Declare and use variables in Python.
- Understand Python data types (int, float, bool, str).

## Key terms

**Variable:** A named container that stores a value in memory, allowing you to reference and reuse it throughout your program.

**Data type:** A classification that tells Python what kind of value a variable holds and what operations can be performed on it.

**Integer (`int`):** A whole number without a decimal point, such as `5`, `-3`, or `100`.

**Float (`float`):** A number with a decimal point, such as `3.14`, `-0.5`, or `2.0`.

**Boolean (`bool`):** A data type with only two possible values: `True` or `False`.

**String (`str`):** A sequence of characters enclosed in quotes, such as `"hello"` or `"data science"`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

[*Python*](https://www.python.org/) is a beginner-friendly programming language that reads almost like plain English, making it an excellent first language for aspiring data scientists. Before you can analyze data or build models, you need to understand how Python stores and works with information. That starts with two fundamental concepts: **variables** and **data types**.

Think of a variable as a labeled box — you can put a value inside, give the box a name, and then refer to that name whenever you need the value. Data types tell Python (and you!) what kind of thing is inside the box, so Python knows what you can do with it.

## Variables

### Declaring a Variable

In Python, you create a variable by writing its name, followed by an `=` sign, and then the value you want to store. No special keyword is needed — Python figures out the type automatically.

```python
# Declaring variables
name = "Alex"
age = 25
height = 5.9
is_student = True
```

Here, we've created four variables:
- `name` stores the text `"Alex"`
- `age` stores the whole number `25`
- `height` stores the decimal number `5.9`
- `is_student` stores `True`

### Using Variables

Once declared, you can use a variable anywhere in your code by referring to its name:

```python
print(name)        # Output: Alex
print(age)         # Output: 25
print(height)      # Output: 5.9
print(is_student)  # Output: True
```

You can also update a variable's value by simply reassigning it:

```python
age = 26
print(age)  # Output: 26
```

### Naming Rules

Python variable names must follow a few rules:
- Names can only contain letters, numbers, and underscores (`_`)
- Names cannot start with a number
- Names are case-sensitive (`name` and `Name` are different variables)
- Use descriptive names to make your code readable (e.g., `student_count` instead of `sc`)

## Data Types

### Integers (`int`)

Integers are whole numbers — positive, negative, or zero. They are commonly used for counting and indexing.

```python
num_students = 30
temperature = -5
year = 2024

print(type(num_students))  # Output: <class 'int'>
```

The built-in `type()` function tells you the data type of any variable.

### Floats (`float`)

Floats are numbers that include a decimal point. They are used when precision matters, such as measurements or calculations.

```python
pi = 3.14159
average_score = 87.5
price = 19.99

print(type(pi))  # Output: <class 'float'>
```

### Booleans (`bool`)

Booleans can only be `True` or `False`. They are often the result of comparisons and are essential for controlling program flow.

```python
is_raining = False
passed_exam = True

print(type(is_raining))   # Output: <class 'bool'>
print(10 > 5)             # Output: True
print(3 == 7)             # Output: False
```

### Strings (`str`)

Strings are sequences of characters enclosed in either single (`'`) or double (`"`) quotes. They are used to represent text.

```python
greeting = "Hello, World!"
course = 'Data Science'

print(type(greeting))  # Output: <class 'str'>
print(len(greeting))   # Output: 13
```

The `len()` function returns the number of characters in a string.

## Basic Operations

### Arithmetic with Numbers

Python supports standard arithmetic operations on integers and floats:

```python
a = 10
b = 3

print(a + b)   # Addition:       13
print(a - b)   # Subtraction:    7
print(a * b)   # Multiplication: 30
print(a / b)   # Division:       3.3333...
print(a // b)  # Floor division: 3
print(a % b)   # Modulus:        1
print(a ** b)  # Exponentiation: 1000
```

### Combining Types

You can mix integers and floats in calculations — Python automatically produces a float:

```python
result = 5 + 2.0
print(result)       # Output: 7.0
print(type(result)) # Output: <class 'float'>
```

### Type Conversion

You can convert between types using built-in functions:

```python
x = "42"
x_as_int = int(x)       # Convert string to int
y = 3
y_as_float = float(y)   # Convert int to float
z = 100
z_as_str = str(z)       # Convert int to string

print(type(x_as_int))   # Output: <class 'int'>
```

## Conclusion

In this lesson, you learned the foundational concepts of Python programming: how to declare and use variables, and how Python's four basic data types — `int`, `float`, `bool`, and `str` — classify different kinds of data. These building blocks appear in every Python program you will write. In the next lesson, you will learn how to organize and reuse your code by writing functions.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="Python variable names cannot start with a number (`2nd_score`), cannot contain spaces (`second score`), and cannot contain hyphens (`second-score`). Underscores are allowed, making `second_score` the only valid option.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Which of the following is a valid Python variable name?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`2nd_score`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`second score`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`second_score`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>`second-score`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="3" data-explanation="Any number containing a decimal point is treated as a `float` in Python. `3.14` has a decimal, so Python classifies it as a floating-point number rather than an integer.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What data type does Python assign to the value `3.14`?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>`int`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>`str`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>`bool`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>`float`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The `type()` function returns a description of what kind of data a variable holds. For example, `type(42)` returns `&lt;class &#039;int&#039;&gt;` and `type(&quot;hello&quot;)` returns `&lt;class &#039;str&#039;&gt;`. This is useful for debugging and understanding your data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What does the `type()` function do in Python?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>It converts a variable to a different data type.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>It returns the data type of a given value or variable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>It prints the value stored in a variable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It checks whether two variables have the same value.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

