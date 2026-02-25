# Conditional Logic

## Overview

In the previous lesson, you learned how to manipulate strings and transform text data. So far, your programs have run the same code every time. In this lesson, you'll learn how to make your programs **make decisions** — executing different code depending on whether a condition is true or false.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Use if-else statements and logical operators (and, or, not).
- Control program flow based on boolean conditions.

## Key terms

**Conditional statement:** A programming construct that executes different code blocks based on whether a condition evaluates to `True` or `False`.

**`if` statement:** Runs a block of code only when its condition is `True`.

**`elif`:** Short for "else if" — adds additional conditions to check when the preceding `if` is `False`.

**`else`:** A catch-all block that runs when none of the preceding conditions are `True`.

**Boolean expression:** An expression that evaluates to either `True` or `False`.

**Logical operator:** A keyword (`and`, `or`, `not`) used to combine or negate boolean expressions.

**Comparison operator:** A symbol used to compare two values, such as `==`, `!=`, `>`, `<`, `>=`, `<=`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Every meaningful program needs to respond differently to different situations. A weather app shows an umbrella icon when it's raining but a sun when it's clear. A data cleaning script skips rows that are missing values but processes rows that are complete. This decision-making ability comes from **conditional logic**.

In Python, conditional logic is built around `if`, `elif`, and `else` statements. Combined with logical operators, these tools give you precise control over which code runs and when.

## Comparison Operators

Before writing conditions, you need to know how to compare values. Python's comparison operators return `True` or `False`:

| Operator | Meaning                  | Example     | Result  |
|----------|--------------------------|-------------|---------|
| `==`     | Equal to                 | `5 == 5`    | `True`  |
| `!=`     | Not equal to             | `5 != 3`    | `True`  |
| `>`      | Greater than             | `7 > 3`     | `True`  |
| `<`      | Less than                | `2 < 1`     | `False` |
| `>=`     | Greater than or equal to | `5 >= 5`    | `True`  |
| `<=`     | Less than or equal to    | `4 <= 3`    | `False` |

```python
score = 85
print(score >= 90)   # Output: False
print(score > 60)    # Output: True
print(score == 100)  # Output: False
```

## The `if` Statement

The simplest conditional runs a block of code only when its condition is `True`:

```python
temperature = 38

if temperature > 37.5:
    print("You have a fever.")
```

The indented code below `if` only runs when the condition evaluates to `True`. If the condition is `False`, Python skips that block entirely.

## `if` / `else`

Add an `else` block to specify what should happen when the condition is `False`:

```python
score = 72

if score >= 60:
    print("You passed!")
else:
    print("You did not pass.")

# Output: You passed!
```

## `if` / `elif` / `else`

Use `elif` (short for "else if") to check multiple conditions in sequence. Python evaluates each condition from top to bottom and executes the first one that is `True`:

```python
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
elif score >= 60:
    print("Grade: D")
else:
    print("Grade: F")

# Output: Grade: B
```

Once one condition matches, Python skips all the remaining conditions.

## Logical Operators

Logical operators let you combine multiple conditions into one expression.

### `and` — Both Conditions Must Be True

```python
age = 22
has_id = True

if age >= 18 and has_id:
    print("Entry allowed.")
else:
    print("Entry denied.")

# Output: Entry allowed.
```

`and` returns `True` only when **both** conditions are `True`. If either is `False`, the result is `False`.

### `or` — At Least One Condition Must Be True

```python
is_member = False
has_coupon = True

if is_member or has_coupon:
    print("Discount applied!")
else:
    print("No discount available.")

# Output: Discount applied!
```

`or` returns `True` when **at least one** condition is `True`.

### `not` — Reverses the Boolean Value

```python
is_weekend = False

if not is_weekend:
    print("It's a weekday — time to study!")

# Output: It's a weekday — time to study!
```

`not` flips `True` to `False` and `False` to `True`.

### Combining Logical Operators

You can combine all three operators in a single expression. Use parentheses to make precedence explicit:

```python
temperature = 22
is_sunny = True
is_windy = False

if temperature > 20 and is_sunny and not is_windy:
    print("Perfect day for a picnic!")
else:
    print("Maybe stay inside.")

# Output: Perfect day for a picnic!
```

## Checking Membership with `in`

The `in` operator checks whether a value exists in a collection (like a string or list):

```python
valid_categories = ["sports", "science", "technology", "art"]
category = "science"

if category in valid_categories:
    print(f"'{category}' is a valid category.")
else:
    print(f"'{category}' is not recognized.")

# Output: 'science' is a valid category.
```

## Putting It Together: A Data Science Example

Here's a practical example that classifies a dataset entry based on multiple conditions:

```python
def classify_bmi(weight_kg, height_m):
    """Classify BMI according to standard categories."""
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return bmi, category

bmi, label = classify_bmi(70, 1.75)
print(f"BMI: {bmi:.1f} — {label}")  # Output: BMI: 22.9 — Normal weight
```

## Conclusion

In this lesson, you learned how to use `if`, `elif`, and `else` statements to make your programs respond to different situations. You also learned how to use comparison operators to evaluate conditions and logical operators (`and`, `or`, `not`) to combine them. Conditional logic is a cornerstone of programming — it will appear in every data processing pipeline, classification task, and validation step you write. In the next lesson, you'll learn about **lists**, Python's most versatile data structure for storing collections of values.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: What does the following code print?**
```python
x = 15
if x > 20:
    print("Big")
elif x > 10:
    print("Medium")
else:
    print("Small")
```
1. `Big`
2. `Medium`
3. `Small`
4. Nothing is printed

**Correct Answer:**
2. `Medium`

**Explanation:**
Python evaluates conditions from top to bottom. The first condition `x > 20` is `False` (15 is not greater than 20), so Python moves to `elif x > 10`, which is `True` (15 is greater than 10). `"Medium"` is printed and the `else` block is skipped.

---

#### **Question 2: Which logical operator returns `True` only when BOTH conditions are `True`?**
1. `or`
2. `not`
3. `and`
4. `in`

**Correct Answer:**
3. `and`

**Explanation:**
The `and` operator requires all conditions to be `True`. If any one condition is `False`, the entire `and` expression is `False`. The `or` operator returns `True` if at least one condition is `True`.

---

#### **Question 3: What is the output of the following code?**
```python
items = ["apple", "banana", "cherry"]
if "grape" not in items:
    print("Grape is not in the list.")
else:
    print("Grape is in the list.")
```
1. `Grape is in the list.`
2. `Grape is not in the list.`
3. `True`
4. `False`

**Correct Answer:**
2. `Grape is not in the list.`

**Explanation:**
The `in` operator checks for membership. Since `"grape"` is not in the list `["apple", "banana", "cherry"]`, `"grape" in items` is `False`. Applying `not` flips it to `True`, so the `if` block runs and prints `"Grape is not in the list."`.
