# Lists

## Overview

In the previous lesson, you learned how to control program flow using conditional logic and boolean expressions. So far, you've stored only one value per variable. In this lesson, you'll learn about **lists** — Python's built-in tool for storing and working with ordered collections of values. Lists are fundamental to data science, where you frequently work with sequences of numbers, names, or categories.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Create and modify lists (append, remove, slicing).
- Use list methods (sort, reverse) and list comprehensions.

## Key terms

**List:** An ordered, mutable collection of items enclosed in square brackets, where items can be of any data type.

**Index:** The numeric position of an item in a list, starting at `0` for the first item.

**Mutable:** Able to be changed after creation. Lists are mutable — you can add, remove, or update items.

**`append()`:** A list method that adds a single item to the end of a list.

**`remove()`:** A list method that removes the first occurrence of a specified value from a list.

**List comprehension:** A concise, one-line syntax for creating a new list by applying an expression to each item in an existing list.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/02-intro-python/04_lists_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

In data science, you rarely work with a single value. You work with datasets — collections of many values. Python's **list** is the most versatile built-in structure for holding an ordered sequence of items. Whether you're storing exam scores, country names, or stock prices, lists give you a flexible way to group related data and process it efficiently.

## Creating Lists

Create a list by placing items inside square brackets `[]`, separated by commas:

```python
# A list of numbers
scores = [85, 92, 78, 90, 88]

# A list of strings
fruits = ["apple", "banana", "cherry"]

# A mixed list (allowed but uncommon in data science)
mixed = [42, "hello", True, 3.14]

# An empty list
empty = []
```

Check the number of items with `len()`:

```python
print(len(scores))   # Output: 5
print(len(fruits))   # Output: 3
```

## Accessing Items

### Indexing

Access individual items using their index inside square brackets. Remember: Python indices start at **0**.

```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])   # Output: apple
print(fruits[1])   # Output: banana
print(fruits[-1])  # Output: cherry  (last item)
print(fruits[-2])  # Output: banana  (second to last)
```

### Slicing

Extract a sub-list using `[start:end]` — the same syntax as string slicing:

```python
scores = [85, 92, 78, 90, 88, 76, 95]
print(scores[1:4])   # Output: [92, 78, 90]
print(scores[:3])    # Output: [85, 92, 78]
print(scores[4:])    # Output: [88, 76, 95]
print(scores[::2])   # Output: [85, 78, 88, 95]  (every other item)
```

## Modifying Lists

### Updating an Item

Assign a new value to a specific index:

```python
fruits = ["apple", "banana", "cherry"]
fruits[1] = "mango"
print(fruits)  # Output: ['apple', 'mango', 'cherry']
```

### `append()` — Adding to the End

Add a single item to the end of a list:

```python
scores = [85, 92, 78]
scores.append(90)
print(scores)  # Output: [85, 92, 78, 90]
```

### `insert()` — Adding at a Specific Position

```python
scores = [85, 92, 78, 90]
scores.insert(1, 100)   # Insert 100 at index 1
print(scores)  # Output: [85, 100, 92, 78, 90]
```

### `remove()` — Deleting by Value

Remove the first occurrence of a specific value:

```python
fruits = ["apple", "banana", "cherry", "banana"]
fruits.remove("banana")
print(fruits)  # Output: ['apple', 'cherry', 'banana']
```

### `pop()` — Removing by Index

`pop()` removes and **returns** the item at a given index (default: the last item):

```python
scores = [85, 92, 78, 90]
removed = scores.pop()     # Remove last item
print(removed)             # Output: 90
print(scores)              # Output: [85, 92, 78]

scores.pop(0)              # Remove first item
print(scores)              # Output: [92, 78]
```

## Useful List Methods

### `sort()` — Sorting In Place

`sort()` rearranges the list's items in ascending order by default:

```python
scores = [78, 92, 85, 90, 88]
scores.sort()
print(scores)  # Output: [78, 85, 88, 90, 92]

# Sort in descending order
scores.sort(reverse=True)
print(scores)  # Output: [92, 90, 88, 85, 78]
```

### `reverse()` — Reversing In Place

```python
fruits = ["apple", "banana", "cherry"]
fruits.reverse()
print(fruits)  # Output: ['cherry', 'banana', 'apple']
```

### Other Helpful Methods

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

print(sum(numbers))           # Output: 36    (total)
print(min(numbers))           # Output: 1     (smallest)
print(max(numbers))           # Output: 9     (largest)
print(numbers.count(1))       # Output: 2     (occurrences of 1)
print(numbers.index(5))       # Output: 4     (first index of 5)
```

## Iterating Through a List

Use a `for` loop to process each item in a list:

```python
scores = [85, 92, 78, 90, 88]

for score in scores:
    if score >= 90:
        print(f"{score} — Excellent!")
    else:
        print(f"{score} — Good")
```

Output:
```
85 — Good
92 — Excellent!
78 — Good
90 — Excellent!
88 — Good
```

## List Comprehensions

A **list comprehension** is a compact, readable way to build a new list by applying an expression to each item in an existing list. The general syntax is:

```python
new_list = [expression for item in existing_list]
```

### Basic Example

Without a list comprehension:
```python
scores = [85, 92, 78, 90, 88]
curved = []
for score in scores:
    curved.append(score + 5)
print(curved)  # Output: [90, 97, 83, 95, 93]
```

With a list comprehension — same result, one line:
```python
scores = [85, 92, 78, 90, 88]
curved = [score + 5 for score in scores]
print(curved)  # Output: [90, 97, 83, 95, 93]
```

### With a Condition (Filtering)

Add an `if` clause to include only items that meet a condition:

```python
scores = [85, 92, 78, 90, 88, 55, 63]
passing = [score for score in scores if score >= 70]
print(passing)  # Output: [85, 92, 78, 90, 88]
```

### Transforming Text Data

```python
names = ["  alice ", "BOB", "  charlie"]
clean_names = [name.strip().title() for name in names]
print(clean_names)  # Output: ['Alice', 'Bob', 'Charlie']
```

## Putting It Together: A Data Science Example

Here's a realistic example using lists to summarize a column of test scores:

```python
scores = [88, 95, 72, 61, 84, 90, 78, 55, 93, 67]

# Summary statistics
print(f"Count:   {len(scores)}")
print(f"Total:   {sum(scores)}")
print(f"Average: {sum(scores) / len(scores):.1f}")
print(f"Min:     {min(scores)}")
print(f"Max:     {max(scores)}")

# Filter passing scores (>= 70)
passing = [s for s in scores if s >= 70]
print(f"Passing: {len(passing)} out of {len(scores)}")
```

Output:
```
Count:   10
Total:   783
Average: 78.3
Min:     55
Max:     95
Passing: 7 out of 10
```

## Conclusion

In this lesson, you learned how to create and modify lists using methods like `append()`, `remove()`, `sort()`, and `reverse()`. You also learned how to access items using indexing and slicing, iterate through lists with loops, and write concise transformations using list comprehensions. Lists are one of the most commonly used data structures in Python and will be a key tool throughout your data science journey. In the next lesson, you'll explore **dictionaries**, which let you store data as meaningful key-value pairs.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/02-intro-python/04_lists_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="Python uses **zero-based indexing**, meaning the first item in a list is always at index `0`. This is consistent across strings, lists, and other ordered sequences in Python.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the index of the first item in a Python list?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`1`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`-1`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`0`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>It depends on the list.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="`append()` adds one item to the **end** of a list and modifies the list in place. If you want to add an item at a specific position, use `insert(index, value)` instead.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does the `append()` method do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Removes the last item from a list.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Inserts an item at a specified index.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Adds a single item to the end of a list.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Sorts the list in ascending order.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The comprehension filters numbers where `n % 2 == 0` (even numbers: 2, 4, 6), then multiplies each by 2. The result is `[2*2, 4*2, 6*2]` = `[4, 8, 12]`.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What does the following list comprehension produce?
  </div>
  <div class="quiz-subquestion">
    <pre><code>numbers = [1, 2, 3, 4, 5, 6]
result = [n * 2 for n in numbers if n % 2 == 0]
print(result)</code></pre>
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>`[2, 4, 6, 8, 10, 12]`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>`[4, 8, 12]`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>`[2, 6, 10]`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>`[1, 3, 5]`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

