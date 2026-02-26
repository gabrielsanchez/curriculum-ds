# Functions

## Overview

In the previous lesson, you learned how to store values in variables and work with Python's basic data types. As your programs grow, you'll often find yourself repeating the same steps over and over. In this lesson, you'll learn how to use **functions** — a way to group reusable blocks of code under a single name so you can call them whenever needed.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Define and call functions to structure code.
- Use parameters and return statements effectively.

## Key terms

**Function:** A named, reusable block of code that performs a specific task when called.

**`def` keyword:** The Python keyword used to define (create) a function.

**Parameter:** A variable listed inside a function's parentheses that acts as a placeholder for values passed in when the function is called.

**Argument:** The actual value supplied to a function when it is called.

**Return statement:** The `return` keyword inside a function that sends a value back to the code that called the function.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Imagine you work at a coffee shop and every morning you follow the exact same steps to brew coffee. Instead of reading the instructions from scratch each time, you memorize the routine and execute it whenever needed. Functions work the same way in programming — you write the instructions once and then **call** the function by name any time you want to run them.

Functions are one of the most important concepts in Python. They help you:
- **Avoid repetition** by writing code once and reusing it many times.
- **Organize** your program into logical, readable pieces.
- **Test** small units of code independently.

## Defining a Function

### Basic Syntax

Use the `def` keyword to create a function, followed by a name, parentheses, and a colon. The indented block below is the function's body — the code that runs when you call it.

```python
def greet():
    print("Hello, welcome to Data Science!")
```

The code above defines a function named `greet`. Notice that nothing is printed yet — the function is only defined here, not executed.

### Calling a Function

To run a function, write its name followed by parentheses:

```python
greet()  # Output: Hello, welcome to Data Science!
```

You can call the function as many times as you like:

```python
greet()
greet()
greet()
```

## Parameters and Arguments

### Adding Parameters

Parameters let you pass information into a function, making it flexible and reusable:

```python
def greet(name):
    print("Hello, " + name + "! Welcome to Data Science!")
```

Here, `name` is a **parameter** — a placeholder that will be filled in when the function is called.

### Passing Arguments

When you call the function, you provide an **argument** — the real value that replaces the parameter:

```python
greet("Alex")    # Output: Hello, Alex! Welcome to Data Science!
greet("Jordan")  # Output: Hello, Jordan! Welcome to Data Science!
```

### Multiple Parameters

Functions can have more than one parameter, separated by commas:

```python
def describe_student(name, age, city):
    print(name + " is " + str(age) + " years old and lives in " + city + ".")

describe_student("Alex", 25, "Chicago")
# Output: Alex is 25 years old and lives in Chicago.
```

### Default Parameter Values

You can provide a default value for a parameter. If no argument is passed, the default is used:

```python
def greet(name, language="English"):
    if language == "Spanish":
        print("Hola, " + name + "!")
    else:
        print("Hello, " + name + "!")

greet("Alex")             # Output: Hello, Alex!
greet("Maria", "Spanish") # Output: Hola, Maria!
```

## Return Statements

### Returning a Value

So far, functions have only printed output. More often, you want a function to **calculate** something and send the result back to the rest of your program. Use the `return` keyword for this:

```python
def add(a, b):
    return a + b
```

Now you can capture the result and use it:

```python
result = add(3, 5)
print(result)       # Output: 8
print(add(10, 20))  # Output: 30
```

### Why `return` Matters

The difference between `print` and `return` is crucial:

```python
def square_print(x):
    print(x ** 2)    # Displays the result but does not save it

def square_return(x):
    return x ** 2    # Sends the result back to the caller

value = square_return(4)
print(value * 2)   # Output: 32  (we can use the returned value)

square_print(4)    # Output: 16  (but we can't use this value further)
```

### Returning Multiple Values

A function can return more than one value at once using a tuple:

```python
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([3, 1, 4, 1, 5, 9, 2, 6])
print("Min:", low)   # Output: Min: 1
print("Max:", high)  # Output: Max: 9
```

## Putting It Together: A Data Science Example

Here's a practical example that combines everything you've learned:

```python
def calculate_average(scores):
    """Calculate the average of a list of scores."""
    total = sum(scores)
    count = len(scores)
    return total / count

student_scores = [85, 92, 78, 90, 88]
avg = calculate_average(student_scores)
print("Average score:", avg)  # Output: Average score: 86.6
```

The triple-quoted text `"""..."""` directly below the `def` line is called a **docstring** — a short description of what the function does. It's good practice to include one.

## Conclusion

In this lesson, you learned how to define and call functions in Python, pass data in using parameters and arguments, and send results back using return statements. Functions are one of the most powerful tools in programming — they keep your code organized, readable, and reusable. In the next lesson, you'll explore strings in depth, learning how to manipulate and transform text data.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="In Python, the `def` keyword is used to define a function. It is followed by the function name, parentheses (which can contain parameters), and a colon. The indented block below becomes the function&#039;s body.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What keyword is used to define a function in Python?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>`function`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>`define`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>`def`</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>`fn`</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When you write `def greet(name):`, `name` is a **parameter** — a placeholder. When you call `greet(&quot;Alex&quot;)`, `&quot;Alex&quot;` is the **argument** — the real value supplied to that placeholder.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What is the difference between a parameter and an argument?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Parameters are used only with `return` statements; arguments are used with `print`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Parameters are placeholders defined in the function; arguments are the actual values passed when calling the function.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Parameters are the output of a function; arguments are the input.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>There is no difference — the terms are interchangeable.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="In Python, every function returns a value. If no `return` statement is present (or `return` is used without a value), the function automatically returns `None`, which represents the absence of a value. This is why using `print` inside a function instead of `return` can lead to unexpected behavior when you try to use the function&#039;s result elsewhere.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What happens if a function does not have a `return` statement?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Python raises an error.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The function returns `0` by default.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The function returns `None` by default.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The function runs indefinitely.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

