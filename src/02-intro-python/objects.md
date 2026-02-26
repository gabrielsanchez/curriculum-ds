# Objects

## Overview

In the previous lesson, you learned how to organize data using dictionaries by pairing keys with values. As programs become more complex, you may want to bundle both **data** and **behavior** together into a single, self-contained unit. In this lesson, you'll learn the basics of **object-oriented programming (OOP)** in Python — how to create your own custom data types called **classes** and bring them to life as **objects**.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Understand the basics of object-oriented programming in Python.
- Create simple classes and instantiate objects.

## Key terms

**Object-Oriented Programming (OOP):** A programming paradigm that organizes code around objects — entities that combine data (attributes) and behavior (methods).

**Class:** A blueprint or template that defines the structure and behavior shared by all objects of that type.

**Object (instance):** A specific realization of a class, created using the class as a template. Each object has its own copy of the class's attributes.

**Attribute:** A variable that belongs to an object, used to store its data (e.g., a student's name or GPA).

**Method:** A function defined inside a class that describes the behavior of objects of that class.

**`__init__()`:** A special method (called a constructor) that Python automatically calls when a new object is created to initialize its attributes.

**`self`:** A reference to the current object instance, used inside methods to access the object's attributes and other methods.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Everything in Python is an object. When you write `"hello".upper()`, you're calling a **method** on a string **object**. When you call `scores.append(90)`, you're calling a method on a list object. Python's built-in types — strings, lists, dictionaries — are all classes under the hood.

Object-oriented programming lets you create your **own** types that work the same way. Instead of managing a student's name, GPA, and courses as separate variables, you can create a `Student` class that bundles all of that together in a clean, reusable package.

## Defining a Class

Use the `class` keyword to define a class:

```python
class Student:
    pass   # 'pass' is a placeholder — the class has no content yet
```

This creates a class called `Student`. The name follows the same rules as variable names, but by convention class names use **PascalCase** (each word capitalized, no underscores).

## The `__init__()` Method

The `__init__()` method is the **constructor** — Python calls it automatically when you create a new object. Use it to set the initial values of the object's attributes:

```python
class Student:
    def __init__(self, name, major, gpa):
        self.name = name
        self.major = major
        self.gpa = gpa
```

- The first parameter is always `self`, which refers to the object being created.
- `self.name = name` stores the argument `name` as an **attribute** on the object.

## Creating Objects (Instantiation)

Create a new object by calling the class like a function:

```python
student1 = Student("Alex", "Data Science", 3.8)
student2 = Student("Jordan", "Computer Science", 3.5)
```

Each object is an independent instance with its own attribute values.

## Accessing Attributes

Use dot notation to access an object's attributes:

```python
print(student1.name)    # Output: Alex
print(student1.major)   # Output: Data Science
print(student1.gpa)     # Output: 3.8

print(student2.name)    # Output: Jordan
print(student2.gpa)     # Output: 3.5
```

## Defining Methods

Methods are functions defined inside a class. They describe what objects of that class can **do**:

```python
class Student:
    def __init__(self, name, major, gpa):
        self.name = name
        self.major = major
        self.gpa = gpa

    def introduce(self):
        print(f"Hi, I'm {self.name} studying {self.major}.")

    def is_honors(self):
        return self.gpa >= 3.5
```

Call a method using dot notation on the object:

```python
student1 = Student("Alex", "Data Science", 3.8)

student1.introduce()
# Output: Hi, I'm Alex studying Data Science.

print(student1.is_honors())
# Output: True
```

## Updating Attributes

You can change an object's attributes after it has been created:

```python
student1.gpa = 3.9
print(student1.gpa)    # Output: 3.9
```

You can also add methods that update attributes:

```python
class Student:
    def __init__(self, name, major, gpa):
        self.name = name
        self.major = major
        self.gpa = gpa
        self.courses = []              # Initialize an empty list

    def enroll(self, course):
        self.courses.append(course)
        print(f"{self.name} enrolled in {course}.")

    def list_courses(self):
        if self.courses:
            print(f"{self.name}'s courses: {', '.join(self.courses)}")
        else:
            print(f"{self.name} has no courses yet.")
```

```python
student1 = Student("Alex", "Data Science", 3.8)
student1.enroll("Python Programming")
student1.enroll("Statistics")
student1.list_courses()
# Output:
# Alex enrolled in Python Programming.
# Alex enrolled in Statistics.
# Alex's courses: Python Programming, Statistics
```

## The `__str__()` Method

Define `__str__()` to control how an object is displayed when printed:

```python
class Student:
    def __init__(self, name, major, gpa):
        self.name = name
        self.major = major
        self.gpa = gpa

    def __str__(self):
        return f"Student({self.name}, {self.major}, GPA: {self.gpa})"
```

```python
student1 = Student("Alex", "Data Science", 3.8)
print(student1)
# Output: Student(Alex, Data Science, GPA: 3.8)
```

Without `__str__`, printing an object shows something like `<__main__.Student object at 0x7f...>`.

## Putting It Together: A Data Science Example

Here's a class that represents a simple dataset column and includes methods for basic analysis:

```python
class DataColumn:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def mean(self):
        return sum(self.data) / len(self.data)

    def max_value(self):
        return max(self.data)

    def min_value(self):
        return min(self.data)

    def count_above(self, threshold):
        return len([x for x in self.data if x > threshold])

    def summary(self):
        print(f"Column: {self.name}")
        print(f"  Count: {len(self.data)}")
        print(f"  Mean:  {self.mean():.2f}")
        print(f"  Min:   {self.min_value()}")
        print(f"  Max:   {self.max_value()}")


scores = DataColumn("Exam Scores", [88, 95, 72, 61, 84, 90, 78, 55, 93, 67])
scores.summary()
print(f"  Above 80: {scores.count_above(80)}")
```

Output:
```
Column: Exam Scores
  Count: 10
  Mean:  78.30
  Min:   55
  Max:   95
  Above 80: 4
```

## Conclusion

In this lesson, you learned the fundamentals of object-oriented programming in Python: how to define a class with the `class` keyword, initialize objects with `__init__()`, store data in attributes, and define behavior through methods. OOP is a powerful way to organize complex programs, and you'll encounter it constantly as you work with Python's data science libraries — `pandas` DataFrames, `scikit-learn` models, and more are all built on classes and objects. In the next lesson, you'll put everything you've learned together in the module assessment.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="`__init__()` is the constructor method. Python calls it automatically every time you create a new instance of the class (e.g., `Student(&quot;Alex&quot;, &quot;Data Science&quot;, 3.8)`). It uses `self` to assign initial values to the object&#039;s attributes.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the purpose of the `__init__()` method in a Python class?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>It imports the class from an external module.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>It is called automatically when a new object is created and sets up the object's initial attributes.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>It defines what is printed when the object is displayed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>It deletes the object when it is no longer needed.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`self` is a reference to the object that the method is being called on. When you write `student1.introduce()`, Python automatically passes `student1` as the `self` argument, so `self.name` inside the method refers to `student1.name`.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does `self` refer to inside a class method?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The class itself.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The specific object instance that the method was called on.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The first argument passed to the method by the user.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The parent class.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="A **class** defines the structure and behavior (like an architectural blueprint). An **object** is a specific thing built from that blueprint — it has its own attribute values but follows the structure defined by the class. You can create many objects from a single class, each with different data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What is the relationship between a class and an object?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>A class is a specific realization of an object.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>An object is the same as a class — the terms are interchangeable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>A class is a blueprint, and an object is a specific instance created from that blueprint.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>An object defines the structure; a class uses that structure to store data.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

