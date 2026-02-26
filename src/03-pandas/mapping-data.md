# Mapping Data

## Overview

In the previous lesson, you learned how to select specific rows and columns and assign new data to a DataFrame. Often, you'll need to go further — transforming every value in a column according to some rule or custom function. In this lesson, you'll learn how to use `.map()`, `.apply()`, and lambda functions to apply transformations across Series and DataFrames efficiently.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Map or apply functions to Series or DataFrame columns.
- Use `.apply()`, `.map()`, and lambda functions for transformations.

## Key terms

**`.map()`:** A Series method that applies a function or a mapping dictionary to each element of the Series, one value at a time.

**`.apply()`:** A flexible method available on both Series and DataFrames that applies a function along an axis (row-by-row or column-by-column).

**Lambda function:** A small, anonymous (unnamed) function written inline using the `lambda` keyword. Often used as a quick transformation inside `.map()` or `.apply()`.

**Vectorized operation:** An operation applied to an entire column at once (e.g., `df["price"] * 1.1`) without looping — faster and more readable than using `.apply()` for simple math.

**`.applymap()` / `.map()` (DataFrame):** In newer versions of Pandas (≥ 2.1), `DataFrame.map()` applies a function element-by-element to every cell in a DataFrame (previously called `.applymap()`).

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Real datasets are rarely clean or in the exact format you need. Column values may need to be rescaled, categories renamed, strings standardized, or new features derived from existing ones. Pandas provides several tools for these transformations. Understanding when to use each one — and when a simpler vectorized operation is better — is a key data wrangling skill.

## Vectorized Operations (The First Choice)

Before reaching for `.map()` or `.apply()`, check whether a simple arithmetic or string operation already does what you need. Vectorized operations are applied to the entire column at once and are much faster:

```python
import pandas as pd

df = pd.DataFrame({
    "product": ["Laptop", "Phone", "Tablet", "Monitor"],
    "price":   [999, 699, 499, 349],
    "rating":  [4.5, 4.2, 3.8, 4.0]
})

# Add 10% tax to every price
df["price_with_tax"] = df["price"] * 1.10

# Convert rating to percentage
df["rating_pct"] = df["rating"] / 5 * 100

print(df)
```

Output:
```
   product  price  rating  price_with_tax  rating_pct
0   Laptop    999     4.5         1098.90        90.0
1    Phone    699     4.2          768.90        84.0
2   Tablet    499     3.8          548.90        76.0
3  Monitor    349     4.0          383.90        80.0
```

Use vectorized operations whenever the transformation is a straightforward arithmetic expression or a built-in string/date method.

## Using `.map()` on a Series

`.map()` transforms each value in a Series individually. It accepts a **function**, a **lambda**, or a **dictionary** (for lookup/replacement).

### With a Function

```python
def categorize_price(price):
    if price >= 800:
        return "Premium"
    elif price >= 500:
        return "Mid-range"
    else:
        return "Budget"

df["price_tier"] = df["price"].map(categorize_price)
print(df[["product", "price", "price_tier"]])
```

Output:
```
   product  price price_tier
0   Laptop    999    Premium
1    Phone    699  Mid-range
2   Tablet    499     Budget
3  Monitor    349     Budget
```

### With a Lambda Function

A **lambda** is a compact, one-line function defined inline. Instead of writing a full `def`, you write `lambda argument: expression`:

```python
# Capitalize product names
df["product_upper"] = df["product"].map(lambda x: x.upper())
print(df["product_upper"])
```

Output:
```
0    LAPTOP
1     PHONE
2    TABLET
3   MONITOR
Name: product_upper, dtype: object
```

Lambda functions are ideal for simple, one-off transformations that don't need a full named function.

### With a Dictionary (Value Replacement)

Pass a dictionary to replace specific values — any value not in the dictionary becomes `NaN`:

```python
tier_code = {"Premium": "P", "Mid-range": "M", "Budget": "B"}
df["tier_code"] = df["price_tier"].map(tier_code)
print(df[["product", "price_tier", "tier_code"]])
```

Output:
```
   product price_tier tier_code
0   Laptop    Premium         P
1    Phone  Mid-range         M
2   Tablet     Budget         B
3  Monitor     Budget         B
```

## Using `.apply()` on a Series

`.apply()` works similarly to `.map()` on a Series, but it is more flexible and is the standard choice when using `.apply()` on a DataFrame:

```python
# Apply a function to round prices to the nearest hundred
df["rounded_price"] = df["price"].apply(lambda x: round(x, -2))
print(df[["product", "price", "rounded_price"]])
```

Output:
```
   product  price  rounded_price
0   Laptop    999           1000
1    Phone    699            700
2   Tablet    499            500
3  Monitor    349            300
```

## Using `.apply()` on a DataFrame (Row-Wise)

When you call `.apply()` on a DataFrame with `axis=1`, the function receives an entire **row** as a Series. This is useful when a transformation depends on multiple columns:

```python
df = pd.DataFrame({
    "name":   ["Alex", "Jordan", "Morgan", "Sam"],
    "score":  [88, 92, 75, 95],
    "bonus":  [5, 3, 8, 2]
})

def final_score(row):
    """Add bonus to score, cap at 100."""
    return min(row["score"] + row["bonus"], 100)

df["final_score"] = df.apply(final_score, axis=1)
print(df)
```

Output:
```
     name  score  bonus  final_score
0    Alex     88      5           93
1  Jordan     92      3           95
2  Morgan     75      8           83
3     Sam     95      2           97
```

Use `axis=1` to apply row-by-row. The default `axis=0` applies column-by-column (less common).

### Row-Wise Lambda with `.apply()`

```python
df["label"] = df.apply(
    lambda row: f"{row['name']}: {row['final_score']}", axis=1
)
print(df["label"])
```

Output:
```
0      Alex: 93
1    Jordan: 95
2    Morgan: 83
3       Sam: 97
Name: label, dtype: object
```

## Choosing the Right Tool

| Situation | Recommended tool |
|-----------|-----------------|
| Simple math or string on one column | Vectorized operation (`df["col"] * 2`) |
| Transform each value in one column | `.map()` with a function, lambda, or dict |
| Transform each value in one column (complex logic) | `.apply()` on a Series |
| Transform based on multiple columns | `.apply(func, axis=1)` on a DataFrame |
| Replace specific category labels | `.map()` with a dictionary |

> **Performance tip:** Vectorized operations are fastest. `.map()` is faster than `.apply()` on a Series. Row-wise `.apply(axis=1)` is the slowest — avoid it on very large datasets by restructuring the logic as vectorized operations where possible.

## Putting It Together: A Data Science Example

Here's a realistic pipeline that cleans and engineers features from a raw dataset:

```python
import pandas as pd

df = pd.DataFrame({
    "student":  ["  alice", "BOB  ", " charlie", "diana"],
    "raw_score": [82, 67, 91, 55],
    "attempts": [1, 3, 1, 4]
})

# 1. Clean student names
df["student"] = df["student"].map(lambda x: x.strip().title())

# 2. Apply a penalty for multiple attempts (-5 per extra attempt)
df["adjusted_score"] = df.apply(
    lambda row: max(row["raw_score"] - (row["attempts"] - 1) * 5, 0),
    axis=1
)

# 3. Assign pass/fail
df["result"] = df["adjusted_score"].map(lambda s: "Pass" if s >= 70 else "Fail")

print(df)
```

Output:
```
   student  raw_score  attempts  adjusted_score result
0    Alice         82         1              82   Pass
1      Bob         67         3              57   Fail
2  Charlie         91         1              91   Pass
3    Diana         55         4              40   Fail
```

## Conclusion

In this lesson, you learned how to transform data in Pandas using vectorized operations, `.map()`, and `.apply()`. You used dictionary mappings for value replacement, lambda functions for compact inline transformations, and row-wise `.apply()` for logic that spans multiple columns. These tools are at the heart of **feature engineering** — the process of transforming raw data into meaningful inputs for analysis and machine learning. In the next lesson, you'll learn how to **group** data by categories and **sort** it to uncover patterns across segments of your dataset.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="A lambda function is written as `lambda arguments: expression`. It has no name and is typically used once, inline — for example, inside `.map()` or `.apply()`. It is equivalent to a `def` function but more concise for simple transformations.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is a lambda function in Python?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>A function imported from the `lambda` library.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>A named function that can be reused multiple times throughout a program.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>A small, anonymous function defined inline using the `lambda` keyword.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>A function that only works with Pandas DataFrames.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When you pass a dictionary to `.map()`, it acts as a lookup table. Each value in the Series is looked up in the dictionary&#039;s keys; if found, it&#039;s replaced with the corresponding value. Any value not present as a key becomes `NaN`. This is useful for recoding categorical variables (e.g., replacing `&quot;M&quot;` with `&quot;Male&quot;`).">
  <div class="quiz-question">
    <strong>Question 2:</strong> What happens when you pass a dictionary to `.map()` on a Series?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Each key in the dictionary becomes a new column.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Values in the Series that match a dictionary key are replaced with the corresponding value; unmatched values become `NaN`.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The dictionary is appended as a new row in the DataFrame.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The Series is sorted according to the dictionary's key order.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Vectorized operations (like `df[&quot;col&quot;] * 2`) are the fastest and should be preferred for simple math. `.apply(axis=1)` is used when a calculation requires values from multiple columns in the same row, or when the logic is complex enough to need a full function. It is slower than vectorized operations, so it should not be the first choice when a simpler approach works.">
  <div class="quiz-question">
    <strong>Question 3:</strong> When should you use `.apply(func, axis=1)` instead of a vectorized operation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Whenever you need to process a single column.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>When the transformation involves multiple columns or complex logic that can't be expressed as a simple arithmetic expression.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>When you want the fastest possible performance on large datasets.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Only when the function returns a string value.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

