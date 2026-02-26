# Correlation Matrix

## Overview

In the previous lessons, you examined each variable individually — its distribution, spread, and outliers. But in data science, some of the most important questions are about **relationships between variables**: Does square footage predict price? Do older homes tend to have lower scores? Are two features so similar they carry redundant information? In this lesson, you'll learn how to compute a correlation matrix to measure the strength of pairwise relationships between all numeric features in your dataset.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Compute pairwise correlations between features.
- Detect relationships that might influence modeling decisions.

## Key terms

**Correlation:** A statistical measure of the strength and direction of the linear relationship between two numeric variables, ranging from −1 to +1.

**Pearson correlation coefficient (r):** The most common correlation measure. A value of +1 means a perfect positive linear relationship, −1 means a perfect negative linear relationship, and 0 means no linear relationship.

**Correlation matrix:** A table where each cell shows the correlation coefficient between the row variable and the column variable. The diagonal is always 1.0 (a variable is perfectly correlated with itself).

**Positive correlation:** As one variable increases, the other tends to increase as well.

**Negative correlation:** As one variable increases, the other tends to decrease.

**Multicollinearity:** A situation where two or more features are highly correlated with each other. This can cause problems in some models (e.g., linear regression), as the model can't distinguish the individual contribution of each feature.

**Heatmap:** A grid visualization where cells are colored according to their value — ideal for displaying a correlation matrix.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Knowing that house prices range from $100K to $1M is useful. Knowing that price is strongly correlated with square footage, and weakly correlated with age, tells you which features are likely to be valuable predictors in a model. Knowing that square footage and number of bedrooms are very highly correlated with each other tells you they may carry redundant information.

A **correlation matrix** surfaces all of these relationships at once. It is one of the most information-dense tools in EDA — a single heatmap can reveal which features deserve deeper investigation and warn you about potential modeling problems before you've written a single line of model code.

## Setup: Sample Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 500

sqft       = np.random.normal(1600, 400, n).clip(600)
bedrooms   = (sqft / 400 + np.random.normal(0, 0.5, n)).clip(1, 6).round()
age_years  = np.random.normal(18, 10, n).clip(0)
price      = (sqft * 120 + bedrooms * 8000
              - age_years * 1500
              + np.random.normal(0, 20000, n)).clip(80000)
score      = (70 - age_years * 0.3
              + np.random.normal(0, 5, n)).clip(0, 100)
garage     = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])

df = pd.DataFrame({
    "price":     price,
    "sqft":      sqft,
    "bedrooms":  bedrooms,
    "age_years": age_years,
    "score":     score,
    "garage":    garage
})
```

## Computing the Correlation Matrix

### With Pandas `.corr()`

```python
corr_matrix = df.corr()
print(corr_matrix.round(2))
```

Output:
```
            price  sqft  bedrooms  age_years  score  garage
price        1.00  0.87      0.63      -0.32  -0.28    0.12
sqft         0.87  1.00      0.71      -0.04  -0.01    0.09
bedrooms     0.63  0.71      1.00      -0.03  -0.02    0.07
age_years   -0.32 -0.04     -0.03       1.00  -0.57   -0.05
score       -0.28 -0.01     -0.02      -0.57   1.00    0.02
garage       0.12  0.09      0.07      -0.05   0.02    1.00
```

Reading the matrix:
- **`price` vs `sqft`: 0.87** — Strong positive correlation. Larger homes cost more.
- **`price` vs `age_years`: −0.32** — Moderate negative correlation. Older homes are cheaper.
- **`age_years` vs `score`: −0.57** — Moderate negative correlation. Older homes have lower scores.
- **`sqft` vs `bedrooms`: 0.71** — Strong positive correlation — potential multicollinearity.
- **`garage` vs `price`: 0.12** — Weak correlation. Garage size is only a minor price signal.

## Interpreting Correlation Strength

Use this rough guide to interpret the magnitude of correlation coefficients:

| |r| range | Interpretation |
|------------|----------------|
| 0.00 – 0.19 | Negligible / no relationship |
| 0.20 – 0.39 | Weak relationship |
| 0.40 – 0.59 | Moderate relationship |
| 0.60 – 0.79 | Strong relationship |
| 0.80 – 1.00 | Very strong relationship |

These thresholds are context-dependent. A correlation of 0.3 between two noisy survey measures might be meaningful; a correlation of 0.3 in a controlled physics experiment would be considered weak.

## Visualizing with a Heatmap

A table of numbers is hard to scan. A **heatmap** makes the pattern immediately visible by encoding correlation values as colors:

```python
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,         # Show the numeric values in each cell
    fmt=".2f",          # Format to 2 decimal places
    cmap="coolwarm",    # Blue = negative, red = positive
    vmin=-1, vmax=1,    # Fix the color scale
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix — Housing Features")
plt.tight_layout()
plt.show()
```

In a `coolwarm` heatmap:
- **Deep red** = strong positive correlation
- **Deep blue** = strong negative correlation
- **White / pale** = near-zero correlation

The diagonal is always deep red (1.0) because every variable is perfectly correlated with itself.

### Masking the Upper Triangle

Since the matrix is symmetric, you can mask the upper half to reduce visual clutter:

```python
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix (Lower Triangle)")
plt.tight_layout()
plt.show()
```

## Identifying the Strongest Correlations

For datasets with many columns, programmatically finding the highest correlations is more practical than scanning a large heatmap:

```python
# Unstack into a Series, remove self-correlations, sort by absolute value
top_corr = (
    corr_matrix
    .unstack()
    .reset_index()
    .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation"})
    .query("var1 != var2")                          # Remove diagonal
    .assign(abs_corr=lambda x: x["correlation"].abs())
    .sort_values("abs_corr", ascending=False)
    .drop_duplicates(subset=["abs_corr"])           # Remove duplicates (A-B and B-A)
    .head(8)
)
print(top_corr[["var1", "var2", "correlation"]])
```

## Correlation and Target Variable

One of the most important uses of a correlation matrix in supervised machine learning is checking which features are most correlated with the **target variable** (e.g., price):

```python
target_corr = corr_matrix["price"].drop("price").sort_values(ascending=False)
print(target_corr)
```

Output:
```
sqft         0.87
bedrooms     0.63
garage       0.12
score       -0.28
age_years   -0.32
Name: price, dtype: float64
```

`sqft` and `bedrooms` are the strongest predictors of price. `garage` has a negligible relationship. `age_years` and `score` have moderate negative relationships — older, lower-scoring homes are cheaper. This output directly informs feature selection for modeling.

## The Scatter Plot Matrix (Pairs Plot)

A **scatter plot matrix** (or pairs plot) shows the relationship between every pair of variables visually. It combines the correlation matrix concept with actual scatter plots:

```python
sns.pairplot(df, corner=True, plot_kws={"alpha": 0.3, "s": 10})
plt.suptitle("Scatter Plot Matrix — Housing Features", y=1.01)
plt.show()
```

The `corner=True` parameter shows only the lower triangle. Each off-diagonal cell is a scatter plot between two variables; the diagonal shows a histogram of each variable. This is a powerful (if dense) overview of the entire dataset at once.

## An Important Caveat: Correlation ≠ Causation

Correlation measures **linear association**, not causation. A strong correlation between two variables does not mean one causes the other. Classic examples of spurious correlations include:

- Ice cream sales and drowning rates are positively correlated — both increase in summer (a **confounding variable**).
- Nicolas Cage films released per year correlates strongly with swimming pool drownings.

Always ask: is there a plausible causal mechanism, or is this a coincidence or a shared underlying cause?

Also note: correlation only captures **linear** relationships. Two variables can have a strong nonlinear relationship and show a correlation near zero. Scatter plots are essential for catching these cases.

## Conclusion

In this lesson, you learned how to compute a correlation matrix using `df.corr()`, visualize it as a heatmap with Seaborn, identify the strongest feature relationships, and extract correlations with the target variable to guide feature selection. You also saw how a scatter plot matrix gives a comprehensive visual overview of all pairwise relationships. In the next lesson, you'll bring together the quantitative and visual outlier detection skills you've been building, learning when and how to handle the anomalous values you've been flagging throughout this module.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="The sign of a correlation coefficient indicates direction: positive means both variables move in the same direction, negative means they move in opposite directions. The magnitude (absolute value) indicates strength. −0.85 has a magnitude of 0.85, which is a very strong relationship. As one variable goes up, the other strongly tends to go down.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A correlation coefficient of −0.85 between two variables indicates:
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>A weak negative relationship — the variables are mostly unrelated.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>No relationship — negative correlations are not meaningful.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>A very strong negative relationship — as one variable increases, the other tends to decrease strongly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>A very strong positive relationship — the negative sign is a typo in the output.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The Pearson correlation coefficient measures how consistently two variables move together. A variable always moves perfectly in sync with itself — for every data point, `x == x` — so the correlation is always exactly 1.0. This is not a convention or rounding; it is mathematically certain.">
  <div class="quiz-question">
    <strong>Question 2:</strong> In a correlation matrix heatmap, the diagonal is always 1.0. Why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It's a display convention — the diagonal is set to 1.0 to make the chart look symmetric.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The diagonal represents the correlation of each variable with itself, which is always a perfect 1.0 by definition.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The diagonal represents the average correlation of each variable with all other variables.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Pandas rounds all correlation values on the diagonal up to 1.0.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="When two features are very highly correlated (multicollinearity), a linear regression model cannot reliably determine how much of the outcome is explained by each one independently. Small changes in the data can cause large swings in the estimated coefficients, making them unstable and difficult to interpret. A common solution is to remove one of the correlated features or combine them using dimensionality reduction.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Two features in your dataset have a correlation of 0.95 with each other. What issue might this cause in a linear regression model?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The model will automatically remove one of the features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The model will be unable to train because the correlation matrix cannot be computed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Multicollinearity — the model will struggle to separate the individual contributions of each feature, leading to unstable and unreliable coefficient estimates.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>No issue — high correlation between features improves model accuracy.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

