# Histograms

## Overview

In the previous lesson, you learned how to compute summary statistics — mean, median, standard deviation, and IQR — to understand the center and spread of your data. Numbers like skewness hint at the shape of a distribution, but they don't show it. In this lesson, you'll learn how to generate **histograms**, the most direct visual tool for seeing how values in a column are distributed across their range.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Generate histograms to assess frequency distributions.
- Identify skewness, normal distributions, and multimodal distributions.

## Key terms

**Histogram:** A chart that groups continuous values into equal-width bins and displays the count (or frequency) of observations falling into each bin as a bar.

**Bin:** A contiguous interval in the value range of a variable. For example, prices from $100,000 to $150,000 might form one bin.

**Frequency:** The number of observations that fall within a given bin.

**Normal distribution (bell curve):** A symmetric, bell-shaped distribution where values are concentrated around the mean and taper off equally in both directions.

**Right-skewed distribution:** A distribution with a long tail to the right — most values are low, but a few very high values stretch the distribution rightward. Also called positively skewed.

**Left-skewed distribution:** A distribution with a long tail to the left — most values are high, but a few very low values stretch the distribution leftward. Also called negatively skewed.

**Bimodal distribution:** A distribution with two distinct peaks, suggesting the data may come from two different underlying groups or processes.

**KDE (Kernel Density Estimate):** A smooth curve that estimates the underlying probability distribution of a variable, used as an alternative or complement to a histogram.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/03_histograms_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

A histogram answers the question: *"Where are my values concentrated?"* It reveals the shape of a distribution — whether most values cluster tightly around the mean, whether the data tails off to one side, or whether there are two distinct clusters suggesting two underlying groups in the data.

Histograms are a core EDA tool because the shape of a distribution has direct consequences for analysis and modeling. Many statistical methods assume a roughly normal distribution. If your data is heavily skewed, a log transformation may be needed. If it is bimodal, you may need to investigate whether two different subpopulations have been combined.

## Setup: Sample Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.DataFrame({
    "price":      np.concatenate([
                      np.random.normal(250000, 40000, 800),
                      np.random.normal(650000, 80000, 50)   # High-end homes
                  ]),
    "sqft":       np.random.lognormal(7.3, 0.4, 850),
    "age_years":  np.random.normal(20, 8, 850).clip(0),
    "score":      np.random.normal(75, 10, 850).clip(0, 100)
})
```

## Creating a Basic Histogram

### With Matplotlib

```python
plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=30, color="steelblue", edgecolor="white")
plt.title("Distribution of House Prices")
plt.xlabel("Price ($)")
plt.ylabel("Number of Homes")
plt.tight_layout()
plt.show()
```

The `bins` parameter controls how many intervals the range is divided into. The bars touch each other (unlike a bar chart) because the bins are continuous ranges.

### With Pandas

Pandas provides a convenient shortcut that creates a histogram for every numeric column at once:

```python
df.hist(figsize=(12, 8), bins=30, edgecolor="white", color="steelblue")
plt.suptitle("Distributions of All Numeric Columns", y=1.02)
plt.tight_layout()
plt.show()
```

This is one of the quickest ways to get a visual overview of your entire dataset.

## Choosing the Number of Bins

The number of bins significantly affects how a histogram looks and what it reveals:

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, bins in zip(axes, [5, 20, 100]):
    ax.hist(df["price"], bins=bins, color="steelblue", edgecolor="white")
    ax.set_title(f"bins={bins}")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")

plt.suptitle("Effect of Bin Count on Histogram Shape")
plt.tight_layout()
plt.show()
```

- **Too few bins** (e.g., 5) over-smooths the data — you lose detail and can miss important features like bimodality.
- **Too many bins** (e.g., 100) under-smooths — you see noise instead of signal, and the bars become very thin.
- **A good starting point** is 20–30 bins for most datasets. Adjust based on what you see.

A common rule of thumb is the **square root rule**: use approximately √n bins, where n is the number of observations.

## Reading Distribution Shapes

### Normal (Symmetric) Distribution

A normal distribution is bell-shaped and symmetric around its mean. The mean and median are approximately equal.

```python
plt.figure(figsize=(8, 5))
plt.hist(df["score"], bins=30, color="mediumseagreen", edgecolor="white")
plt.axvline(df["score"].mean(), color="red", linestyle="--", label=f"Mean: {df['score'].mean():.1f}")
plt.axvline(df["score"].median(), color="orange", linestyle="-", label=f"Median: {df['score'].median():.1f}")
plt.title("Normal Distribution: Exam Scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.legend()
plt.show()
```

Adding vertical lines for mean and median is a useful habit — their positions relative to each other confirm whether the distribution is symmetric.

### Right-Skewed Distribution

A right-skewed distribution has most values on the left with a long tail stretching to the right. The mean is pulled higher than the median by the high-value tail.

```python
plt.figure(figsize=(8, 5))
plt.hist(df["sqft"], bins=30, color="coral", edgecolor="white")
plt.axvline(df["sqft"].mean(), color="red", linestyle="--", label=f"Mean: {df['sqft'].mean():.0f}")
plt.axvline(df["sqft"].median(), color="orange", linestyle="-", label=f"Median: {df['sqft'].median():.0f}")
plt.title("Right-Skewed Distribution: Square Footage")
plt.xlabel("Square Feet")
plt.ylabel("Count")
plt.legend()
plt.show()
```

Square footage and prices are commonly right-skewed — most homes are moderately sized, but a few very large properties pull the distribution rightward.

### Bimodal Distribution

A bimodal distribution has two distinct peaks, suggesting the presence of two subgroups in the data:

```python
plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=40, color="mediumpurple", edgecolor="white")
plt.title("Bimodal Distribution: House Prices")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.show()
```

The two peaks in price — one around $250,000 and a smaller one around $650,000 — reflect the two groups in our synthetic data: standard homes and high-end properties. In a real dataset, this would prompt the question: "What separates these two groups?"

## Adding a KDE Curve

A **Kernel Density Estimate (KDE)** overlays a smooth curve that estimates the underlying distribution. It is a useful complement to the histogram, especially for comparing shapes:

```python
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

data = df["price"].dropna()
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 300)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, color="steelblue", edgecolor="white",
        density=True, alpha=0.6, label="Histogram")
ax.plot(x_range, kde(x_range), color="darkblue", linewidth=2, label="KDE")
ax.set_title("House Prices: Histogram with KDE")
ax.set_xlabel("Price ($)")
ax.set_ylabel("Density")
ax.legend()
plt.show()
```

Note the use of `density=True` to normalize the histogram to a density scale so it is comparable to the KDE curve.

## Log Transformation for Skewed Data

When a right-skewed distribution spans many orders of magnitude, a **log transformation** can make it more symmetric and easier to work with:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["sqft"], bins=30, color="coral", edgecolor="white")
axes[0].set_title("Square Footage — Original")
axes[0].set_xlabel("sqft")

axes[1].hist(np.log1p(df["sqft"]), bins=30, color="steelblue", edgecolor="white")
axes[1].set_title("Square Footage — Log Transformed")
axes[1].set_xlabel("log(sqft + 1)")

plt.tight_layout()
plt.show()
```

`np.log1p()` computes log(x + 1), which safely handles zero values. After transformation, the distribution often becomes much more symmetric, satisfying the assumptions of many statistical models.

## Conclusion

In this lesson, you learned how to generate histograms using Matplotlib and Pandas, how to choose an appropriate bin count, and how to read and interpret the key distribution shapes: normal, right-skewed, left-skewed, and bimodal. You also added mean and median lines to confirm symmetry, overlaid KDE curves for smoother estimates, and applied log transformations to tame heavily skewed data. In the next lesson, you'll complement histograms with **box plots**, which provide a compact five-number summary and make it easy to compare distributions across groups and spot outliers.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/03_histograms_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="3" data-explanation="A right-skewed (positively skewed) distribution has most values concentrated on the left side (lower values) with a long tail extending to the right (higher values). Salary data is a classic example — most employees earn moderate salaries, but a small number of very high earners stretch the distribution to the right, pulling the mean above the median.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A histogram of employee salaries shows a long tail stretching to the right with most values below $80,000 and a few above $300,000. How would you describe this distribution?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Normal (symmetric)</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Left-skewed</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Bimodal</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Right-skewed</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="With too few bins, many different values get lumped into the same bar, blurring fine-grained features. A bimodal distribution might look like a single wide hump. A gap in the data might disappear entirely. Too many bins has the opposite problem — the chart becomes jagged and noisy. Finding the right balance is key, and it often requires trying a few different values.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What happens if you use too few bins in a histogram?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The histogram becomes too noisy and shows random fluctuations rather than the true shape.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The histogram over-smooths the data, potentially hiding important features like bimodality or gaps.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Pandas raises an error because the bin width is too large.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The histogram automatically switches to a bar chart.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="When the mean sits to the right of the median on a histogram, it means the mean has been pulled rightward by a few high values in the tail. This is the hallmark of a right-skewed distribution. The median is resistant to these extremes, so it stays closer to where most of the data actually sits. This pattern is very common in price and income data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You plot a histogram of house prices and notice the mean line sits clearly to the right of the median line. What does this indicate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The data has been normalized correctly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The distribution is symmetric and well-behaved.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The distribution is right-skewed, likely influenced by a small number of very high-priced properties.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The median was calculated incorrectly.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

