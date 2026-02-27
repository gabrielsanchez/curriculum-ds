# Box Plots

## Overview

In the previous lesson, you used histograms to visualize the full shape of a distribution. Histograms are excellent for a single variable, but they become hard to compare when you want to show distributions across multiple groups side by side. In this lesson, you'll learn about **box plots** — a compact visualization that summarizes a distribution in five numbers and makes group comparisons and outlier detection easy.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Use box plots to understand data spread and quartiles.
- Identify outliers visually.

## Key terms

**Box plot (box-and-whisker plot):** A chart that displays a five-number summary (minimum, Q1, median, Q3, maximum) of a distribution, with points beyond the whiskers plotted individually as potential outliers.

**Five-number summary:** The five values that describe a distribution: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum (after excluding outliers).

**Q1 (first quartile):** The 25th percentile — 25% of values fall below this point.

**Q3 (third quartile):** The 75th percentile — 75% of values fall below this point.

**IQR (Interquartile Range):** The height of the box: Q3 − Q1. Represents the spread of the middle 50% of the data.

**Whiskers:** The lines extending from the box to the smallest and largest values within 1.5 × IQR of the box edges.

**Outlier (box plot):** A value that falls more than 1.5 × IQR below Q1 or above Q3. Plotted as individual points beyond the whiskers.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/04_box-plots_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

A histogram shows you the full shape of a distribution but becomes cluttered when you need to compare six or eight groups at once. A box plot sacrifices some detail in exchange for compactness — five numbers instead of thirty bars — making it ideal for **comparing distributions across groups**. In a single glance, you can compare the median, spread, and outlier patterns of housing prices across different neighborhoods or salaries across different departments.

Box plots are also the standard first step for **visual outlier detection**: any point plotted beyond the whiskers is flagged as a potential outlier by the 1.5 × IQR rule.

## Anatomy of a Box Plot

```
        |
   _____|_____         ← Upper whisker: Q3 + 1.5 × IQR (or the largest non-outlier value)
  |           |
  |   Q3      |  ─────── 75th percentile (top of box)
  |           |
  |═══════════|  ─────── Median (Q2, line inside box)
  |           |
  |   Q1      |  ─────── 25th percentile (bottom of box)
  |___________|
        |
   _____|_____         ← Lower whisker: Q1 - 1.5 × IQR (or the smallest non-outlier value)

      ●  ●            ← Outlier points (beyond the whiskers)
```

- **Box height = IQR** — a taller box means more spread in the middle of the data.
- **Median line position** — if the median is near the center of the box, the distribution is symmetric; if it's near the top or bottom, the distribution is skewed.
- **Whisker length** — longer whiskers mean more spread in the tails.
- **Individual points** — plotted outliers beyond the whiskers.

## Setup: Sample Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.DataFrame({
    "price": np.concatenate([
        np.random.normal(220000, 35000, 300),   # South
        np.random.normal(280000, 45000, 300),   # North
        np.random.normal(195000, 28000, 300),   # East
    ]),
    "neighborhood": ["South"] * 300 + ["North"] * 300 + ["East"] * 300,
    "bedrooms": np.random.choice([1, 2, 3, 4, 5], 900,
                                  p=[0.05, 0.25, 0.40, 0.20, 0.10])
})

# Inject a few outliers
df.loc[0, "price"] = 950000
df.loc[1, "price"] = 875000
df.loc[2, "price"] = -15000   # Data error — impossible value
```

## Creating a Basic Box Plot

### With Matplotlib

```python
plt.figure(figsize=(6, 6))
plt.boxplot(df["price"], patch_artist=True,
            boxprops=dict(facecolor="steelblue", color="navy"),
            medianprops=dict(color="white", linewidth=2))
plt.title("Distribution of House Prices")
plt.ylabel("Price ($)")
plt.xticks([1], ["All Homes"])
plt.show()
```

### With Pandas

```python
df[["price"]].boxplot(figsize=(6, 6))
plt.title("Distribution of House Prices")
plt.ylabel("Price ($)")
plt.show()
```

## Grouped Box Plots: Comparing Distributions

The real strength of box plots is comparing multiple groups. Here's how to plot price distributions by neighborhood:

### With Matplotlib

```python
# Separate the data by group
groups = [df[df["neighborhood"] == n]["price"] for n in ["South", "North", "East"]]

plt.figure(figsize=(8, 6))
bp = plt.boxplot(groups,
                 labels=["South", "North", "East"],
                 patch_artist=True,
                 boxprops=dict(facecolor="steelblue", alpha=0.7),
                 medianprops=dict(color="red", linewidth=2))
plt.title("House Price Distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Price ($)")
plt.grid(axis="y", alpha=0.3)
plt.show()
```

### With Seaborn (Recommended for Grouped Box Plots)

Seaborn's `boxplot()` works directly with a DataFrame and a grouping column, making grouped box plots much easier:

```python
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="neighborhood", y="price",
            palette="Set2", order=["South", "North", "East"])
plt.title("House Price Distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Price ($)")
plt.grid(axis="y", alpha=0.3)
plt.show()
```

Reading this chart at a glance:
- **North** has the highest median price and the widest box (most spread).
- **East** has the lowest median and the tightest box (most consistent prices).
- **All neighborhoods** show outlier points above the upper whisker.

## Identifying Outliers Visually

The negative price value and two extreme high values we injected are immediately visible as dots beyond the whiskers. This is one of the most practical uses of box plots in EDA: **spotting values that need investigation**.

To extract the outlier thresholds programmatically:

```python
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
print(f"Lower bound: ${lower_bound:,.0f}")
print(f"Upper bound: ${upper_bound:,.0f}")
print(f"Outlier count: {len(outliers)}")
print(outliers[["price", "neighborhood"]].head())
```

Output:
```
Lower bound: $113,472
Upper bound: $387,472
Outlier count: 5
     price neighborhood
0   950000        South
1   875000        South
2   -15000        South
...
```

The negative price is clearly a **data entry error** — a price cannot be negative. The two very high prices may be legitimate luxury properties or errors — you'd need to investigate further.

## Box Plot vs. Histogram: When to Use Each

| Feature | Histogram | Box Plot |
|---------|-----------|----------|
| Shows full distribution shape | ✓ | ✗ |
| Reveals bimodality | ✓ | ✗ |
| Compact five-number summary | ✗ | ✓ |
| Easy multi-group comparison | ✗ | ✓ |
| Visual outlier detection | Partial | ✓ |
| Works well for large number of groups | ✗ | ✓ |

**In practice:** Use histograms to understand the shape of an individual variable. Use box plots to compare that variable across groups and to flag outliers. Both are tools in your EDA toolkit — they answer different questions.

## Conclusion

In this lesson, you learned how to read a box plot's five-number summary — Q1, median, Q3, whiskers, and outlier points — and how to create them with Matplotlib and Seaborn. You saw how grouped box plots make it fast to compare distributions across many categories, and how any point outside the 1.5 × IQR whiskers is flagged as a potential outlier for further investigation. In the next lesson, you'll shift from individual columns to **relationships between columns**, computing a correlation matrix to measure how strongly numeric features are associated with each other.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/04_box-plots_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="The box in a box plot spans from Q1 (25th percentile) to Q3 (75th percentile). Its height is therefore Q3 − Q1, which is the IQR. A taller box means greater variability in the middle of the data. The whiskers extend further to capture non-outlier values beyond the box.">
  <div class="quiz-question">
    <strong>Question 1:</strong> In a box plot, what does the height of the box represent?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The range from the minimum to the maximum value.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The standard deviation of the distribution.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The interquartile range (IQR) — the spread of the middle 50% of the data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The distance between the mean and the median.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Box plot whiskers extend to the furthest data point that is still within 1.5 × IQR of the box edges. Any value beyond that boundary is considered a **potential outlier** and plotted as an individual dot. It doesn&#039;t mean the value is definitely wrong — it just flags it for investigation. It could be a legitimate extreme value or a data error.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A box plot shows a data point plotted as an individual dot far below the lower whisker. What does this indicate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The dot represents the minimum value in the dataset regardless of how far it is.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The dot is a value that falls more than 1.5 × IQR below Q1 and is flagged as a potential outlier.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The dot represents the mean of the distribution.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The plotting library made an error — data points should never appear outside the whiskers.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="A grouped box plot is the ideal tool for comparing distributions across multiple groups. Each box shows the median, IQR, and outliers of one school&#039;s scores, and placing them side by side makes it immediately clear which school has the highest median, the most consistent scores (smallest IQR), or the most outliers. Four separate histograms would require more effort to compare, and line charts and pie charts are not designed for comparing distributions.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You have a dataset of test scores for students from four different schools and want to compare their distributions. Which visualization is most appropriate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Four separate histograms plotted one above the other.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>A grouped box plot with one box per school.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>A single line chart with four lines.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>A pie chart showing the proportion of students per school.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

