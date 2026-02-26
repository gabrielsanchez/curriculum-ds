# Introduction to Data Visualization

## Overview

Welcome to the Data Visualization module! In the previous module, you learned how to load, clean, and transform tabular data using Pandas. Now you'll learn how to communicate what that data means. A well-crafted visualization can reveal a trend, expose an outlier, or tell a story that a table of numbers never could. In this lesson, you'll learn why visualization matters in data science and which chart types are best suited to different kinds of data and questions.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Recognize the importance of clear and compelling visualizations.
- Understand basic chart types and their use cases (line, bar, scatter, histogram, and more).

## Key terms

**Data visualization:** The graphical representation of data and information using charts, graphs, maps, or other visual formats to communicate patterns, trends, and insights.

**Chart (graph):** A visual display of data in which values are encoded as visual elements such as bars, lines, or points.

**Axis:** A reference line along which values are plotted. Most charts have an x-axis (horizontal) and a y-axis (vertical).

**Legend:** A key that explains the colors, shapes, or patterns used in a chart to distinguish between different data series or categories.

**Categorical variable:** A variable that takes a limited number of distinct values representing groups or categories (e.g., product type, country, gender).

**Continuous variable:** A variable that can take any numeric value within a range, often measured (e.g., temperature, price, age).

**Distribution:** The way values of a variable are spread across its range, showing how often different values occur.

## Introduction

Numbers alone rarely tell a complete story. A column of 10,000 sales figures tells you very little at a glance. But a line chart of those same figures over time immediately shows whether sales are growing, falling, or seasonal. A histogram of customer ages tells you your target demographic far faster than any summary table.

Data visualization is one of the most important skills a data scientist can develop — not just for building models, but for **understanding** data before analysis and **communicating** results afterward. In the words of data visualization pioneer Edward Tufte: *"The purpose of visualization is insight, not pictures."*

In this module, you'll learn two of Python's most widely used visualization libraries: **Matplotlib** for creating precise, publication-quality static charts, and **Plotly** for building interactive visualizations that let users explore data directly.

## Why Visualization Matters

### Exploring Data

Before you build a model or test a hypothesis, you need to understand your data. Visualization helps you:

- **Spot outliers** — a scatter plot can reveal data points far outside the normal range in seconds.
- **Understand distributions** — a histogram shows whether data is normally distributed, skewed, or multimodal.
- **Identify relationships** — a scatter plot between two variables shows whether they move together.
- **Detect trends** — a line chart shows patterns over time that are invisible in a table.

### Communicating Results

After analysis, visualization helps you share your findings with others:

- A stakeholder who won't read a statistics report will understand a well-labeled bar chart.
- An interactive dashboard lets a business user explore the data themselves.
- A clear chart in a presentation is more persuasive than a slide full of numbers.

### Anscombe's Quartet

A famous example of why visualization is indispensable is **Anscombe's Quartet** — four datasets that share nearly identical summary statistics (mean, variance, correlation, and regression line), but look completely different when plotted:

| Dataset | Mean X | Mean Y | Correlation |
|---------|--------|--------|-------------|
| I       | 9.0    | 7.5    | 0.816       |
| II      | 9.0    | 7.5    | 0.816       |
| III     | 9.0    | 7.5    | 0.816       |
| IV      | 9.0    | 7.5    | 0.817       |

Despite matching statistics, one dataset has a linear relationship, one is curved, one has a single outlier driving the correlation, and one is nearly vertical. You would never know this from the numbers alone. Visualization reveals what statistics hide.

## Chart Types and When to Use Them

Choosing the right chart type is one of the most important decisions in visualization. The wrong chart type can mislead or confuse; the right one makes the insight immediately obvious.

### Line Chart

**Best for:** Trends over time or ordered categories.

A line chart connects data points with a line, making it ideal for showing how a value changes continuously — stock prices, monthly sales, website traffic over weeks.

- The x-axis is usually time or an ordered sequence.
- The y-axis shows the measured value.
- Multiple lines can compare different series (e.g., sales in different regions).

**Ask yourself:** "Am I showing change over time?" → Line chart.

### Bar Chart

**Best for:** Comparing values across discrete categories.

A bar chart uses the length of rectangular bars to represent values. It's the go-to choice when you want to compare a numeric value (like revenue or population) across distinct groups (like countries or product categories).

- **Vertical bars** (column chart) are most common.
- **Horizontal bars** work better when category names are long.
- **Grouped bars** compare multiple series within the same categories.
- **Stacked bars** show part-to-whole relationships within each category.

**Ask yourself:** "Am I comparing counts or totals across categories?" → Bar chart.

### Scatter Plot

**Best for:** Exploring the relationship between two continuous variables.

A scatter plot places each data point at its x- and y-coordinates. The pattern of points reveals whether two variables are correlated (as one increases, so does the other), inversely related, or unrelated.

- Color or size of points can encode a third variable.
- Useful for spotting clusters and outliers.
- The basis of correlation analysis and regression.

**Ask yourself:** "Does one variable change as another changes?" → Scatter plot.

### Histogram

**Best for:** Showing the distribution of a single continuous variable.

A histogram divides a continuous range of values into equal-width bins and shows how many data points fall into each bin. It answers: "How are values distributed? Is the data symmetric, skewed, or bimodal?"

- Different from a bar chart — bins are continuous ranges, not discrete categories, so bars touch each other.
- Useful for checking whether data is normally distributed before modeling.

**Ask yourself:** "How are the values of one variable spread out?" → Histogram.

### Box Plot

**Best for:** Comparing distributions across groups and spotting outliers.

A box plot (or box-and-whisker plot) shows the median, quartiles, and outliers of a distribution in a compact form. It's particularly useful when you want to compare the distribution of a variable across multiple categories side by side.

**Ask yourself:** "How do distributions compare across groups?" → Box plot.

### Pie / Donut Chart

**Best for:** Showing part-to-whole proportions with a small number of categories.

A pie chart shows how a total is divided among parts. Use it sparingly — it becomes hard to read with more than 4–5 slices, and humans are not good at comparing angles. A bar chart often communicates the same information more clearly.

**Ask yourself:** "Do I need to show proportions of a whole?" → Pie chart (with ≤5 categories) or stacked bar chart.

### Heatmap

**Best for:** Showing magnitude across two categorical dimensions, or correlation matrices.

A heatmap encodes values as colors in a grid. You'll use this frequently in data science to visualize correlation matrices between features.

**Ask yourself:** "Do I have a matrix of values I want to compare visually?" → Heatmap.

## Principles of Effective Visualization

A technically correct chart is not automatically a good chart. Keep these principles in mind:

### 1. Choose the right chart type
Match your chart type to the question you're answering and the type of data you have (categorical vs. continuous, one variable vs. two).

### 2. Label clearly
Always include a title, axis labels with units, and a legend when needed. A chart with no labels forces the viewer to guess.

### 3. Avoid chartjunk
Remove unnecessary gridlines, 3D effects, shadows, and decorations that add visual complexity without adding information. Simpler is almost always better.

### 4. Use color purposefully
Use color to encode information (categories, magnitude), not for decoration. Be mindful of colorblind-friendly palettes. Avoid using more than 6–7 distinct colors.

### 5. Don't distort the data
Start bar chart y-axes at 0. Don't truncate axes to exaggerate differences. Don't use 3D charts (they distort proportions).

### 6. Consider your audience
A detailed technical chart is appropriate for a data science team. A clean, annotated chart with minimal jargon is better for a business stakeholder.

## Quick Reference: Chart Type Decision Guide

| Your question | Recommended chart |
|---------------|-------------------|
| How does X change over time? | Line chart |
| How do categories compare? | Bar chart |
| How are two variables related? | Scatter plot |
| How is a variable distributed? | Histogram |
| How do distributions compare across groups? | Box plot |
| What share does each category have? | Pie chart / Stacked bar |
| What is the correlation between many variables? | Heatmap |

## Conclusion

In this lesson, you learned why visualization is a core data science skill — for both exploring data and communicating results. You explored the most common chart types, when each is appropriate, and the principles that separate a clear, insightful visualization from a confusing one. In the next two lessons, you'll build these charts hands-on using **Matplotlib** (for precise static visualizations) and **Plotly** (for interactive visualizations), using real datasets.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="Line charts are designed to show **trends over time** or ordered sequences. Each month&#039;s revenue is a data point, and the line connecting them makes the trend — growth, decline, seasonality — immediately visible. A bar chart could work for month-by-month comparisons, but a line chart is more natural for continuous time-series data.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Which chart type is most appropriate for showing how monthly revenue has changed over the past two years?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Bar chart</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Scatter plot</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Line chart</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Pie chart</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Anscombe&#039;s Quartet consists of four datasets with nearly identical mean, variance, correlation, and regression line — yet they look completely different when plotted. This demonstrates that visualization is essential to understand data and that relying solely on summary statistics can lead to incorrect conclusions.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does Anscombe's Quartet demonstrate about data visualization?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>That all four common chart types produce the same result on the same dataset.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>That summary statistics alone can be misleading — datasets with identical statistics can have completely different patterns when visualized.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>That scatter plots are more accurate than histograms for all types of data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>That you should always use multiple chart types to display the same data.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Bar length encodes value — the eye judges magnitude by how tall the bar is relative to the others. If the y-axis starts at, say, 95 instead of 0, a bar at 98 looks three times taller than a bar at 96, even though the actual difference is only 2%. Starting at 0 ensures the visual proportions reflect the true data proportions.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Why is it generally recommended to start the y-axis of a bar chart at 0?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>It is a Python convention enforced by Matplotlib and Plotly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Starting at a non-zero value can visually exaggerate the differences between bars, misleading the viewer about the true magnitude of the differences.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Bar charts can only display non-negative values, so starting at 0 is the only option.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It makes the chart title easier to read.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

