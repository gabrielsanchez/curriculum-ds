# Module Assessment

## Overview

Congratulations on completing the Data Visualization module! You've learned the principles of choosing the right chart for the right question, built static charts with Matplotlib, and created interactive visualizations with Plotly. This assessment brings those skills together in a project where you'll analyze a real dataset and communicate your findings through multiple visualizations.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/04-data-visualization/04-data-disualization-module-assessment_starter.ipynb).

## Learning Objective

By the end of this assessment, you will have demonstrated your ability to:

- Apply various visualization techniques to a given dataset.
- Effectively communicate insights through visuals.

## Key terms

**Insight:** A meaningful observation drawn from data — a pattern, trend, anomaly, or relationship — that answers a question or supports a decision.

**Visual encoding:** The way data values are mapped to visual properties (position, length, color, size, shape) in a chart.

**Annotation:** Text, arrows, or highlights added directly to a chart to draw attention to a specific data point or insight.

## Assessment Overview

This module assessment is a coding project completed in a Google Colaboratory notebook. You will be given a dataset and asked to produce a series of charts using both Matplotlib and Plotly, each designed to answer a specific analytical question. For each chart, you will also write a one- to two-sentence interpretation explaining what the visualization reveals.

### Skills Assessed

| Skill | Lesson |
|-------|--------|
| Selecting appropriate chart types | Introduction to Data Visualization |
| Creating scatter plots, line charts, and bar charts | Matplotlib |
| Customizing titles, labels, and colors | Matplotlib |
| Creating subplots | Matplotlib |
| Creating interactive charts with Plotly | Plotly |
| Customizing Plotly layouts and themes | Plotly |
| Interpreting and communicating visual insights | All lessons |

## Coding Assessment

Complete the project in this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/04-data-visualization/04-data-disualization-module-assessment_starter.ipynb). After completing all tasks, save your notebook to GitHub and submit the link to the [AI Grader](https://ai-grader-pql9.onrender.com/) for grading.

### Task Overview

The notebook provides a dataset containing information about global city populations, GDP per capita, average temperature, and quality-of-life scores across multiple years and continents. You will complete the following tasks:

1. **Line chart (Matplotlib)** — Plot the average quality-of-life score over time for at least three continents on the same axes, using a different color for each. Add a title, axis labels, legend, and grid. Write one sentence explaining the trend.

2. **Bar chart (Matplotlib)** — Create a horizontal bar chart comparing the average GDP per capita by continent for the most recent year in the dataset. Sort the bars from highest to lowest. Add a title and axis labels.

3. **Scatter plot (Matplotlib)** — Plot GDP per capita (x-axis) vs. quality-of-life score (y-axis) for all cities in the most recent year. Color the points by continent. Add a title, axis labels, and a legend. In one sentence, describe whether a relationship is visible.

4. **Subplots (Matplotlib)** — Create a figure with two side-by-side subplots: a histogram of city populations and a histogram of average temperatures. Give each subplot its own title and axis labels.

5. **Interactive scatter plot (Plotly)** — Recreate the scatter plot from Task 3 using Plotly Express, adding a `hover_data` parameter so that city names appear when hovering. Apply the `plotly_white` template.

6. **Interactive bar chart (Plotly)** — Create a grouped bar chart using Plotly that compares average GDP and average quality-of-life score for each continent side by side. Add a descriptive title and axis labels using `update_layout()`.

7. **Insight summary** — Write three bullet points (one per visualization of your choice) identifying the most important insight you found and which chart type made it visible.

### Grading Rubric

| Criteria | Points |
|----------|--------|
| Matplotlib line chart with correct data, labels, and legend | 15 |
| Matplotlib bar chart sorted and correctly labeled | 10 |
| Matplotlib scatter plot with color encoding and legend | 15 |
| Matplotlib subplots created correctly with individual labels | 10 |
| Plotly interactive scatter plot with hover data | 15 |
| Plotly grouped bar chart with `update_layout()` customization | 15 |
| Written insight summary (3 bullet points) | 10 |
| Code is clean, readable, and uses descriptive variable names | 10 |
| **Total** | **100** |

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Pie charts rely on humans comparing the sizes of angles and areas, which we do poorly even with a handful of slices. With 12 categories, many slices will be so thin they become indistinguishable. A horizontal bar chart sorted by value would communicate the same information far more clearly, making it easy to rank and compare categories.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A colleague creates a pie chart to compare the revenue of 12 product categories. What is the main problem with this choice?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Pie charts can only display data from two categories.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>With 12 slices, the chart becomes extremely difficult to read because humans struggle to compare many angles and small areas accurately.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Pie charts cannot display revenue data because it is a continuous variable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Plotly does not support pie charts with more than 5 categories.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The defining advantage of Plotly is **interactivity**. While Matplotlib produces static images (ideal for reports and publications), Plotly charts render in the browser and allow users to explore the data directly — hovering over points to see exact values, clicking legend items to show/hide series, and zooming in on regions of interest. This makes Plotly especially powerful for dashboards and exploratory presentations.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What is the key advantage of Plotly charts over Matplotlib charts?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Plotly charts always look better than Matplotlib charts.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Plotly charts are interactive — users can zoom, pan, hover to see data values, and toggle series on and off.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Plotly charts require less code to create.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Plotly is faster to render than Matplotlib for large datasets.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="A histogram divides house prices into bins and shows how many homes fall into each price range, revealing whether the distribution is symmetric, skewed (many cheap homes, few expensive ones), or bimodal (two distinct price clusters). A line chart requires an ordered sequence, a bar chart compares discrete categories, and a scatter plot needs two variables — none of which apply here.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You want to visualize the distribution of house prices in a dataset of 5,000 homes. Which chart type is most appropriate, and why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Line chart — because house prices are a continuous variable measured over time.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Bar chart — because you need to compare prices across different houses.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Histogram — because it shows how a single continuous variable is distributed across a range of values.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Scatter plot — because you need two variables to create a meaningful chart.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

