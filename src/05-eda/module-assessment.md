# Module Assessment

## Overview

Congratulations on completing the Exploratory Data Analysis module! You've built the full EDA toolkit: summary statistics, histograms, box plots, correlation matrices, and outlier detection and treatment. This assessment brings all of those skills together in a single, realistic end-to-end EDA project on a new dataset.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/05-eda-module-assessment_starter.ipynb).

## Learning Objective

By the end of this assessment, you will have demonstrated your ability to:

- Apply EDA techniques to a new dataset.
- Summarize findings and identify potential anomalies.

## Key terms

**EDA report:** A structured summary of the findings from an exploratory analysis — including data quality issues, distribution characteristics, key relationships, and hypotheses for further investigation.

**Anomaly:** An unexpected pattern or value in the data that warrants further investigation — could be an outlier, a suspicious cluster of missing values, an impossible category value, or an unexpected correlation.

**Feature selection hypothesis:** A preliminary judgment, based on EDA, about which features are likely to be useful predictors for a given modeling task.

## Assessment Overview

This module assessment is a coding project completed in a Google Colaboratory notebook. You will receive a new dataset — a fictional retail store dataset with sales, product, and customer information — and perform a complete EDA, documenting your findings at each step.

### Skills Assessed

| Skill | Lesson |
|-------|--------|
| Loading data and performing an initial inspection | Introduction to EDA |
| Computing and interpreting summary statistics | Summary Statistics |
| Generating and reading histograms | Histograms |
| Creating and interpreting box plots | Box Plots |
| Computing and visualizing a correlation matrix | Correlation Matrix |
| Detecting and treating outliers | Outliers |
| Communicating findings in writing | All lessons |

## Coding Assessment

Complete the project in this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/05-eda/05-eda-module-assessment_starter.ipynb). After completing all tasks, save your notebook to GitHub and [submit the link for grading](https://ai-grader-pql9.onrender.com/).

### Task Overview

The dataset contains the following columns: `order_id`, `customer_age`, `product_category`, `units_sold`, `unit_price`, `total_revenue`, `discount_pct`, `region`, `days_to_ship`.

You will complete the following tasks:

1. **Initial inspection** — Load the dataset and display its shape, column types, and first 10 rows. Count missing values per column. Count duplicate rows. Write two sentences describing what the dataset contains.

2. **Summary statistics** — Run `.describe()` on all numeric columns. For any column where the mean and median differ by more than 20%, explain in one sentence why that gap exists and what it implies about the distribution.

3. **Histograms** — Plot histograms (with mean and median lines) for `customer_age`, `total_revenue`, and `days_to_ship`. For each, classify the distribution shape (normal, right-skewed, left-skewed, or bimodal) and write one sentence of interpretation.

4. **Box plots** — Create grouped box plots comparing `total_revenue` by `region` and `unit_price` by `product_category`. Identify which region has the highest median revenue and which category has the most outliers.

5. **Correlation matrix** — Compute and visualize the full correlation matrix as a heatmap. List the three strongest feature-to-feature correlations and explain whether each one makes intuitive sense.

6. **Outlier detection** — Apply the IQR method to `total_revenue` and `days_to_ship`. Print the outlier count and boundary values for each. Examine the flagged rows and classify each as a data error or a legitimate extreme value. Apply appropriate treatment (remove, cap, or keep with justification).

7. **EDA summary** — Write a short paragraph (4–6 sentences) summarizing the most important findings from your EDA: what the data looks like, any data quality issues you found, the strongest relationships between features, and which features you would prioritize as predictors if building a model to predict `total_revenue`.

### Grading Rubric

| Criteria | Points |
|----------|--------|
| Initial inspection completed with written description | 10 |
| Summary statistics computed and mean/median gaps explained | 10 |
| Histograms with mean/median lines and correct shape classification | 15 |
| Grouped box plots correctly created and interpreted | 15 |
| Correlation heatmap correct; three correlations identified and explained | 15 |
| Outlier detection applied; rows classified and treated with justification | 20 |
| EDA summary paragraph is accurate and clearly written | 15 |
| **Total** | **100** |

## Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="The cardinal rule of outlier treatment is: investigate before acting. Discounts above 90% could be legitimate clearance sales, employee discounts, or bundle promotions — in which case they should be kept. Or they could be data entry errors. Removing them without investigation risks deleting real business events; keeping errors without treatment risks distorting your analysis. The evidence (mean vs. median gap, flagged values) is a prompt to investigate, not a command to delete.">
  <div class="quiz-question">
    <strong>Question 1:</strong> During EDA, you find that the `discount_pct` column has a mean of 18% and a median of 12%. You also find 15 rows where `discount_pct` is greater than 90%. What should your next steps be?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Immediately remove all rows where `discount_pct` > 90% and recompute statistics.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Apply a log transformation to `discount_pct` to reduce skewness.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Investigate whether discounts above 90% are real promotions or data errors, then decide on treatment based on your findings.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Replace all values above 90% with the mean value of 18%.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="When two features are nearly perfectly correlated (0.97), a regression model cannot reliably separate their individual contributions. This multicollinearity inflates standard errors and makes coefficients unstable. A practical solution is to drop one of the correlated features, or better yet, engineer a new feature that captures the unique information (e.g., `avg_unit_price = total_revenue / units_sold`).">
  <div class="quiz-question">
    <strong>Question 2:</strong> Your EDA reveals that `units_sold` and `total_revenue` have a correlation of 0.97. Should you include both as features in a regression model predicting `profit`? Why or why not?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Yes — high correlation means both features are valuable predictors and should always be included.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>No — you should include neither feature because correlated variables are unreliable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Including both may cause multicollinearity, making coefficient estimates unstable. Consider using only one, or creating a derived feature like average unit price instead.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The correlation is too high to be real — recheck your calculation before making any decision.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="A bimodal distribution often indicates that two distinct subpopulations have been combined into a single variable. Two shipping peaks (2 days and 12 days) strongly suggest two shipping tiers (express vs. standard) or two fulfillment centers with different lead times. The right next step is to segment by a categorical variable (like `shipping_type` or `region`) to confirm the source of the bimodality — this is a valuable EDA insight that would directly affect feature engineering.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You plot a histogram of `days_to_ship` and see a clear bimodal distribution with one peak around 2 days and another around 12 days. What is the most likely explanation, and what would you do next?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The data is corrupted — a bimodal distribution is always a sign of a data error. Remove all rows in the lower peak.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The distribution is right-skewed, not bimodal — apply a log transformation to correct it.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>There may be two distinct groups in the data (e.g., standard and express shipping, or two regional warehouses). Investigate by examining the distribution of `days_to_ship` broken down by a categorical variable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Use Winsorization to cap `days_to_ship` at the median value to flatten the distribution.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

