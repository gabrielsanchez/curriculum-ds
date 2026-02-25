# Module Assessment

## Overview

Congratulations on completing the Exploratory Data Analysis module! You've built the full EDA toolkit: summary statistics, histograms, box plots, correlation matrices, and outlier detection and treatment. This assessment brings all of those skills together in a single, realistic end-to-end EDA project on a new dataset.

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

Complete the project in this [notebook](#). After completing all tasks, save your notebook to GitHub and submit the link for grading.

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

#### **Question 1: During EDA, you find that the `discount_pct` column has a mean of 18% and a median of 12%. You also find 15 rows where `discount_pct` is greater than 90%. What should your next steps be?**
1. Immediately remove all rows where `discount_pct` > 90% and recompute statistics.
2. Apply a log transformation to `discount_pct` to reduce skewness.
3. Investigate whether discounts above 90% are real promotions or data errors, then decide on treatment based on your findings.
4. Replace all values above 90% with the mean value of 18%.

**Correct Answer:**
3. Investigate whether discounts above 90% are real promotions or data errors, then decide on treatment based on your findings.

**Explanation:**
The cardinal rule of outlier treatment is: investigate before acting. Discounts above 90% could be legitimate clearance sales, employee discounts, or bundle promotions — in which case they should be kept. Or they could be data entry errors. Removing them without investigation risks deleting real business events; keeping errors without treatment risks distorting your analysis. The evidence (mean vs. median gap, flagged values) is a prompt to investigate, not a command to delete.

---

#### **Question 2: Your EDA reveals that `units_sold` and `total_revenue` have a correlation of 0.97. Should you include both as features in a regression model predicting `profit`? Why or why not?**
1. Yes — high correlation means both features are valuable predictors and should always be included.
2. No — you should include neither feature because correlated variables are unreliable.
3. Including both may cause multicollinearity, making coefficient estimates unstable. Consider using only one, or creating a derived feature like average unit price instead.
4. The correlation is too high to be real — recheck your calculation before making any decision.

**Correct Answer:**
3. Including both may cause multicollinearity, making coefficient estimates unstable. Consider using only one, or creating a derived feature like average unit price instead.

**Explanation:**
When two features are nearly perfectly correlated (0.97), a regression model cannot reliably separate their individual contributions. This multicollinearity inflates standard errors and makes coefficients unstable. A practical solution is to drop one of the correlated features, or better yet, engineer a new feature that captures the unique information (e.g., `avg_unit_price = total_revenue / units_sold`).

---

#### **Question 3: You plot a histogram of `days_to_ship` and see a clear bimodal distribution with one peak around 2 days and another around 12 days. What is the most likely explanation, and what would you do next?**
1. The data is corrupted — a bimodal distribution is always a sign of a data error. Remove all rows in the lower peak.
2. The distribution is right-skewed, not bimodal — apply a log transformation to correct it.
3. There may be two distinct groups in the data (e.g., standard and express shipping, or two regional warehouses). Investigate by examining the distribution of `days_to_ship` broken down by a categorical variable.
4. Use Winsorization to cap `days_to_ship` at the median value to flatten the distribution.

**Correct Answer:**
3. There may be two distinct groups in the data (e.g., standard and express shipping, or two regional warehouses). Investigate by examining the distribution of `days_to_ship` broken down by a categorical variable.

**Explanation:**
A bimodal distribution often indicates that two distinct subpopulations have been combined into a single variable. Two shipping peaks (2 days and 12 days) strongly suggest two shipping tiers (express vs. standard) or two fulfillment centers with different lead times. The right next step is to segment by a categorical variable (like `shipping_type` or `region`) to confirm the source of the bimodality — this is a valuable EDA insight that would directly affect feature engineering.
