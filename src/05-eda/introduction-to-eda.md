# Introduction to EDA

## Overview

Welcome to the Exploratory Data Analysis module! In the previous modules, you built the Python and Pandas skills needed to load and manipulate data, and you learned how to communicate findings through visualizations. Now you'll combine those skills into a structured practice called **Exploratory Data Analysis (EDA)** — the process data scientists use to truly understand a dataset before drawing conclusions or building models.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Outline the goals and importance of exploratory data analysis.
- Understand how EDA fits into the data science lifecycle.

## Key terms

**Exploratory Data Analysis (EDA):** An approach to analyzing datasets that uses summary statistics and visualizations to discover patterns, spot anomalies, test assumptions, and understand the structure of the data before formal modeling.

**Data science lifecycle:** The sequence of phases in a data science project, from problem definition and data collection through analysis, modeling, and communication of results.

**Missing values:** Entries in a dataset where no data was recorded, represented in Pandas as `NaN` (Not a Number).

**Data distribution:** The way values of a variable are spread across its possible range — whether concentrated, spread out, symmetric, or skewed.

**Feature:** A single measurable variable in a dataset, corresponding to a column in a DataFrame. Also called a variable or attribute.

**Target variable:** The variable a model is being built to predict (e.g., house price, churn, disease diagnosis).

## Introduction

Imagine you receive a new dataset and are told to build a predictive model. Where do you start? If you jump straight into modeling without first understanding the data, you risk building a model on top of hidden problems: missing values, extreme outliers, mislabeled categories, or features with no meaningful signal. The model may fit the training data but fail in production — and you won't know why.

**Exploratory Data Analysis** is the answer. Coined by statistician John Tukey in his 1977 book of the same name, EDA is a philosophy as much as a process: approach data with an open mind, ask questions, look before you model, and let the data reveal its own structure. As Tukey put it: *"The greatest value of a picture is when it forces us to notice what we never expected to see."*

EDA is not a step you skip when you're in a hurry — it's the step that prevents you from building on a faulty foundation.

## The Data Science Lifecycle

EDA sits at the center of the data science lifecycle, bridging raw data and modeling. A typical lifecycle looks like this:

1. **Problem definition** — What question are you trying to answer? What would a good answer look like?
2. **Data collection** — Gather the data you need from databases, APIs, files, or surveys.
3. **Data cleaning** — Handle missing values, fix errors, standardize formats.
4. **→ Exploratory Data Analysis ←** — Understand the data's structure, distributions, and relationships.
5. **Feature engineering** — Create new variables or transform existing ones based on EDA insights.
6. **Modeling** — Train and evaluate machine learning models.
7. **Communication** — Present results and insights to stakeholders.

EDA often loops back to data cleaning (you discover problems during exploration) and forward to feature engineering (you discover which transformations are needed). It is rarely a one-pass process.

## Goals of EDA

EDA has several concrete objectives:

### 1. Understand the shape and size of the data
- How many rows and columns are there?
- What are the data types of each column?
- How much data is missing?

### 2. Understand each variable individually (univariate analysis)
- What is the range of values?
- What does the distribution look like — is it symmetric, skewed, or bimodal?
- Are there any unusual values?

### 3. Understand relationships between variables (bivariate and multivariate analysis)
- Do two variables tend to move together (correlation)?
- Do distributions differ across groups (e.g., prices in different cities)?
- Which features seem most related to the target variable?

### 4. Identify data quality issues
- Are there outliers that could distort a model?
- Are there duplicate rows?
- Are there categorical values with typos or inconsistent formatting?

### 5. Generate hypotheses
- Based on what you observe, what patterns might be worth testing or modeling?

## A Standard EDA Checklist

When you start a new dataset, work through this checklist:

| Step | Action | Pandas / Tool |
|------|--------|---------------|
| **1. Load** | Read the data | `pd.read_csv()` |
| **2. Shape** | Check dimensions | `df.shape` |
| **3. Types** | Check column data types | `df.dtypes`, `df.info()` |
| **4. Missing** | Count missing values | `df.isnull().sum()` |
| **5. Duplicates** | Check for duplicate rows | `df.duplicated().sum()` |
| **6. Summary stats** | Compute mean, std, min, max | `df.describe()` |
| **7. Distributions** | Plot histograms for numeric columns | `df.hist()` |
| **8. Box plots** | Compare distributions, spot outliers | `df.boxplot()` |
| **9. Correlation** | Compute pairwise correlations | `df.corr()` |
| **10. Relationships** | Scatter plots for key variable pairs | `plt.scatter()` |

## A Quick First Look in Practice

Here's what the very first lines of an EDA session typically look like:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("housing.csv")

# Shape and types
print(df.shape)
df.info()

# Missing values
print(df.isnull().sum())

# Duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# Summary statistics
print(df.describe())
```

This small block of code immediately answers: How big is the dataset? What columns exist and what types are they? Where is data missing? Are there duplicates? What are the ranges of numeric columns? Getting this information in the first two minutes of working with a dataset is standard practice.

## What EDA Is Not

It's worth clarifying what EDA is **not**:

- **Not hypothesis testing** — EDA generates hypotheses; statistical testing confirms them.
- **Not final** — EDA insights should inform modeling decisions, not replace them.
- **Not a fixed procedure** — Every dataset is different. The checklist above is a starting point, not a rigid script. Follow what the data tells you.

## Conclusion

In this lesson, you learned what Exploratory Data Analysis is, why it is an essential step in every data science project, and where it fits in the data science lifecycle. You now have a standard checklist to follow whenever you encounter a new dataset. In the lessons that follow, you'll build out each step of that checklist in depth: computing and interpreting summary statistics, visualizing distributions with histograms and box plots, measuring relationships with a correlation matrix, and detecting and handling outliers.

## Practice

### Knowledge Check

#### **Question 1: Where does EDA sit in the data science lifecycle?**
1. At the very end, after modeling is complete, to validate results.
2. Before data collection, to plan what data to gather.
3. After data cleaning and before feature engineering and modeling.
4. EDA is a one-time step done only at the beginning of a project.

**Correct Answer:**
3. After data cleaning and before feature engineering and modeling.

**Explanation:**
EDA comes after the data has been collected and initially cleaned, and before you commit to specific modeling decisions. However, EDA often loops back — you discover new cleaning needs during exploration — and forward — EDA insights directly inform feature engineering choices. It is an iterative, central part of the workflow.

---

#### **Question 2: Which of the following is a goal of EDA?**
1. Training a machine learning model on the full dataset.
2. Writing the final report for stakeholders.
3. Understanding the distribution of variables, identifying data quality issues, and generating hypotheses.
4. Choosing a production deployment strategy for the model.

**Correct Answer:**
3. Understanding the distribution of variables, identifying data quality issues, and generating hypotheses.

**Explanation:**
EDA is about **understanding** the data — its structure, distributions, relationships, and problems. It does not involve training models or writing final reports. The insights from EDA guide all subsequent steps, including data cleaning, feature engineering, and model selection.

---

#### **Question 3: Why is it important to run `df.isnull().sum()` early in EDA?**
1. It identifies which rows have duplicate values.
2. It reveals how many missing values exist per column, which affects how you clean the data and which analyses are valid.
3. It computes the mean and standard deviation of each column.
4. It checks whether the dataset has been correctly loaded from the CSV file.

**Correct Answer:**
2. It reveals how many missing values exist per column, which affects how you clean the data and which analyses are valid.

**Explanation:**
Missing values are one of the most common data quality issues and can silently distort summary statistics, correlations, and model training. Knowing early where data is missing allows you to decide whether to drop affected rows, impute values, or investigate whether the missingness itself is meaningful (e.g., a missing income field could indicate unemployed respondents).
