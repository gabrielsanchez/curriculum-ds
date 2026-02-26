# Introduction to Regression

## Overview

In module 09, you learned to predict **discrete categories** — edible or poisonous, survived or not. Many real-world problems require something different: predicting a **number** on a continuous scale. How much will this house sell for? How many units will we ship next month? What will a patient's blood pressure be in six months? These questions call for **regression** — the other major family of supervised learning. In module 08, you spent an entire case study preparing data for a regression model. Now you'll build and evaluate those models.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Differentiate regression from classification.
- Recognize common regression use cases and match problems to the right approach.

## Key Terms

**Regression:** A supervised learning task where the target variable is a continuous numeric value. The model learns a function that maps feature values to a number on a continuous scale.

**Continuous target:** An output variable that can take any numeric value within a range (e.g., price, temperature, duration, rate).

**Residual:** The difference between a prediction and the true value: `residual = y_true − y_pred`. Residuals are the model's errors, and analyzing them reveals systematic problems.

**Baseline model:** The simplest possible prediction you could make — for regression, this is often predicting the mean of the training target for every new sample. Any useful model must beat this baseline.

**Underfitting:** A model that is too simple to capture the patterns in the data. It performs poorly on both training and test data.

**Overfitting:** A model that has memorized the training data so thoroughly that it fails to generalize. It performs well on training data but poorly on test data.

**Bias-variance trade-off:** The tension between underfitting (high bias, low variance) and overfitting (low bias, high variance). Good models find the middle ground.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

The distinction between classification and regression comes down to the **type of answer you're looking for**:

| Question | Answer type | Task |
|----------|-------------|------|
| Will this customer churn? | yes / no | Classification |
| How much revenue will we earn? | $412,000 | Regression |
| What species is this flower? | setosa / virginica | Classification |
| How long will this patient stay in hospital? | 4.2 days | Regression |
| Is this email spam? | spam / not spam | Classification |
| What is this apartment's monthly rent? | ₹22,000 | Regression |

When the answer is a number that can take any value in a range — not a fixed label from a set — it's regression.

## Common Regression Use Cases

| Domain | Regression Application |
|--------|----------------------|
| **Real estate** | House price prediction, rental price estimation |
| **Finance** | Stock return forecasting, loan default amount, insurance premium |
| **Supply chain** | Demand forecasting, inventory optimization |
| **Healthcare** | Patient readmission risk score, drug dosage optimization |
| **Energy** | Electricity consumption forecasting, solar output prediction |
| **Marketing** | Customer lifetime value, ad click-through rate |
| **Manufacturing** | Equipment failure time prediction, quality score estimation |

In each case, the model outputs a number, not a category.

## The Regression Pipeline

The regression pipeline closely mirrors the classification pipeline from module 09:

```
1. Define the problem (what number are you predicting?)
        ↓
2. Collect and explore data (EDA — you practiced this in modules 05 and 08)
        ↓
3. Clean and engineer features (module 07 and 08)
        ↓
4. Train/test split
        ↓
5. Train a regression model
        ↓
6. Evaluate using regression metrics (MAE, RMSE, R²)
        ↓
7. Iterate (different algorithms, hyperparameter tuning)
```

Steps 1–4 are familiar. This module focuses on steps 5–7.

## The Dataset: California Housing

Throughout this module, you'll use the **California Housing dataset** — a standard regression benchmark built into scikit-learn. It contains data from the 1990 California census, with one row per census block group.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Shape: {df.shape}")
print(f"\nFeatures:")
print(housing.DESCR[housing.DESCR.index("Attribute"):housing.DESCR.index("Missing")])
```

Output:
```
Shape: (20640, 9)

Attribute Information:
  MedInc      Median income in block group (in $10,000s)
  HouseAge    Median house age in block group
  AveRooms    Average number of rooms per household
  AveBedrms   Average number of bedrooms per household
  Population  Block group population
  AveOccup    Average number of household members
  Latitude    Block group latitude
  Longitude   Block group longitude
  MedHouseVal Median house value in $100,000s — this is the target
```

```python
print(df.describe().round(2))
```

Output:
```
        MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
count  20640.00  20640.00  20640.00   20640.00    20640.00  20640.00  20640.00   20640.00     20640.00
mean       3.87     28.64      5.43       1.10     1425.48      3.07     35.63    -119.57         2.07
std        1.90     12.59      2.47       0.47     1132.46     10.39      2.14       2.00         1.15
min        0.50      1.00      0.85       0.33        3.00      0.69     32.54    -124.35         0.15
25%        2.56     18.00      4.44       1.01      787.00      2.43     33.93    -121.80         1.20
50%        3.53     29.00      5.23       1.05     1166.00      2.82     34.26    -118.49         1.80
75%        4.74     37.00      6.05       1.10     1725.00      3.28     37.71    -118.01         2.65
max       15.00     52.00    141.91      34.07    35682.00   1243.33     41.95    -114.31         5.00
```

The target `MedHouseVal` is the median house value in units of $100,000 — so a value of 2.0 means $200,000. The dataset has 20,640 rows and no missing values.

## Your First Regression Model

Let's build a complete pipeline end-to-end:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train/test split (no stratify for regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target mean: ${y_train.mean() * 100_000:,.0f}")
```

Output:
```
Train: (16512, 8), Test: (4128, 8)
Target mean: $206,600
```

```python
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae * 100_000:,.0f}")
print(f"R² Score:            {r2:.3f}")
```

Output:
```
Mean Absolute Error: $52,700
R² Score:            0.576
```

The model's predictions are off by an average of about $53,000, and it explains 57.6% of the variance in house prices. This is a reasonable starting point for a simple linear model — the next four lessons will show you how to interpret and improve it.

## The Baseline: How Good Must "Good" Be?

Before declaring any model result good or bad, compute the **mean baseline** — the result you'd get by always predicting the training set mean:

```python
from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_baseline = baseline.predict(X_test)

mae_base = mean_absolute_error(y_test, y_baseline)
r2_base  = r2_score(y_test, y_baseline)

print(f"Baseline MAE: ${mae_base * 100_000:,.0f}")
print(f"Baseline R²:  {r2_base:.3f}")
```

Output:
```
Baseline MAE: $90,800
Baseline R²:  0.000
```

The baseline (always predict the mean) has R² = 0 by definition and MAE of $90,800. Our linear model reduces error by 42% ($90,800 → $52,700) and explains 57.6% of variance. That's meaningful progress — but there's room to improve, as you'll see in the lessons that follow.

## A Map of This Module

| Lesson | Topic | Key concepts |
|--------|-------|-------------|
| 1 (this one) | Introduction | Regression vs. classification, pipeline, first model |
| 2 | Linear Regression | OLS, coefficients, residuals, assumptions |
| 3 | Additional Regression Methods | Polynomial regression, decision tree regression |
| 4 | Regularization | Ridge (L2), Lasso (L1), overfitting, alpha tuning |
| 5 | Model Evaluation | MAE, RMSE, R², cross-validation, model comparison |

## Conclusion

In this lesson, you established that regression predicts continuous numbers rather than discrete categories, traced the regression pipeline through familiar territory from modules 07 and 08, loaded the California Housing dataset, and built your first linear regression model — reducing prediction error by 42% over the mean baseline. In the next lesson, you'll open the black box: learning how linear regression works, what its coefficients mean, and how to diagnose its errors with residual analysis.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="&quot;More than $500&quot; is a binary outcome — yes or no — which is binary classification. Predicting an exact dollar amount (e.g., $347.82) is a regression task because the target is a continuous number. The same underlying data can be framed as either a classification or regression problem depending on what business question you&#039;re answering. Choosing the right framing affects both the algorithm and the evaluation metrics.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A data scientist is asked to predict whether a customer will spend more than $500 in the next month. Their manager then asks them to instead predict exactly how much the customer will spend. How does this change the ML task?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>It doesn't change anything — both questions use the same algorithms.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The first question is regression (predicting a threshold); the second is classification (predicting a label).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The first question is binary classification (above/below $500); the second is regression (predicting a continuous dollar amount).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Both are classification problems because spending categories can be defined.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="R² (the coefficient of determination) measures the proportion of variance in the target that the model accounts for. R² = 0 corresponds to always predicting the mean. R² = 1 corresponds to perfect predictions. R² = 0.55 means the model captures 55% of the variation — meaningfully better than the baseline. Whether 0.55 is &quot;good enough&quot; depends on the domain: housing prices may have inherent noise (neighborhood factors, renovation quality) that no model can capture, so 0.55 might be close to the achievable ceiling.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A linear regression model achieves R² = 0.55 on the test set. The mean baseline (always predict the training mean) achieves R² = 0.0. How should you interpret the linear regression result?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>R² = 0.55 is a poor result — a good model should always have R² > 0.9.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>R² = 0.55 means the model explains 55% of the variance in the target, which is a substantial improvement over the mean baseline, though there may be room to improve further.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>R² = 0.55 means the model is correct 55% of the time.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>R² = 0.55 means the model makes errors 55% of the time.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="`stratify` in scikit-learn requires a categorical array — it ensures each category appears proportionally in both splits. Continuous regression targets aren&#039;t categorical, so `stratify` can&#039;t be used directly. For a dataset of 50,000 samples, a random 80/20 split will produce train and test sets with very similar target distributions by the law of large numbers. If the dataset were small (a few hundred samples), you might bin the target into quantiles and stratify on those bins to ensure balance — but this is rarely needed for large datasets.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You have a dataset of 50,000 customer records and want to predict annual purchase amount. You split the data 80/20 into train and test sets. A colleague suggests you use `stratify=y` as you did for classification. Should you?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Yes — stratification is always required to get valid evaluation results.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>No — `stratify` works with discrete class labels, not continuous values. For regression, a random split is appropriate; the large sample size (40,000 train, 10,000 test) makes the split reliable without stratification.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Yes — stratifying on the target ensures the mean purchase amount is the same in both sets.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>No — you should always use all data for training and never hold out a test set.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

