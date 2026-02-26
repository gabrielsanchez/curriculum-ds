# Linear Regression

## Overview

In the previous lesson, you built your first linear regression model in a few lines and saw that it explained 57.6% of variance in California house prices. In this lesson, you'll open that black box: understand how ordinary least squares works, what the learned coefficients actually mean, how to detect systematic errors with residual analysis, and what assumptions linear regression relies on. Understanding the mechanics of one algorithm deeply is more valuable than knowing five algorithms superficially — and linear regression is the foundation everything else in this module builds on.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Fit a linear regression model and interpret its coefficients.
- Analyze residuals to diagnose model problems.

## Key Terms

**Ordinary Least Squares (OLS):** The standard method for fitting linear regression. It finds the coefficients that minimize the sum of squared residuals.

**Coefficient (weight):** A learned parameter in the linear model. It represents the change in the predicted target for a one-unit increase in that feature, holding all other features constant.

**Intercept:** The predicted target value when all features are zero. Also called the bias term.

**Residual:** `y_true − y_pred`. Positive means the model underestimated; negative means it overestimated.

**Homoscedasticity:** The assumption that residuals have roughly constant variance across all predicted values. Violated when errors grow larger for larger predictions (heteroscedasticity).

**Multicollinearity:** When two or more features are highly correlated with each other. This inflates coefficient standard errors and makes individual coefficients unreliable.

**Normal equations:** The closed-form mathematical solution to OLS. For large datasets, an iterative solver (gradient descent) is used instead.

**Feature scaling:** Standardizing features to have mean=0 and std=1. Not required for OLS accuracy, but makes coefficients comparable and speeds up iterative solvers.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## How Linear Regression Works

A linear regression model predicts the target as a **weighted sum** of the input features plus an intercept:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
```

Where:
- `ŷ` is the predicted value
- `x₁, x₂, ..., xₙ` are the feature values
- `w₁, w₂, ..., wₙ` are the learned weights (coefficients)
- `w₀` is the intercept

**The algorithm's job:** find the values of `w₀, w₁, ..., wₙ` that minimize the total prediction error on the training data.

### The Cost Function: Mean Squared Error

OLS minimizes the **sum of squared residuals**:

```
Cost = Σ (yᵢ − ŷᵢ)²
```

Squaring the residuals ensures:
1. Positive and negative errors don't cancel out
2. Large errors are penalized more severely than small ones (a residual of 2 costs 4×, not 2×, as much as a residual of 1)

The solution to this minimization problem is exact — it can be computed analytically using linear algebra (the normal equations). For datasets that fit in memory, scikit-learn uses this direct approach.

### A Visual Intuition

For a single feature, linear regression fits the best straight line through the data:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Plot MedInc vs MedHouseVal (the strongest linear relationship)
plt.figure(figsize=(8, 5))
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.1, s=5, color="steelblue")

# Fit and plot the line
lr_single = LinearRegression()
lr_single.fit(df[["MedInc"]], df["MedHouseVal"])
x_range = np.linspace(df["MedInc"].min(), df["MedInc"].max(), 100).reshape(-1, 1)
plt.plot(x_range, lr_single.predict(x_range), color="red", linewidth=2)

plt.xlabel("Median Income ($10,000s)")
plt.ylabel("Median House Value ($100,000s)")
plt.title("Linear Regression: Income → House Value")
plt.tight_layout()
plt.show()

print(f"Slope: {lr_single.coef_[0]:.4f}")
print(f"Intercept: {lr_single.intercept_:.4f}")
```

Output:
```
Slope: 0.4170
Intercept: 0.4505
```

For every $10,000 increase in median income, the model predicts a $41,700 increase in median house value. At zero income, it predicts $45,050 (the intercept) — not meaningful here, but mathematically required.

## Setup: Full Multi-Feature Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (not required for OLS accuracy, but makes coefficients comparable)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
print(f"R²:  {r2_score(y_test, y_pred):.3f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred) * 100_000:,.0f}")
```

Output:
```
R²:  0.576
MAE: $52,700
```

## Interpreting Coefficients

With scaled features, the magnitude of each coefficient reflects its relative importance:

```python
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print(coef_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")
```

Output:
```
      feature  coefficient
       MedInc       0.8290    ← strongest positive predictor
     Latitude      -0.8010    ← further south → lower value
    Longitude      -0.7310    ← coastal → higher value
     AveRooms       0.3010
     HouseAge       0.1260
   AveBedrms       -0.2780
    AveOccup       -0.0423
   Population      -0.0081

Intercept: 2.0724
```

### Reading the Coefficients

Because features are scaled (z-scored), coefficients measure change in **standard deviation units**:

- **MedInc (0.829):** A 1 standard deviation increase in median income is associated with a $82,900 increase in predicted house value, holding all other features constant.
- **Latitude (−0.801):** Moving 1 standard deviation further north is associated with a $80,100 *decrease* in value. This reflects the California pattern: southern coastal cities (LA, San Diego) are expensive but so are northern ones (SF) — the relationship is non-linear, which explains why this isn't the strongest predictor.
- **AveBedrms (−0.278):** More bedrooms per household is associated with lower prices. This seems counterintuitive — but this coefficient controls for `AveRooms`. Given the same total rooms, more bedrooms means smaller individual rooms, which lowers value.

**Caution:** Coefficients measure **association, not causation**. Adding bedrooms to your house won't lower its value — this reflects the correlation structure of the data, not a causal mechanism.

### Multicollinearity Check

`AveRooms` and `AveBedrms` are correlated (more rooms usually means more bedrooms). Check the correlation:

```python
print(f"Correlation between AveRooms and AveBedrms: {df['AveRooms'].corr(df['AveBedrms']):.3f}")
```

Output:
```
Correlation between AveRooms and AveBedrms: 0.847
```

High correlation (0.847) — multicollinearity is present. This inflates the uncertainty on both coefficients and makes their individual values less reliable, even if the model's overall predictions are fine. In practice, you'd either drop one, combine them (e.g., `AveRooms − AveBedrms` = average non-bedroom rooms), or use regularization (covered in lesson 4).

## Residual Analysis

Residuals are the model's errors: `residual = y_true − y_pred`. Analyzing them reveals whether the model is missing systematic patterns.

### Residuals vs. Predicted Values

```python
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: residuals vs. predicted values
axes[0].scatter(y_pred, residuals, alpha=0.2, s=5, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Predicted Value ($100,000s)")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs. Predicted Values")

# Plot 2: histogram of residuals
axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Residuals")

plt.tight_layout()
plt.show()
```

What to look for in the residual vs. predicted plot:

| Pattern | Diagnosis |
|---------|-----------|
| Random scatter around zero | Good — no systematic bias |
| Fan shape (variance increases with prediction) | Heteroscedasticity — log-transform the target |
| U-shape or curve | Non-linearity — the model is missing a non-linear relationship |
| Cluster of large positive residuals at high predictions | Ceiling effect — predictions are capped (e.g., dataset has a $500,000 cap) |

For the California housing data, you'll observe a fan shape and a cluster of large residuals at high predicted values — partly because the dataset caps `MedHouseVal` at $500,001, truncating all expensive properties at the same value.

### Residual Distribution

A well-specified linear regression model should have residuals that are approximately normally distributed around zero. The histogram lets you check this visually.

```python
print(f"Residual mean:   {residuals.mean():.4f}  (should be ≈ 0)")
print(f"Residual std:    {residuals.std():.4f}")
print(f"Residual min:    {residuals.min():.3f}")
print(f"Residual max:    {residuals.max():.3f}")
```

Output:
```
Residual mean:   0.0000  (should be ≈ 0)
Residual std:    0.7534
Residual min:   -4.068
Residual max:    3.507
```

The mean is essentially zero — OLS guarantees this when an intercept is included. But the distribution has heavy tails (max residual ≈ $350,700), indicating the model struggles with the most expensive properties.

### Actual vs. Predicted Plot

A complementary diagnostic — the closer points are to the 45° diagonal, the better:

```python
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.2, s=5, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Value ($100,000s)")
plt.ylabel("Predicted Value ($100,000s)")
plt.title("Actual vs. Predicted")
plt.legend()
plt.tight_layout()
plt.show()
```

The horizontal band at `y_test = 5.0` (the dataset's cap) is visible — many high-value homes were recorded identically, and the model can't fit that spike accurately.

## The Four Assumptions of Linear Regression

Linear regression makes assumptions that, when violated, make the coefficients unreliable or the model invalid:

| Assumption | What it means | How to check | What to do if violated |
|-----------|---------------|-------------|----------------------|
| **Linearity** | The relationship between features and target is linear | Residuals vs. predicted: should be flat | Add polynomial terms, transform features |
| **Independence** | Observations are independent of each other | Domain knowledge | Use time-series models if data is sequential |
| **Homoscedasticity** | Constant variance of residuals | Residuals vs. predicted: no fan shape | Log-transform the target |
| **Normality** | Residuals are approximately normally distributed | Histogram of residuals | Larger samples make this less critical |

For the California housing data, linearity and homoscedasticity are the most likely violations — the data has non-linear relationships (geography, luxury properties) that a purely linear model can't capture. The next lessons address this.

## When to Use Linear Regression

Linear regression works best when:
- The relationship between features and target is approximately linear
- Interpretability matters (you need to explain every prediction)
- You have a clean, numeric dataset with limited non-linearity
- You're establishing a baseline before trying more complex models

It struggles when:
- The target has a very skewed distribution (log-transform helps)
- Feature relationships are non-linear or include interactions
- There is significant multicollinearity among features

## Conclusion

In this lesson, you learned how ordinary least squares finds coefficients by minimizing squared residuals, how to interpret those coefficients (with the caveat that they measure association, not causation), and how to diagnose model quality through residual analysis. The residual plots revealed that the California housing data has heteroscedasticity and non-linearity — patterns a purely linear model can't capture. In the next lesson, you'll add tools to handle non-linearity: polynomial features and decision tree regressors.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="After standardizing features, coefficients are in units of standard deviations. A coefficient of 0.829 means: moving income up by one standard deviation (≈$19,000) is associated with a 0.829 standard deviation increase in predicted value (≈$83,000), all else equal. The phrase &quot;all else equal&quot; is crucial — coefficients are partial effects, estimated while holding every other feature at its mean. Without standardization, the coefficient would be in the original units (e.g., $41,700 per $10,000 increase in income), making cross-feature comparison harder.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A linear regression model has `MedInc` coefficient = 0.829 (after standardizing all features). What does this mean?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>For every $1 increase in income, house value increases by $0.829.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>A 1 standard deviation increase in median income is associated with an 0.829 standard deviation increase in predicted house value, holding all other features constant.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Income accounts for 82.9% of the variance in house prices.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The model predicts a house value of 0.829 when all other features are zero.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="A fan shape in the residual vs. predicted plot is the classic signature of heteroscedasticity: variance increases with the magnitude of the predicted value. This violates a key assumption of OLS (constant residual variance) and can make predictions for high-value cases unreliable. Log-transforming the target (`y_log = np.log1p(y)`) compresses the distribution and often makes residual variance more constant. You then invert the transformation at prediction time: `y_pred = np.expm1(model.predict(X))`.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A residual vs. predicted plot shows a clear fan shape: residuals are small and tightly clustered for low predicted values, but grow much larger for high predicted values. What does this indicate and what is the recommended fix?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model is overfitting — reduce the number of features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Heteroscedasticity — the model's errors are not constant across the range of predictions. A common fix is to log-transform the target variable, which tends to compress the high-value tail and stabilize variance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The data has too many outliers — remove the top 5% of predictions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The model needs more training data to handle high-value predictions.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When two features are highly correlated, small changes in the data can cause large swings in their estimated coefficients — the model can&#039;t reliably attribute the effect to one feature vs. the other. The model&#039;s *predictions* can still be good (the collinear features together capture the signal), but *interpretations* of individual coefficients become unreliable. Dropping one correlated feature simplifies the model; Ridge regression (L2 regularization) is a more principled approach that keeps both features while constraining their coefficients — covered in lesson 4.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Two features, `AveRooms` and `AveBedrms`, have a correlation of 0.85 in the training data. What problem does this create for linear regression, and what are two ways to address it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The model cannot be trained — scikit-learn will raise an error when features are correlated.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Multicollinearity — highly correlated features make individual coefficient estimates unreliable, though the model's overall predictions may still be accurate. Two fixes: (1) drop one of the correlated features, or (2) use Ridge regression (L2 regularization), which handles multicollinearity by shrinking and stabilizing coefficients.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Multicollinearity means both features are redundant and should always be removed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Correlated features cause overfitting and require more training data to resolve.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

