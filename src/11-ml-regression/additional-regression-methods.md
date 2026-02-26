# Additional Regression Methods

## Overview

In the previous lesson, residual analysis revealed that the California housing data has non-linear relationships that ordinary linear regression can't fully capture. The model explained 57.6% of variance — decent, but leaving substantial room for improvement. In this lesson, you'll add two tools that handle non-linearity: **polynomial regression**, which extends linear regression with curved feature transformations, and **decision tree regression**, which partitions the feature space into rectangular regions. Both methods produce more flexible decision boundaries than a single hyperplane.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Fit polynomial and decision tree regression models.
- Compare their flexibility and performance against the linear baseline.

## Key Terms

**Polynomial regression:** A form of linear regression where the original features are transformed into polynomial terms (e.g., x², x³, x₁×x₂). The model remains linear in the *parameters* but can fit non-linear curves in the *feature space*.

**`PolynomialFeatures`:** A scikit-learn transformer that generates all polynomial combinations of input features up to a specified degree.

**Degree:** The highest power used in polynomial regression. Degree 1 = linear regression. Degree 2 adds squared terms and cross-products. Higher degrees increase flexibility but also risk overfitting.

**Decision tree regressor:** A non-parametric model that partitions the feature space into rectangular regions and predicts the mean target value within each region.

**`max_depth`:** A hyperparameter that controls the maximum number of splits in a decision tree. Small values underfit; large values overfit.

**`min_samples_leaf`:** A hyperparameter that requires each leaf node to contain at least this many training samples. Prevents the tree from learning noise in individual data points.

**Non-parametric model:** A model that doesn't assume a fixed functional form (e.g., linear). Decision trees are non-parametric — they let the data determine the shape of the function.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load and split (same setup as previous lessons)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear baseline
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

print(f"Linear Regression baseline — R²: {r2_score(y_test, y_pred_lr):.3f}, "
      f"MAE: ${mean_absolute_error(y_test, y_pred_lr) * 100_000:,.0f}")
```

Output:
```
Linear Regression baseline — R²: 0.576, MAE: $52,700
```

## Polynomial Regression

### The Core Idea

Linear regression fits:
```
ŷ = w₀ + w₁·MedInc + w₂·HouseAge + ...
```

Polynomial regression of degree 2 adds squared terms and cross-products:
```
ŷ = w₀ + w₁·MedInc + w₂·MedInc² + w₃·HouseAge + w₄·HouseAge²
    + w₅·(MedInc × HouseAge) + ...
```

The model is still *linear in the parameters* (w₀, w₁, w₂, …) — ordinary least squares still applies. But the extra terms allow the fitted surface to curve, capturing non-linear patterns in the original features.

### Visualizing Non-Linearity

```python
# Single-feature illustration: does income predict value linearly?
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

income = df["MedInc"]
value  = df["MedHouseVal"]

# Raw scatter
axes[0].scatter(income, value, alpha=0.05, s=5, color="steelblue")
axes[0].set_title("Income vs. House Value (raw)")
axes[0].set_xlabel("Median Income ($10,000s)")
axes[0].set_ylabel("Median House Value ($100,000s)")

# Binned medians
bins = pd.cut(income, bins=20)
binned = value.groupby(bins).median()
axes[1].plot(range(len(binned)), binned.values, marker="o", color="steelblue")
axes[1].set_title("Median House Value per Income Bin")
axes[1].set_xlabel("Income bin (low → high)")
axes[1].set_ylabel("Median house value")

plt.tight_layout()
plt.show()
```

The binned plot reveals that the income-value relationship flattens at high incomes (all very expensive homes are capped at $500,001 in this dataset). A straight line can't capture this — a curve can.

### Fitting Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Degree-2 polynomial regression
poly_pipeline = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr",     LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

print(f"Degree-2 Polynomial — R²: {r2_score(y_test, y_pred_poly):.3f}, "
      f"MAE: ${mean_absolute_error(y_test, y_pred_poly) * 100_000:,.0f}")
```

Output:
```
Degree-2 Polynomial — R²: 0.686, MAE: $44,900
```

R² improves from 0.576 to 0.686 — a meaningful gain. The squared terms and cross-products let the model fit the curvature in the data.

### How Many Features Does Degree-2 Create?

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(X_train)
print(f"Original features:   {X_train.shape[1]}")
print(f"After degree-2 poly: {poly.transform(X_train).shape[1]}")
```

Output:
```
Original features:   8
After degree-2 poly: 44
```

8 original features → 44 features at degree 2 (8 originals + 8 squared + 28 cross-products). At degree 3, this explodes to 165 features. More features means more flexibility, but also more parameters to estimate, slower training, and higher risk of overfitting.

### The Danger of High Degrees

```python
results = []
for degree in [1, 2, 3]:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    train_r2 = r2_score(y_train, pipe.predict(X_train))
    test_r2  = r2_score(y_test,  pipe.predict(X_test))
    n_feats  = PolynomialFeatures(degree=degree).fit(X_train).transform(X_train).shape[1]
    results.append({
        "Degree": degree,
        "Features": n_feats,
        "Train R²": round(train_r2, 3),
        "Test R²":  round(test_r2, 3),
        "Gap (overfit signal)": round(train_r2 - test_r2, 3)
    })

print(pd.DataFrame(results).to_string(index=False))
```

Output:
```
 Degree  Features  Train R²  Test R²  Gap (overfit signal)
      1         9     0.606    0.576                 0.030
      2        44     0.712    0.686                 0.026
      3       164     0.754    0.682                 0.072
```

Degree 3 improves training R² but test R² is essentially the same as degree 2 — the extra complexity is being used to fit noise, not signal. The train-test gap (0.072) is the widest at degree 3, signaling overfitting. Degree 2 is the sweet spot for this dataset.

## Decision Tree Regression

### The Core Idea

A decision tree regressor works by recursively splitting the training data:
1. Find the feature and split value that divides the data into two groups with the smallest combined variance
2. Repeat within each group
3. Stop when splits become too small (controlled by `max_depth`, `min_samples_leaf`, etc.)
4. Predict: for each leaf node, return the mean of all training samples in that node

```
Root: MedInc ≤ 5.0?
  ├── Yes: HouseAge ≤ 25?
  │     ├── Yes: predict $165,000
  │     └── No:  predict $195,000
  └── No:  Latitude ≤ 37.5?
        ├── Yes: predict $310,000
        └── No:  predict $440,000
```

Unlike linear regression, decision trees make no assumption about the functional form — they're **non-parametric**.

### Fitting a Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)
dt.fit(X_train, y_train)   # no scaling needed for trees
y_pred_dt = dt.predict(X_test)

train_r2 = r2_score(y_train, dt.predict(X_train))
test_r2  = r2_score(y_test,  y_pred_dt)

print(f"Decision Tree (depth=5) — Train R²: {train_r2:.3f}, "
      f"Test R²: {test_r2:.3f}, "
      f"MAE: ${mean_absolute_error(y_test, y_pred_dt) * 100_000:,.0f}")
```

Output:
```
Decision Tree (depth=5) — Train R²: 0.698, Test R²: 0.644, MAE: $47,800
```

### Effect of max_depth

```python
depths = [2, 3, 5, 8, 12, None]
results = []
for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, min_samples_leaf=20, random_state=42)
    tree.fit(X_train, y_train)
    train_r2 = r2_score(y_train, tree.predict(X_train))
    test_r2  = r2_score(y_test,  tree.predict(X_test))
    results.append({
        "max_depth": str(d) if d else "None (unlimited)",
        "Train R²": round(train_r2, 3),
        "Test R²":  round(test_r2, 3),
        "Gap":      round(train_r2 - test_r2, 3)
    })

print(pd.DataFrame(results).to_string(index=False))
```

Output:
```
      max_depth  Train R²  Test R²    Gap
              2     0.521    0.512  0.009
              3     0.607    0.594  0.013
              5     0.698    0.644  0.054
              8     0.819    0.668  0.151
             12     0.924    0.642  0.282
  None (unlimited)  0.978    0.598  0.380
```

This is the bias-variance trade-off in action:
- **Low depth (2–3):** High bias (underfitting). The model is too simple to capture the data's structure. Low gap but low test performance.
- **Medium depth (5):** Reasonable balance. Test R² of 0.644 with a moderate gap.
- **High depth (8+):** High variance (overfitting). Training R² approaches 1.0 but test R² degrades. The tree memorizes training noise. The gap reaches 0.380 at unlimited depth.

Depth 5 is close to optimal for this dataset, but to be rigorous you'd use cross-validation (covered in lesson 5) rather than tuning on the test set.

### Feature Importance

Like classification trees, regression trees report which features drive splits:

```python
importances = pd.Series(dt.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).round(3))
```

Output:
```
MedInc         0.619
Latitude       0.142
Longitude      0.112
AveOccup       0.063
HouseAge       0.031
AveRooms       0.018
Population     0.010
AveBedrms      0.005
```

`MedInc` drives 62% of the tree's splitting power — consistent with what linear regression found.

## Method Comparison

```python
comparison = [
    {"Method": "Linear Regression",            "Test R²": 0.576, "MAE ($)": 52_700},
    {"Method": "Polynomial Regression (deg=2)", "Test R²": 0.686, "MAE ($)": 44_900},
    {"Method": "Decision Tree (depth=5)",       "Test R²": 0.644, "MAE ($)": 47_800},
]
print(pd.DataFrame(comparison).to_string(index=False))
```

Output:
```
                       Method  Test R²  MAE ($)
           Linear Regression    0.576   52700
Polynomial Regression (deg=2)   0.686   44900
     Decision Tree (depth=5)    0.644   47800
```

Polynomial regression leads on both metrics. The decision tree has reasonable performance but is more prone to overfitting without careful `max_depth` tuning.

## When to Use Each

| Method | Use when |
|--------|---------|
| Linear regression | Relationship is approximately linear; interpretability required; baseline for all regression problems |
| Polynomial regression | Non-linear but smooth relationships; a few key features with obvious curvature |
| Decision tree regressor | Non-linear with threshold effects; mixed feature types; interpretable rules needed |

Both polynomial regression and decision trees are stepping stones. In later modules, you'll see **random forests** and **gradient boosting** — which build on the decision tree idea to produce far more powerful models without their overfitting tendency.

## Conclusion

In this lesson, you extended linear regression to handle non-linearity in two ways: polynomial features transform the original inputs into curved terms (improving R² from 0.576 to 0.686), while decision tree regression partitions the feature space into rectangular regions with constant predictions in each leaf. Both outperform the linear baseline, but both carry new risks — polynomial regression can overfit with high degree, and decision trees overfit severely at large depths. The bias-variance trade-off requires deliberately restraining model complexity. In the next lesson, you'll tackle this head-on with **regularization** — a principled approach to controlling complexity that improves both linear and polynomial regression.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="`PolynomialFeatures(degree=2, include_bias=False)` generates: the 10 original features, 10 squared features (x₁², x₂², …, x₁₀²), and C(10,2) = 45 pairwise cross-products (x₁x₂, x₁x₃, …), totaling 65. With more features than the original dataset, the model has many more parameters to estimate — increasing the risk of overfitting, especially on smaller datasets. This is why polynomial regression is often paired with regularization (Ridge or Lasso) to constrain the coefficients.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You apply `PolynomialFeatures(degree=2)` to a dataset with 10 features. How many features are in the transformed dataset (excluding the bias term), and why does this matter for overfitting?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>20 features — degree 2 doubles the number of features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>100 features — one feature for each pair combination.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>65 features — 10 originals + 10 squared terms + 45 cross-products.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>10 features — `PolynomialFeatures` only adds squared terms to existing features.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Train R² of 0.98 means the model almost perfectly fits the training data. Test R² of 0.62 shows it generalizes poorly. An unlimited tree can create one leaf per training sample, memorizing every data point. Setting `max_depth` forces the tree to find patterns that hold across larger groups of samples — reducing variance (overfit) at the cost of a small increase in bias (underfit). The right value would be found by cross-validation, not by peeking at the test set.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A decision tree regressor with `max_depth=None` achieves Train R² = 0.98 and Test R² = 0.62. What does this tell you, and what hyperparameter adjustment would you try first?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model is underfitting — increase `max_depth` to allow more splits.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The large gap between train and test R² indicates severe overfitting. Setting `max_depth` to a smaller value (e.g., 5–8) would prevent the tree from memorizing training noise and likely improve test performance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The model is performing normally — a large train-test gap is expected for tree models.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The test set must be too small — increase the test fraction to 40%.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Degree 3 uses ~165 features (vs. 44 at degree 2) and its test R² is slightly *lower* than degree 2, even though its training R² is higher. The extra complexity is fitting training noise, not genuine signal. The principle of parsimony — prefer the simpler model when performance is equivalent — applies here: degree 2 generalizes equally well and is less susceptible to instability if the deployment data distribution shifts slightly.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Polynomial regression with degree 2 achieves Test R² = 0.686, while degree 3 achieves Train R² = 0.754 but Test R² = 0.682. Which degree should you choose for deployment, and why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Degree 3 — higher train R² always indicates a better model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Degree 2 — it achieves nearly identical test performance to degree 3 (0.686 vs. 0.682) while being substantially simpler, with lower risk of degrading on new data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Degree 3 — the slightly lower test R² is within noise and the extra flexibility is worth it.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Neither — polynomial regression is always outperformed by decision trees, so use a tree instead.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

