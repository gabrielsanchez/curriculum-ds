# Module Assessment

## Overview

This assessment brings together everything you have learned in module 11: fitting ordinary least squares regression, diagnosing residuals, adding polynomial features, applying Ridge and Lasso regularization, and selecting a final model using cross-validation and systematic hyperparameter search. You will work with the **California Housing dataset** — the same dataset used throughout the module — and walk through a complete regression workflow from baseline to regularized model, culminating in a justified recommendation.

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Train and evaluate linear regression models using appropriate metrics.
- Apply polynomial features and regularization to improve model performance.
- Use cross-validation and GridSearchCV to tune hyperparameters without data leakage.
- Interpret residual plots and select the best model with justification.

## Starter Code

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/11-ml-regression/11-ml-regression-module-assessment_starter.ipynb).

---

## The Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(X.shape)
print(X.head())
```

Output:
```
(20640, 8)
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
...
```

**Columns:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

**Target:** MedHouseVal — median house value in $100,000s (e.g., a value of 2.5 means $250,000)

---

## Coding Assessment

Practice the concepts from this module using this [notebook](#). After completing all tasks, save your notebook to GitHub and [submit the link for grading](https://ai-grader-pql9.onrender.com/).

### Task 1: Baseline Model

Load the California Housing dataset, create an 80/20 train/test split with `random_state=42`, and train a `LinearRegression` model on all 8 features. Report MAE, RMSE, and R² on the test set.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
```

Record your results — you will use them as the baseline in Task 5.

---

### Task 2: Residual Analysis

Using the predictions from Task 1, generate two diagnostic plots and write a brief interpretation.

**Plot 1 — Residuals vs. Predicted Values:**

```python
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="red", linewidth=1.5)
plt.xlabel("Predicted Value ($100k)")
plt.ylabel("Residual")
plt.title("Residuals vs. Predicted Values")
plt.tight_layout()
plt.show()
```

**Plot 2 — Histogram of Residuals:**

```python
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=60, edgecolor="white")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.show()
```

After generating both plots, write 2 sentences in a markdown cell addressing:

- Do the residuals appear normally distributed and centered near zero?
- Is there any pattern in the residuals vs. predicted plot that suggests the linear model's assumptions may be violated?

---

### Task 3: Polynomial Features + Ridge

Build a `Pipeline` that combines `PolynomialFeatures(degree=2)` with a `StandardScaler` and `Ridge` regression. Tune the regularization strength using `RidgeCV`.

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 3, 20)

poly_ridge = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model",  RidgeCV(alphas=alphas, cv=5)),
])

poly_ridge.fit(X_train, y_train)

best_alpha = poly_ridge.named_steps["model"].alpha_
print(f"Best alpha selected by RidgeCV: {best_alpha:.4f}")

# Cross-validated RMSE on training data
cv_scores = cross_val_score(
    poly_ridge, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error"
)
cv_rmse = -cv_scores.mean()
print(f"CV RMSE (5-fold, train): {cv_rmse:.4f}")

# Test RMSE
y_pred_ridge = poly_ridge.predict(X_test)
test_rmse = root_mean_squared_error(y_test, y_pred_ridge)
print(f"Test RMSE: {test_rmse:.4f}")
```

Compare the test RMSE from this pipeline to your Task 1 baseline. Does adding polynomial features and regularization improve performance?

---

### Task 4: Lasso for Feature Selection

Fit a `Lasso` model with `alpha=0.01` (after scaling). Identify which of the 8 original features receive non-zero coefficients, and how many are zeroed out.

```python
lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Lasso(alpha=0.01, max_iter=5000)),
])

lasso_pipe.fit(X_train, y_train)

coefs = pd.Series(
    lasso_pipe.named_steps["model"].coef_,
    index=X.columns
)
print("Lasso coefficients:")
print(coefs)
print(f"\nFeatures zeroed out: {(coefs == 0).sum()}")
print(f"Features retained:   {(coefs != 0).sum()}")

y_pred_lasso = lasso_pipe.predict(X_test)
lasso_rmse   = root_mean_squared_error(y_test, y_pred_lasso)
lasso_r2     = r2_score(y_test, y_pred_lasso)
print(f"\nLasso Test RMSE: {lasso_rmse:.4f}")
print(f"Lasso Test R²:   {lasso_r2:.4f}")
```

*Note: With `alpha=0.01` on the California Housing data, Lasso typically retains most features. Increase alpha to observe more aggressive feature elimination.*

---

### Task 5: Model Comparison Table

Fill in the following table with your results from Tasks 1, 3, and 4. For cross-validated RMSE on the linear regression baseline, use `cross_val_score` with `scoring="neg_root_mean_squared_error"` and 5-fold CV.

| Model | CV RMSE | Test RMSE | Test R² |
|-------|---------|-----------|---------|
| Linear Regression | | | |
| Polynomial (deg=2) + Ridge | | | |
| Lasso (alpha=0.01) | | | |

*Which model achieves the lowest test RMSE? Does that same model also achieve the highest R²?*

---

### Task 6: Recommendation

In a markdown cell, write 2–3 sentences justifying which model you would deploy based on your results. Cite at least two specific metrics from your comparison table. Consider both predictive accuracy and the practical implications of model complexity.

---

## Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Baseline model | 15 | Correct train/test split; LinearRegression fitted and evaluated; MAE, RMSE, and R² reported |
| Task 2: Residual analysis | 20 | Both plots generated; markdown interpretation addresses normality and any fan-shaped or curved pattern |
| Task 3: Polynomial + Ridge | 20 | Pipeline constructed correctly; RidgeCV used to select alpha; CV RMSE and test RMSE reported and compared to baseline |
| Task 4: Lasso feature selection | 15 | Lasso fitted inside a scaling pipeline; zero-coefficient features identified and counted; test RMSE reported |
| Task 5: Comparison table | 15 | All three rows filled with correct values from prior tasks |
| Task 6: Recommendation | 15 | Recommendation cites specific metrics; reasoning addresses accuracy-complexity trade-off |
| **Total** | **100** | |

---

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="RMSE squares each error before averaging, which means a single prediction that is off by 4 units contributes 16 to the sum, while four predictions each off by 1 unit only contribute 4 total. When RMSE is substantially larger than MAE, it signals that the model is making occasional very large errors rather than consistently moderate ones. High-leverage outliers — predictions far from the training distribution — are common sources of this pattern in housing data, where a handful of extremely high-value properties can dominate RMSE.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Your model reports MAE = 0.48 and RMSE = 0.79 on the California Housing test set. The gap between these two metrics is relatively large. What does this suggest about the model's error distribution?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The model is performing poorly overall — both metrics should be below 0.5 for a good regression model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The model is making occasional very large errors. RMSE penalizes large errors more heavily by squaring them, so a large RMSE relative to MAE indicates a few high-magnitude mistakes rather than consistently moderate errors.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>MAE and RMSE should always be equal; a gap means one of the metrics was computed incorrectly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The model is underfitting — increasing model complexity will always bring both metrics closer together.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="If you evaluate hyperparameters (such as Ridge alpha) directly on the test set, you are implicitly using the test set to make modeling decisions. Every time you pick the alpha that minimizes test error, the test set influences your model — it is no longer a truly held-out evaluation. Cross-validation keeps the test set completely untouched during hyperparameter selection: the training data is split into folds, and each fold acts as a validation set in turn. Only after alpha is chosen via cross-validation should the test set be used — once — to report final performance.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You want to choose the best Ridge alpha from the set [0.01, 0.1, 1.0, 10.0, 100.0]. Why is it wrong to evaluate each alpha on the test set and pick the one with the lowest test RMSE?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It is not wrong — the test set is the most reliable source of performance information and should always be used for hyperparameter selection.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It is wrong because Ridge requires cross-validation internally and cannot be evaluated on a static test set.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It is wrong because selecting the alpha with the lowest test RMSE uses the test set to make a modeling decision, which constitutes data leakage. The test set should only be used once — for the final evaluation of the chosen model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It is wrong only if the test set has fewer than 1,000 samples; larger test sets can be used for hyperparameter tuning without bias.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="As polynomial degree increases, the model gains enough flexibility to fit the training data nearly perfectly — training error approaches zero. However, high-degree polynomials also fit the noise specific to the training set, and when evaluated on unseen data the test error rises sharply. This is the bias-variance tradeoff: low-degree polynomials have high bias (underfitting); high-degree polynomials have high variance (overfitting). Test error is minimized at an intermediate degree, after which adding more polynomial terms hurts generalization even though it continues to reduce training error.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You train polynomial regression models of degree 1, 2, 3, 4, and 5 on the California Housing training data. Training RMSE decreases consistently as degree increases. What happens to test RMSE?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Test RMSE also decreases consistently — more complex models always generalize better when trained on enough data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Test RMSE decreases initially then increases. Higher-degree polynomials fit the training data more closely but eventually overfit, capturing noise rather than the true signal, which causes test error to rise.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Test RMSE stays flat — polynomial degree only affects training speed, not predictive accuracy.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Test RMSE increases monotonically from degree 1 onward — polynomial features always hurt generalization compared to the linear baseline.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>
