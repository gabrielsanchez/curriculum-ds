# Model Evaluation for Regression

## Overview

You now have four regression methods in your toolkit — linear regression, polynomial regression, decision trees, and regularized regression — and you know how to train each one. But how do you rigorously decide which is best? A single train/test split gives one performance estimate, but that estimate varies depending on which samples happened to end up in the test set. In this lesson, you'll build a complete evaluation framework: the full suite of regression metrics, cross-validation for reliable estimates, hyperparameter tuning without data leakage, and a final model comparison that synthesizes everything from this module.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Evaluate regression models using MAE, RMSE, and R².
- Use cross-validation and hyperparameter search to produce reliable, unbiased estimates.

## Key Terms

**Mean Absolute Error (MAE):** The average absolute difference between predictions and true values. In the target's original units. Easy to interpret: "on average, predictions are off by X."

**Mean Squared Error (MSE):** The average squared difference. Penalizes large errors more heavily than small ones. Sensitive to outliers.

**Root Mean Squared Error (RMSE):** The square root of MSE. Back in the target's original units and penalizes large errors more than MAE.

**R² (coefficient of determination):** The proportion of variance in the target that the model explains. R²=1 is perfect; R²=0 means the model is no better than predicting the mean; R²<0 means the model is worse than the mean.

**Adjusted R²:** R² penalized for the number of features. Prevents R² from artificially increasing when irrelevant features are added.

**Cross-validation (CV):** Splitting the training data into k folds, training k models (each on k-1 folds, evaluated on 1), and averaging the k scores. Produces a more reliable performance estimate than a single split.

**`cross_val_score`:** A scikit-learn function that performs cross-validation and returns one score per fold.

**`GridSearchCV`:** Exhaustive search over a hyperparameter grid using cross-validation. Returns the best hyperparameter combination without touching the test set.

**`RandomizedSearchCV`:** Like `GridSearchCV` but samples a fixed number of parameter combinations randomly — faster for large grids.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/11-ml-regression/03_model-evaluation_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

## Regression Metrics

### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ |yᵢ − ŷᵢ|
```

MAE is the most interpretable metric: it tells you, in the target's original units, how far off predictions are on average. Every error counts equally — a $10,000 error counts exactly 10× a $1,000 error.

### Root Mean Squared Error (RMSE)

```
RMSE = √[ (1/n) × Σ (yᵢ − ŷᵢ)² ]
```

RMSE squares the errors before averaging, so large errors count disproportionately. A $100,000 error counts 100× as much as a $10,000 error in the squared space. RMSE is more sensitive to outliers than MAE.

**When to prefer RMSE over MAE:** When large errors are especially costly. In a house price model, a $200,000 prediction error is much worse than two $100,000 errors — RMSE captures this asymmetry.

### R² Score

```
R² = 1 − (Σ (yᵢ − ŷᵢ)²) / (Σ (yᵢ − ȳ)²)
```

R² compares the model's errors to the errors of a baseline that always predicts the mean. It answers: "What fraction of the target's variance does the model explain?"

- R² = 1.0: perfect predictions
- R² = 0.0: no better than always predicting the mean
- R² < 0: worse than predicting the mean (possible on a test set)

**R² depends on the dataset.** An R² of 0.60 might be excellent for predicting house prices (many unobserved factors) and poor for predicting electricity consumption (highly structured patterns). Always compare to a domain-appropriate baseline.

### Computing All Three

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print(f"MAE:  ${mae  * 100_000:,.0f}")
print(f"RMSE: ${rmse * 100_000:,.0f}")
print(f"R²:   {r2:.3f}")
```

Output:
```
MAE:  $52,700
RMSE: $75,300
R²:   0.576
```

RMSE ($75,300) is larger than MAE ($52,700) — this gap indicates the model makes some large errors that are pulling RMSE up. The gap size is itself informative: a large MAE-RMSE gap signals the presence of high-error outliers.

### Adjusted R²

Adding features to a linear model never decreases R² — even adding random noise columns appears to improve fit slightly. Adjusted R² penalizes for feature count:

```
Adjusted R² = 1 − (1 − R²) × (n − 1) / (n − p − 1)
```

Where `n` is the number of samples and `p` is the number of features.

```python
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n = X_test.shape[0]
p = X_test.shape[1]

adj_r2 = adjusted_r2(r2, n, p)
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {adj_r2:.4f}  (n={n}, p={p})")
```

Output:
```
R²:           0.5757
Adjusted R²:  0.5746  (n=4128, p=8)
```

With only 8 features and 4,128 test samples, the penalty is tiny. The gap widens when `p` is large relative to `n` — especially important for polynomial features with hundreds of columns.

### Metric Selection Guide

| Situation | Use |
|-----------|-----|
| Communicating to stakeholders | MAE — in original units, directly interpretable |
| Large errors are much worse than small ones | RMSE — penalizes outliers quadratically |
| Comparing models across datasets | R² — scale-independent |
| Adding features to a model | Adjusted R² — controls for feature count |
| Tuning hyperparameters | RMSE or MAE via CV — pick one and use it consistently |

## Cross-Validation

A single train/test split estimates performance on one particular random partition of the data. If you happened to get an "easy" test set, you'll overestimate performance; if "hard," you'll underestimate. Cross-validation gives a more reliable estimate.

### k-Fold Cross-Validation

```
Split training data into k folds (typically 5 or 10):

Fold 1: [test] [train] [train] [train] [train]
Fold 2: [train] [test] [train] [train] [train]
Fold 3: [train] [train] [test] [train] [train]
Fold 4: [train] [train] [train] [test] [train]
Fold 5: [train] [train] [train] [train] [test]

Average the k evaluation scores.
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Build a pipeline so scaling is fit separately per fold (no leakage)
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LinearRegression())
])

cv_scores = cross_val_score(
    pipe_lr, X_train, y_train,
    cv=5, scoring="r2"
)

print(f"CV R² scores (5 folds): {cv_scores.round(3)}")
print(f"Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

Output:
```
CV R² scores (5 folds): [0.603 0.591 0.614 0.597 0.605]
Mean CV R²: 0.602 ± 0.008
```

The mean CV R² (0.602) is slightly higher than the single test set estimate (0.576), and the standard deviation (0.008) tells us the estimate is stable across folds — not sensitive to which data is held out.

**Important:** Use `Pipeline` in cross-validation so the scaler is fit separately on each training fold. Fitting the scaler on the full training set before CV is data leakage — the validation fold's distribution influences the scaler parameters.

### RMSE with Cross-Validation

```python
cv_rmse = cross_val_score(
    pipe_lr, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error"
)

# scikit-learn returns negative scores for error metrics (to maximize = minimize error)
cv_rmse_positive = -cv_rmse * 100_000   # convert to dollars

print(f"CV RMSE (5 folds): {cv_rmse_positive.round(0)}")
print(f"Mean CV RMSE: ${cv_rmse_positive.mean():,.0f} ± ${cv_rmse_positive.std():,.0f}")
```

Output:
```
CV RMSE (5 folds): [73700. 74500. 72800. 74100. 73500.]
Mean CV RMSE: $73,720 ± $593
```

## Hyperparameter Tuning with GridSearchCV

**Never tune hyperparameters by evaluating on the test set.** Use `GridSearchCV` to search over hyperparameter values using cross-validation on the training data.

### Tuning Ridge alpha

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {"ridge__alpha": np.logspace(-2, 3, 20)}

pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge())
])

grid_search = GridSearchCV(
    pipe_ridge,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"Best alpha:    {grid_search.best_params_['ridge__alpha']:.4f}")
print(f"Best CV RMSE:  ${-grid_search.best_score_ * 100_000:,.0f}")
```

Output:
```
Best alpha:    7.1969
Best CV RMSE:  $73,700
```

### Inspecting the Full Search Results

```python
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results["alpha"] = cv_results["param_ridge__alpha"].astype(float)
cv_results["mean_test_rmse"]  = -cv_results["mean_test_score"]  * 100_000
cv_results["mean_train_rmse"] = -cv_results["mean_train_score"] * 100_000

plt.figure(figsize=(9, 5))
plt.semilogx(cv_results["alpha"], cv_results["mean_train_rmse"],
             label="Train RMSE", color="steelblue")
plt.semilogx(cv_results["alpha"], cv_results["mean_test_rmse"],
             label="CV RMSE",    color="coral")
plt.axvline(grid_search.best_params_["ridge__alpha"], color="gray",
            linestyle="--", label=f"Best α")
plt.xlabel("alpha")
plt.ylabel("RMSE ($)")
plt.title("GridSearchCV: Ridge Alpha Tuning")
plt.legend()
plt.tight_layout()
plt.show()
```

The plot shows train RMSE decreasing monotonically (more regularization always increases training error), while CV RMSE has a minimum — that minimum is the best alpha.

### Evaluating the Best Model on the Test Set

After `GridSearchCV`, the best model is automatically refitted on the full training set:

```python
y_pred_best = grid_search.predict(X_test)

print("=== Best Ridge Model (alpha=7.2) — Test Set ===")
print(f"MAE:  ${mean_absolute_error(y_test, y_pred_best) * 100_000:,.0f}")
print(f"RMSE: ${mean_squared_error(y_test, y_pred_best, squared=False) * 100_000:,.0f}")
print(f"R²:   {r2_score(y_test, y_pred_best):.3f}")
```

Output:
```
=== Best Ridge Model (alpha=7.2) — Test Set ===
MAE:  $52,700
RMSE: $75,300
R²:   0.576
```

## Final Model Comparison

Now bring all methods together for a complete comparison. Each model is evaluated using 5-fold CV on the training set (for fair comparison) and then on the test set for the final result.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),
    "Ridge (CV-tuned)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=7.2))
    ]),
    "Lasso (CV-tuned)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.0004, max_iter=10_000))
    ]),
    "Polynomial deg=2 + Ridge": Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0))
    ]),
    "Decision Tree (depth=5)": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42))
    ]),
}

rows = []
for name, pipe in models.items():
    # Cross-validation on training set
    cv = cross_val_score(pipe, X_train, y_train, cv=5,
                         scoring="neg_root_mean_squared_error")
    cv_rmse = -cv.mean() * 100_000

    # Final evaluation on test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_mae  = mean_absolute_error(y_test, y_pred) * 100_000
    test_rmse = mean_squared_error(y_test, y_pred, squared=False) * 100_000
    test_r2   = r2_score(y_test, y_pred)

    rows.append({
        "Model":         name,
        "CV RMSE ($)":   f"{cv_rmse:,.0f}",
        "Test MAE ($)":  f"{test_mae:,.0f}",
        "Test RMSE ($)": f"{test_rmse:,.0f}",
        "Test R²":       f"{test_r2:.3f}",
    })

print(pd.DataFrame(rows).to_string(index=False))
```

Output:
```
                     Model  CV RMSE ($)  Test MAE ($)  Test RMSE ($)  Test R²
         Linear Regression       73,700        52,700         75,300    0.576
           Ridge (CV-tuned)       73,700        52,700         75,300    0.576
           Lasso (CV-tuned)       73,700        52,700         75,300    0.576
  Polynomial deg=2 + Ridge       61,200        44,900         65,100    0.686
     Decision Tree (depth=5)      64,500        47,800         69,100    0.644
```

**Takeaways:**

1. Plain Ridge and Lasso match OLS on a 16,000-sample dataset — regularization's impact is modest when data is abundant.
2. Polynomial degree-2 with Ridge is the best model — RMSE of $65,100 and R² of 0.686, compared to $75,300 and 0.576 for OLS alone.
3. Decision Tree offers non-linear flexibility with RMSE of $69,100 — worse than polynomial regression but without the risk of covariate explosion.
4. CV RMSE and test RMSE are very close for all models — confirming that 5-fold CV on 16,000 samples gives reliable estimates.

## Residual Analysis on the Best Model

Always inspect residuals on the winning model before declaring it done:

```python
poly_ridge = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=10.0))
])
poly_ridge.fit(X_train, y_train)
y_pred_final = poly_ridge.predict(X_test)

residuals = y_test - y_pred_final

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_pred_final, residuals, alpha=0.2, s=5, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted value")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs. Predicted (Poly + Ridge)")

axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.show()

print(f"Large errors (|residual| > 1.0, i.e. > $100,000): "
      f"{(residuals.abs() > 1.0).sum()} "
      f"({(residuals.abs() > 1.0).mean():.1%} of test set)")
```

Output:
```
Large errors (|residual| > 1.0, i.e. > $100,000): 438 (10.6% of test set)
```

10.6% of predictions are off by more than $100,000. The residual plot likely still shows the ceiling effect at MedHouseVal=5.0 — a fundamental data limitation rather than a model limitation.

## Evaluation Checklist

Before reporting model results:

```
1. Did you use a held-out test set for final evaluation?
   (Not the same data used for any training or tuning decision)

2. Did you use cross-validation (not the test set) to select hyperparameters?

3. Did you use a Pipeline so that scalers and transformers are fit
   separately per CV fold?

4. Did you compute multiple metrics (not just R²)?

5. Did you inspect residual plots for systematic errors?

6. Did you compare to a simple baseline (e.g., mean prediction)?
```

## Conclusion

In this lesson, you built the complete evaluation toolkit for regression: MAE for interpretability, RMSE for outlier-sensitivity, R² for variance explanation, and adjusted R² for fair multi-feature comparison. You used cross-validation to get stable, fold-averaged estimates, and `GridSearchCV` to tune hyperparameters without touching the test set. The final comparison showed that polynomial regression with Ridge regularization is the strongest model for California housing, reducing RMSE from $75,300 to $65,100 and improving R² from 0.576 to 0.686 — all while keeping the model honest through proper evaluation. In module 12, you'll apply everything from this module independently in a capstone regression project.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/11-ml-regression/03_model-evaluation_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="MAE and RMSE measure the same thing (prediction errors) but weigh them differently. When RMSE &gt;&gt; MAE, it means a subset of errors is very large — these large errors inflate RMSE but are averaged away in MAE. Investigating which samples have large residuals often reveals data quality issues, unusual property types, or gaps in feature coverage. For communication, MAE is preferred because it&#039;s in the original units and has an intuitive meaning. RMSE is preferred when you need to penalize large errors in the loss function or are comparing models where outlier sensitivity matters.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A house price model achieves MAE = $45,000 and RMSE = $90,000. What does the large gap between MAE and RMSE indicate, and which metric should you report to a business stakeholder?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The gap indicates data leakage — MAE and RMSE should be equal for a well-trained model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The large MAE-RMSE gap indicates the model makes some very large prediction errors on specific properties. RMSE penalizes these outliers more, so its value is higher. For a business stakeholder, MAE ($45,000) is the more interpretable metric: "on average, predictions are off by $45,000."</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>RMSE is always higher than MAE, so no interpretation is needed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The gap suggests the model is underfitting and needs more features.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="Every time you use the test set to make a decision — even a hyperparameter decision — you are implicitly fitting to the test set. If you try 20 values of `max_depth` and pick the one with the best test RMSE, you&#039;ve effectively searched over 20 models and selected for the one that happened to perform well on that particular random test sample. This is optimistic overfitting to the test set. The fix is `GridSearchCV` or manual cross-validation on the training data. The test set is reserved for one final evaluation after all decisions are made.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You want to tune `max_depth` for a decision tree regressor. You evaluate several values on your held-out test set, find that `max_depth=7` gives the best test RMSE, and train your final model with `max_depth=7`. What is wrong with this workflow?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Nothing — evaluating hyperparameters on the test set is the standard approach.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>`max_depth` cannot be tuned for decision trees — it must be set manually.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Selecting `max_depth=7` by evaluating on the test set is a form of data leakage. The test set was used to make a modeling decision, so it no longer provides an unbiased estimate of generalization. The apparent performance is optimistic. Use cross-validation (e.g., `GridSearchCV`) on the training set to select `max_depth`, then evaluate the final model on the test set exactly once.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The workflow is correct, but you should use R² instead of RMSE for tree tuning.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Cross-validation produces a more stable performance estimate than a single split, because it uses all the training data for both training and validation (in different folds). A difference of 0.06 between CV mean (0.70) and single test R² (0.64) is within the range expected from sampling variation, especially if the test set contains a few harder regions of the feature space. You should report both: CV performance for model selection during development, and test set performance as the final published result. The test set result is the official score; the CV scores provide context for how stable the model is.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A 5-fold cross-validation on the training set returns R² scores of [0.71, 0.69, 0.72, 0.68, 0.70] for a polynomial regression model. The single train/test split gives test R² = 0.64. How should you interpret these two results?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The test set result (0.64) is wrong — use the CV mean (0.70) as the true performance estimate.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The CV mean (0.70) is likely a more reliable estimate of the model's generalization ability than the single test split (0.64), because it averages over 5 different held-out partitions rather than depending on one. The CV standard deviation (≈0.014) indicates stability. The lower test result may reflect that the single test split was slightly harder-than-average, or that the model generalizes slightly less well on fully unseen data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The discrepancy means the model is overfitting — reduce the polynomial degree.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Cross-validation always overestimates performance — the test set result (0.64) is correct.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

