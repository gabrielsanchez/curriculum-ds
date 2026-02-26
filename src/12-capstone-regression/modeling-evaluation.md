# Modeling & Evaluation

## Overview

Your dataset is cleaned, your features are engineered, and your train/test split is in place. In this lesson, you'll complete the second half of your capstone notebook: training at least three regression models, comparing them fairly, tuning the best one with cross-validation, and interpreting your results with residual analysis and a written conclusion.

## Learning Objectives

By the end of this lesson, you will have:

- Trained and evaluated multiple regression models on the same held-out test set.
- Tuned hyperparameters using cross-validation without data leakage.
- Communicated results clearly with a comparison table and written interpretation.

## Starter Code

Continue in your capstone notebook from the previous lesson. All variables from Phase 5 (the processed feature matrix and target arrays) should be in memory.

---

## Phase 6: Training Models

You must train at least **three** regression models. Choose from this list — you do not need to use all of them:

| Model | Import | Notes |
|-------|--------|-------|
| Linear Regression | `from sklearn.linear_model import LinearRegression` | Always train this as the baseline |
| Ridge Regression | `from sklearn.linear_model import Ridge` | Good default regularized linear model |
| Lasso Regression | `from sklearn.linear_model import Lasso` | Use if you suspect many features are irrelevant |
| Polynomial + Ridge | `PolynomialFeatures` + `Ridge` in a Pipeline | Good for non-linear patterns with many features |
| Decision Tree | `from sklearn.tree import DecisionTreeRegressor` | Non-parametric; tune `max_depth` |
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | Powerful; covered more deeply in module 13 |
| Gradient Boosting | `from sklearn.ensemble import GradientBoostingRegressor` | Often the best single model; slower to train |

**Required:** Linear Regression must be one of your three. It serves as the baseline that all other models must beat.

### Training Template

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# ── Helper function ───────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te, log_target=False):
    """Train a model and return evaluation metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # If target was log-transformed, invert for interpretable metrics
    if log_target:
        y_pred = np.expm1(y_pred)
        y_te   = np.expm1(y_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2   = r2_score(y_te, y_pred)

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "_model": model}

# ── Train models ─────────────────────────────────────────────────────────────
results = []

results.append(evaluate(
    "Linear Regression",
    LinearRegression(),
    X_train_processed, y_train,
    X_test_processed,  y_test
))

results.append(evaluate(
    "Ridge (alpha=1.0)",
    Ridge(alpha=1.0),
    X_train_processed, y_train,
    X_test_processed,  y_test
))

results.append(evaluate(
    "Decision Tree (depth=5)",
    DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
    X_train_processed, y_train,
    X_test_processed,  y_test
))

# Add more models here as needed

# ── Comparison table ──────────────────────────────────────────────────────────
summary = pd.DataFrame([{k: v for k, v in r.items() if k != "_model"} for r in results])
summary["MAE"]  = summary["MAE"].map("{:.2f}".format)
summary["RMSE"] = summary["RMSE"].map("{:.2f}".format)
summary["R²"]   = summary["R²"].map("{:.3f}".format)

print(summary.to_string(index=False))
```

**After printing the comparison table, write one paragraph interpreting the results.** Which model performs best? Is the improvement over the linear baseline meaningful? Are any models surprisingly close to each other?

### Guiding Questions While Training

- Does the decision tree's train R² greatly exceed its test R²? (Overfitting signal — lower `max_depth`)
- Does any model perform *worse* than simply predicting the mean? (R² < 0 — check for a bug)
- Are all models evaluated on the **same** test set? (They must be)

---

## Phase 7: Cross-Validation

A single train/test split gives one performance estimate. Cross-validation gives a more reliable picture by averaging over multiple held-out partitions.

### 5-Fold Cross-Validation for Each Model

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Note: if you used a preprocessor, wrap it with the model in a Pipeline
# so scaling is re-fit separately for each fold

def cv_score(name, pipeline, X, y, cv=5, log_target=False):
    """Return mean ± std of cross-validated RMSE."""
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -scores
    if log_target:
        # RMSE in log space is not directly interpretable — report as-is and note this
        pass
    print(f"{name:35s}  CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
    return rmse_scores.mean()

# Example (adapt to your preprocessor structure)
cv_score("Linear Regression",  LinearRegression(),      X_train, y_train)
cv_score("Ridge (alpha=1.0)",  Ridge(alpha=1.0),        X_train, y_train)
cv_score("Decision Tree (d=5)",
         DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
         X_train, y_train)
```

**Write:** Do the CV scores agree with the single test set scores? If the CV mean is substantially higher or lower than the test set score, what might explain the discrepancy?

---

## Phase 8: Hyperparameter Tuning

Tune **at least one** model's hyperparameters using `GridSearchCV`. The hyperparameter must be selected using cross-validation on the training set — not by evaluating on the test set.

### GridSearchCV Template

```python
from sklearn.model_selection import GridSearchCV

# ── Example: tuning Ridge alpha ───────────────────────────────────────────────
param_grid = {"alpha": np.logspace(-3, 3, 20)}

grid_ridge = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)
grid_ridge.fit(X_train_processed, y_train)

print(f"Best alpha:   {grid_ridge.best_params_['alpha']:.4f}")
print(f"Best CV RMSE: {-grid_ridge.best_score_:.4f}")

# ── Example: tuning Decision Tree max_depth ───────────────────────────────────
param_grid_dt = {
    "max_depth":       [3, 4, 5, 6, 7, 8, 10],
    "min_samples_leaf":[10, 20, 40, 60]
}

grid_dt = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid_dt,
    cv=5,
    scoring="neg_root_mean_squared_error"
)
grid_dt.fit(X_train_processed, y_train)

print(f"\nBest params:  {grid_dt.best_params_}")
print(f"Best CV RMSE: {-grid_dt.best_score_:.4f}")
```

**After finding the best hyperparameters, retrain the tuned model and add it to your comparison table:**

```python
# GridSearchCV automatically retrains on the full training set with the best params
y_pred_tuned = grid_ridge.predict(X_test_processed)

tuned_result = {
    "Model": "Ridge (CV-tuned)",
    "MAE":   mean_absolute_error(y_test, y_pred_tuned),
    "RMSE":  mean_squared_error(y_test, y_pred_tuned, squared=False),
    "R²":    r2_score(y_test, y_pred_tuned)
}
print(tuned_result)
```

**Write:** How much did tuning improve performance over the default hyperparameter? Was the improvement substantial or marginal? What does this tell you about the sensitivity of this algorithm to its hyperparameters on your dataset?

---

## Phase 9: Final Model Selection

Based on your comparison table and cross-validation scores, select your best model. Document the decision explicitly in your notebook.

A good justification addresses:
1. **Performance:** Which model has the best test RMSE and R²?
2. **Stability:** Is the CV standard deviation low? (Low std = consistent performance)
3. **Interpretability:** Does your use case require explaining predictions?
4. **Simplicity:** If two models perform similarly, prefer the simpler one

```python
# ── Define your final model ────────────────────────────────────────────────────
final_model = grid_ridge   # replace with your chosen model

# Final evaluation on the held-out test set
y_pred_final = final_model.predict(X_test_processed)

final_mae  = mean_absolute_error(y_test, y_pred_final)
final_rmse = mean_squared_error(y_test, y_pred_final, squared=False)
final_r2   = r2_score(y_test, y_pred_final)

print("=== Final Model Results ===")
print(f"Model:  {type(final_model.best_estimator_).__name__}")
print(f"MAE:    {final_mae:.4f}   (in target units)")
print(f"RMSE:   {final_rmse:.4f}  (in target units)")
print(f"R²:     {final_r2:.3f}")
```

---

## Phase 10: Residual Analysis

Always inspect residuals on your final model. These plots often reveal problems that metrics alone don't surface.

```python
residuals = y_test.values - y_pred_final

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Residuals vs. Predicted
axes[0].scatter(y_pred_final, residuals, alpha=0.3, s=8, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Predicted value")
axes[0].set_ylabel("Residual (actual − predicted)")
axes[0].set_title("Residuals vs. Predicted")

# Plot 2: Histogram of residuals
axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_title("Residual Distribution")

# Plot 3: Actual vs. Predicted
axes[2].scatter(y_test, y_pred_final, alpha=0.3, s=8, color="steelblue")
lo = min(y_test.min(), y_pred_final.min())
hi = max(y_test.max(), y_pred_final.max())
axes[2].plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
axes[2].set_xlabel("Actual value")
axes[2].set_ylabel("Predicted value")
axes[2].set_title("Actual vs. Predicted")

plt.suptitle(f"Residual Analysis — {type(final_model).__name__}", y=1.02)
plt.tight_layout()
plt.show()

# Summary statistics
print(f"Residual mean:  {residuals.mean():.4f}  (should be ≈ 0)")
print(f"Residual std:   {residuals.std():.4f}")
print(f"% large errors (|r| > 1 std): {(np.abs(residuals) > residuals.std()).mean():.1%}")
```

**After each plot, write:**

1. **Residuals vs. Predicted:** Is there a fan shape (heteroscedasticity)? A curve (non-linearity the model missed)? Clusters of large errors at specific predicted values?
2. **Residual histogram:** Are residuals approximately normally distributed? Are there heavy tails? What do outlier residuals correspond to (if you can identify the rows)?
3. **Actual vs. Predicted:** How tight is the scatter around the diagonal? Are there ceiling/floor effects? Which region of the target range does the model predict best and worst?

---

## Phase 11: Conclusion

Write a **conclusion section** in your notebook (not a code cell — a Markdown cell) with these components:

### Best Model Summary

State your best model, its hyperparameters, and its performance metrics in plain English:

> "The best-performing model was a Ridge regression with alpha=7.2, achieving RMSE = 0.43 and R² = 0.72 on the held-out test set. On average, predictions are off by $43,000 — a meaningful but not extreme error for the $200,000–$500,000 price range represented in the dataset."

### Feature Importance (if available)

If your best model supports feature importances or coefficients, include them:

```python
# For linear/Ridge/Lasso models (after scaling):
if hasattr(final_model, "coef_"):
    coef_series = pd.Series(
        final_model.coef_,
        index=feature_names   # use get_feature_names_out() from your preprocessor
    ).sort_values(key=abs, ascending=False)
    print("Top 10 features by coefficient magnitude:")
    print(coef_series.head(10).round(4))

# For tree-based models:
if hasattr(final_model, "feature_importances_"):
    imp = pd.Series(final_model.feature_importances_, index=feature_names)
    print("Top 10 features by importance:")
    print(imp.sort_values(ascending=False).head(10).round(4))
```

**Write:** Which features drove the model's predictions? Does this match your expectations from EDA? Are there any features that are more important than you expected?

### Limitations

Every honest analysis acknowledges what it cannot do. Address at least two of the following:

- Does the dataset capture all factors that influence the target? What is missing?
- Are there subgroups in the data where the model performs worse?
- Is the dataset representative of the population you'd deploy this model on?
- What would you need to improve the model further?

### What You Would Do Next

List 2–3 concrete next steps — specific model improvements, additional data sources, or engineering ideas that you believe would improve performance.

---

## Submission

### Notebook Checklist

Before submitting, review against the rubric from lesson 1. Specifically confirm:

- [ ] Problem statement is in a Markdown cell at the top of the notebook
- [ ] Every visualization has a written interpretation in a Markdown cell below it
- [ ] Every cleaning decision is documented with a rationale
- [ ] Every engineered feature has a justification
- [ ] Three models are trained and compared in a table
- [ ] At least one model is tuned with `GridSearchCV`
- [ ] Cross-validation is used for model comparison
- [ ] Residual analysis includes at least the residuals-vs-predicted plot
- [ ] Conclusion addresses best model, feature importance, and limitations
- [ ] Notebook runs top-to-bottom without errors (Restart & Run All)
- [ ] Notebook is committed and pushed to a public GitHub repository

### Three-Paragraph Summary

Write this as a standalone document (not in the notebook) and submit alongside the GitHub link:

**Paragraph 1:** What dataset did you use, what are you predicting, and why does this problem matter?

**Paragraph 2:** What did EDA reveal? Name two specific findings and explain how they influenced your cleaning or feature engineering decisions.

**Paragraph 3:** Which model performed best and by what metrics? What does the error mean in practical terms? What would you do differently or next?

---

## Final Comparison Table Template

Your notebook must contain a table in this format:

| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|-------|---------|---------|---------|---------|
| Linear Regression (baseline) | | | | |
| [Model 2] | | | | |
| [Model 3] | | | | |
| [Best model, tuned] | | | | |

Values should be in the target's original units (or clearly labeled as log-space if you log-transformed the target).
