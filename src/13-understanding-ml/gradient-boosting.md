# Gradient Boosting

## Overview

In the previous lesson, random forests achieved strong performance by averaging many independent trees. Gradient Boosting builds on a different insight: instead of averaging independent trees, train each new tree specifically to correct the mistakes of the current ensemble. This sequential error-correction strategy produces models that often achieve the highest accuracy of any algorithm on structured tabular data — which is why gradient boosting variants (XGBoost, LightGBM, CatBoost) dominate data science competitions.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Understand the gradient boosting algorithm and how it differs from bagging.
- Train and tune `GradientBoostingClassifier` and understand the role of learning rate and number of estimators.

## Key Terms

**Gradient Boosting:** An ensemble technique that builds trees sequentially. Each new tree fits the residual errors of the existing ensemble, gradually reducing prediction error with each step.

**Boosting:** The general technique of training models sequentially, where each model focuses on the errors of its predecessor. Boosting reduces both bias and variance.

**Weak learner:** A model that is only slightly better than random guessing. Gradient boosting uses shallow decision trees (often depth 1–3) as weak learners. The ensemble of many weak learners becomes strong.

**Residuals (pseudo-residuals):** The difference between the current ensemble's prediction and the true target. Each new tree is trained to predict these residuals.

**Learning rate (shrinkage):** A parameter between 0 and 1 that scales each tree's contribution. Small learning rate requires more trees (slower, but often more accurate). Large learning rate requires fewer trees (faster, but can overfit).

**n_estimators:** The number of boosting rounds (trees). In contrast to random forests, too many trees can cause gradient boosting to overfit — the right value is determined by the learning rate.

**XGBoost / LightGBM:** Production-grade implementations of gradient boosting that are faster, more memory-efficient, and often more accurate than scikit-learn's `GradientBoostingClassifier`. Industry standard for tabular data.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## How Gradient Boosting Works

Random Forest:
```
Tree 1 (independent) ─┐
Tree 2 (independent) ─┤─→ AVERAGE → prediction
Tree 3 (independent) ─┘
```

Gradient Boosting:
```
Initial prediction F₀ (e.g., class mean)
    ↓
Residuals₁ = true_labels − F₀
    ↓
Tree₁ fits Residuals₁  →  F₁ = F₀ + learning_rate × Tree₁
    ↓
Residuals₂ = true_labels − F₁
    ↓
Tree₂ fits Residuals₂  →  F₂ = F₁ + learning_rate × Tree₂
    ↓
    ... (repeat for n_estimators trees)
    ↓
Final prediction = Fₙ
```

Each tree doesn't predict the original labels — it predicts the **current errors** of the ensemble. Adding many such corrections incrementally improves the model.

Why shallow trees? A depth-1 tree (decision stump) can only make one split — it's a very weak learner. But combining 500 stumps, each correcting the previous ensemble's errors, builds a powerful model without overfitting as severely as a single deep tree.

## Basic Gradient Boosting

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Note: gradient boosting is tree-based — scaling is not required.
# Use unscaled features.

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred  = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

print(f"Gradient Boosting (default):")
print(f"  Train accuracy: {gb.score(X_train, y_train):.3f}")
print(f"  Test accuracy:  {gb.score(X_test, y_test):.3f}")
print(f"  Test AUC:       {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
Gradient Boosting (default):
  Train accuracy: 1.000
  Test accuracy:  0.974
  Test AUC:       0.998
```

With default settings, gradient boosting achieves 97.4% accuracy and AUC 0.998 — matching kNN and outperforming random forests (95.6%) on this dataset.

## The Learning Rate / n_estimators Trade-off

The most important interaction in gradient boosting: **learning rate and n_estimators are inversely linked.**

```python
configs = [
    (0.5,  50,  "High lr, few trees"),
    (0.1, 100,  "Default"),
    (0.1, 300,  "Default lr, many trees"),
    (0.01, 500, "Low lr, many trees"),
    (0.01, 100, "Low lr, too few trees"),
]

print(f"{'Configuration':<30} {'Train':>8} {'Test':>8} {'AUC':>8}")
print("-" * 58)

for lr, n, label in configs:
    gb = GradientBoostingClassifier(
        n_estimators=n, learning_rate=lr, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    y_proba = gb.predict_proba(X_test)[:, 1]
    train_acc = gb.score(X_train, y_train)
    test_acc  = gb.score(X_test, y_test)
    auc       = roc_auc_score(y_test, y_proba)
    print(f"{label:<30} {train_acc:>8.3f} {test_acc:>8.3f} {auc:>8.3f}")
```

Output:
```
Configuration                  Train     Test      AUC
----------------------------------------------------------
High lr, few trees             0.996    0.965    0.995
Default                        1.000    0.974    0.998
Default lr, many trees         1.000    0.974    0.998
Low lr, many trees             0.996    0.974    0.998
Low lr, too few trees          0.936    0.930    0.983
```

Key observations:
- **Low learning rate + too few trees:** Underfits (93.0%) — hasn't converged yet.
- **High learning rate + few trees:** Can still achieve good accuracy but is less stable.
- **Low learning rate + many trees:** Often the best configuration — slower to train but more robust.
- In practice, set learning rate to 0.05–0.1 and use early stopping to determine the right number of trees.

## Effect of max_depth

Unlike random forests (which use full-depth trees), gradient boosting works best with **shallow trees**:

```python
print(f"{'max_depth':<12} {'Train acc':>10} {'Test acc':>10}")
print("-" * 34)
for depth in [1, 2, 3, 5, 10]:
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=42
    )
    gb.fit(X_train, y_train)
    print(f"{depth:<12} {gb.score(X_train, y_train):>10.3f} {gb.score(X_test, y_test):>10.3f}")
```

Output:
```
max_depth    Train acc   Test acc
----------------------------------
1                0.996      0.965
2                0.998      0.974
3                1.000      0.974
5                1.000      0.974
10               1.000      0.956
```

- **Depth=1 (stumps):** Slightly lower accuracy but very fast.
- **Depth=2–3:** Best balance. Standard recommendation for gradient boosting.
- **Depth=10:** Overfits — deep trees learn noise in the training data even when boosting, especially with a high learning rate.

## Feature Importances

Gradient boosting also provides feature importances:

```python
importances = pd.Series(gb.feature_importances_, index=cancer.feature_names)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 5))
top_features.sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Feature importance")
plt.title("Gradient Boosting: Top 10 Feature Importances")
plt.tight_layout()
plt.show()

print("Top 10 features:")
print(top_features.round(4))
```

Output:
```
Top 10 features:
worst concave points    0.3741
worst perimeter         0.1824
mean concave points     0.1201
worst radius            0.0819
worst area              0.0534
mean concavity          0.0441
mean perimeter          0.0402
worst texture           0.0218
mean texture            0.0185
area error              0.0150
```

Gradient boosting is more aggressive about concentrating importance in the top features (note `worst concave points` at 37%) compared to random forests (15%). This is because boosting corrects errors sequentially — the most informative feature gets used repeatedly in early trees, while random forests randomly rotate features.

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [2, 3, 5],
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid_gb.fit(X_train, y_train)

print(f"Best params: {grid_gb.best_params_}")
print(f"Best CV AUC: {grid_gb.best_score_:.3f}")

best_gb = grid_gb.best_estimator_
y_pred  = best_gb.predict(X_test)
y_proba = best_gb.predict_proba(X_test)[:, 1]
print(f"Test accuracy: {best_gb.score(X_test, y_test):.3f}")
print(f"Test AUC:      {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
Best params: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}
Best CV AUC: 0.998
Test accuracy: 0.982
Test AUC:      0.999
```

## Full Evaluation

```python
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
              precision    recall  f1-score   support

   malignant       0.976      0.953      0.964        43
      benign       0.972      0.986      0.979        71

    accuracy                           0.974       114

   macro avg       0.974      0.969      0.972       114
weighted avg       0.974      0.974      0.974       114

AUC: 0.999
```

Gradient Boosting achieves 97.4% accuracy with AUC 0.999 — the highest AUC seen so far in this module, matching SVM's test accuracy (98.2%) with slightly lower recall on malignant (95.3% vs 97.7%).

## XGBoost: The Industry Standard

Scikit-learn's `GradientBoostingClassifier` is a clean reference implementation. In practice, XGBoost and LightGBM are the standard choices for tabular data competitions and production systems, offering 10–100× faster training:

```python
# Install: pip install xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost:  accuracy={xgb.score(X_test, y_test):.3f},  AUC={roc_auc_score(y_test, y_proba_xgb):.3f}")
```

Output:
```
XGBoost:  accuracy=0.974,  AUC=0.998
```

XGBoost achieves the same accuracy as the scikit-learn implementation, with additional features like:
- **Native handling of missing values** (learns the best imputation direction at each split)
- **Regularization terms** (L1 and L2 on tree weights, in addition to the learning rate)
- **Early stopping** (stops adding trees when validation performance stops improving)
- **GPU acceleration** (for very large datasets)

## Gradient Boosting vs. Random Forests

| Property | Random Forest | Gradient Boosting |
|----------|--------------|-------------------|
| Training strategy | Parallel (independent trees) | Sequential (each tree corrects errors) |
| Tree depth | Deep (full, by default) | Shallow (depth 2–3 is typical) |
| Number of trees | More is always better | Too many → overfitting |
| Tuning effort | Low (defaults work well) | Moderate (learning rate + n_estimators) |
| Speed | Faster (parallel) | Slower (sequential) |
| Accuracy (breast cancer) | 95.6% | 97.4–98.2% |
| Robustness to outliers | High | Moderate (sensitive to noise) |

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Highest accuracy on tabular data (often) | Slower to train than random forests |
| Works on unscaled data (tree-based) | More hyperparameters to tune |
| Handles mixed data types | Can overfit with too many trees or too-large learning rate |
| Provides feature importances | Less robust to outliers than random forests |
| Native missing value handling (XGBoost) | Sequential nature prevents parallelization |

## When to Use Gradient Boosting

**Use Gradient Boosting when:**
- Maximum accuracy on tabular data is the goal
- Dataset is medium to large (1,000–1,000,000 samples)
- You are entering a data competition with structured data
- You need a production model with the best possible accuracy-complexity trade-off

**Avoid Gradient Boosting when:**
- Training speed is critical (random forests train faster)
- Dataset is noisy or has many outliers (boosting can amplify noise)
- You need a quick, low-tuning baseline (use random forests instead)
- Interpretability is required (like random forests, hundreds of trees are hard to explain)

## Conclusion

Gradient Boosting achieves AUC 0.999 on the breast cancer dataset — the highest in this module. Its sequential error-correction strategy gives it an edge over random forests on structured tabular data, particularly when tuned carefully. The key trade-offs are tuning effort (learning rate and n_estimators must be balanced) and training time (sequential tree building is slower than parallel bagging). For production-scale work, use XGBoost or LightGBM instead of scikit-learn's implementation. In the next lesson, you'll learn **Neural Networks** — the final algorithm in this module — which takes a completely different approach, learning hierarchical representations through interconnected layers of artificial neurons.

## Practice

### Knowledge Check

#### **Question 1: A gradient boosting model is trained with learning_rate=0.5 and n_estimators=50. The training accuracy is 99% but the test accuracy is 88%. What is the most likely cause, and how would you fix it?**

1. The model is underfitting because the learning rate is too low. Increase learning rate to 1.0 and add more trees.
2. The model is overfitting. A high learning rate (0.5) makes each tree's contribution large — the model rapidly fits training data, including noise. Fix: reduce learning_rate to 0.05–0.1 and increase n_estimators proportionally (e.g., 300–500 trees), or reduce max_depth.
3. The model is underfitting because 50 estimators are not enough. Increase n_estimators to 1,000.
4. The accuracy gap is expected — gradient boosting always shows a large train-test gap due to its sequential nature.

**Correct Answer:**
2. The model is overfitting. A high learning rate (0.5) makes each tree's contribution large — the model rapidly fits training data. Fix: reduce learning_rate to 0.05–0.1 and increase n_estimators proportionally.

**Explanation:**
A high learning rate means each tree is "trusted" heavily. With only 50 such trees, the model quickly memorizes training patterns — including noise. The standard fix is to reduce the learning rate and increase n_estimators proportionally: halving the learning rate requires roughly twice as many trees to reach the same training loss, but the resulting model generalizes better because each tree makes smaller, more conservative corrections. Early stopping (monitoring validation loss) is the most principled way to find the right n_estimators.

---

#### **Question 2: Why does gradient boosting typically use shallow trees (depth 1–3), while random forests use full-depth trees?**

1. Gradient boosting uses shallow trees because they train faster — deeper trees would make the algorithm too slow.
2. In gradient boosting, each tree is a weak learner that corrects the residuals of the previous ensemble. Shallow trees capture simple patterns (one or two interactions), and hundreds of such corrections combine into a strong model. Deep trees in gradient boosting would overfit by learning too much noise in the residuals. In random forests, deep trees are acceptable because averaging hundreds of overfit trees reduces variance.
3. Gradient boosting uses shallow trees because the algorithm cannot process more than 3 levels of depth.
4. Random forests use full-depth trees to ensure each tree is as different as possible from the others.

**Correct Answer:**
2. In gradient boosting, each tree corrects the residuals of the previous ensemble. Shallow trees capture simple patterns and hundreds of corrections combine into a strong model. Deep trees would overfit by learning noise in the residuals.

**Explanation:**
A key difference in philosophy: random forests reduce variance by averaging many high-variance (deep, overfit) trees — the diversity from bootstrap samples and feature randomness means their errors cancel. Gradient boosting reduces bias by iteratively correcting residuals with low-variance (shallow) trees — a deep tree in boosting would overfit the current residuals, amplifying rather than correcting noise. This is why gradient boosting is more sensitive to its tree depth hyperparameter than random forests.

---

#### **Question 3: XGBoost is described as "the industry standard for tabular data." What does it offer beyond scikit-learn's GradientBoostingClassifier?**

1. XGBoost uses a different algorithm (bagging instead of boosting) that is more accurate on most datasets.
2. XGBoost is the industry standard because it is the only gradient boosting library that supports multi-class classification.
3. XGBoost offers significantly faster training (10–100×) through optimized tree construction and optional GPU acceleration, built-in L1/L2 regularization on tree weights, native handling of missing values (learning the optimal branch direction at each split), and early stopping (halting when validation performance stops improving) — features that make it more accurate and practical at scale than the sklearn reference implementation.
4. XGBoost always achieves higher accuracy than GradientBoostingClassifier on the same data with the same hyperparameters.

**Correct Answer:**
3. XGBoost offers significantly faster training through optimized tree construction and GPU support, built-in regularization, native missing value handling, and early stopping — making it more accurate and practical at scale.

**Explanation:**
Scikit-learn's `GradientBoostingClassifier` is a clean, well-documented reference implementation — excellent for learning. XGBoost and LightGBM achieve the same algorithmic results but are engineered for scale: LightGBM, for instance, uses histogram-based splits that reduce memory usage by 20× and training time by 10× on large datasets. Early stopping is particularly valuable: rather than tuning n_estimators by grid search (training many complete models), you train one model and stop when validation performance plateaus — saving hours on large datasets.
