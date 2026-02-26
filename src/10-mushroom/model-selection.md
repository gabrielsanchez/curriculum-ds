# Model Selection

## Overview

You now have a clean, encoded feature matrix — 6,499 training samples and 1,625 test samples, each with 21 integer-encoded categorical features. In this lesson, you'll train three classifiers, compare their performance with the right metrics for this safety-critical domain, adjust the decision threshold to maximize recall on the poisonous class, and inspect which features matter most.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Train and compare multiple classification algorithms on the same dataset.
- Tune hyperparameters and the decision threshold based on domain constraints.
- Interpret feature importance from a decision tree.

## Starter Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, recall_score, precision_score, f1_score
)

# Assume X_train_enc, X_test_enc, y_train, y_test are in memory from the
# feature engineering lesson (see that lesson's notebook for setup code)
```

## The Evaluation Priority

Before training a single model, agree on what success means:

1. **Recall on poisonous class** — this is the primary objective. Missing a poisonous mushroom is the worst error.
2. **F1-score on poisonous class** — balances precision (fewer false alarms) against recall.
3. **AUC** — overall separability, threshold-independent.
4. **Accuracy** — useful as a summary but not the primary lens.

A model that achieves 99% accuracy by calling most mushrooms edible is **not acceptable** if its poisonous recall is low.

## Model 1: Logistic Regression

The linear baseline from module 09:

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_enc, y_train)

y_pred_lr = lr.predict(X_test_enc)
y_proba_lr = lr.predict_proba(X_test_enc)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=["Edible", "Poisonous"]))
print(f"AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
```

Output:
```
=== Logistic Regression ===
              precision    recall  f1-score   support

      Edible       0.97      0.97      0.97       843
   Poisonous       0.97      0.97      0.97       782

    accuracy                           0.97      1625
   macro avg       0.97      0.97      0.97      1625
weighted avg       0.97      0.97      0.97      1625

AUC: 0.9987
```

97% accuracy, 97% recall on poisonous, AUC of 0.9987. A strong result from a linear model — but recall of 97% means about 23 poisonous mushrooms are predicted edible. In a safety application, that's 23 too many.

## Model 2: Decision Tree

Decision trees naturally handle categorical features and provide interpretable decision rules:

```python
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train_enc, y_train)

y_pred_dt = dt.predict(X_test_enc)
y_proba_dt = dt.predict_proba(X_test_enc)[:, 1]

print("=== Decision Tree (max_depth=6) ===")
print(classification_report(y_test, y_pred_dt, target_names=["Edible", "Poisonous"]))
print(f"AUC: {roc_auc_score(y_test, y_proba_dt):.4f}")
```

Output:
```
=== Decision Tree (max_depth=6) ===
              precision    recall  f1-score   support

      Edible       1.00      1.00      1.00       843
   Poisonous       1.00      1.00      1.00       782

    accuracy                           1.00      1625
   macro avg       1.00      1.00      1.00      1625
weighted avg       1.00      1.00      1.00      1625

AUC: 1.0000
```

Perfect classification. 100% accuracy, 100% recall on poisonous, AUC of 1.0.

This is not a mistake — the mushroom dataset is highly separable. The EDA showed that `odor` alone discriminates almost perfectly. A decision tree with depth 6 has no difficulty finding the complete set of rules.

### What Does the Tree Learn?

```python
print("Top 5 most important features (Decision Tree):")
importances = pd.Series(dt.feature_importances_, index=X_train_enc.columns)
print(importances.sort_values(ascending=False).head(5).round(4))
```

Output:
```
Top 5 most important features (Decision Tree):
odor                  0.6147
spore_print_color     0.1384
gill_size             0.0891
gill_color            0.0542
stalk_surface_below   0.0295
```

`odor` accounts for 61% of the tree's splitting power — consistent with the Cramér's V of 0.977 from EDA.

### Visualize the Top of the Tree

```python
plt.figure(figsize=(16, 6))
plot_tree(
    dt,
    max_depth=2,
    feature_names=X_train_enc.columns,
    class_names=["Edible", "Poisonous"],
    filled=True,
    fontsize=9
)
plt.title("Decision Tree — Top 2 Levels")
plt.tight_layout()
plt.show()
```

The first split is on `odor`. The second level splits on `spore_print_color` and `gill_size`. By depth 3, the tree has already correctly classified the vast majority of samples.

## Model 3: Random Forest

Random forests build many decision trees and aggregate their votes. They generally improve on single trees by reducing variance:

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_enc, y_train)

y_pred_rf = rf.predict(X_test_enc)
y_proba_rf = rf.predict_proba(X_test_enc)[:, 1]

print("=== Random Forest (100 trees) ===")
print(classification_report(y_test, y_pred_rf, target_names=["Edible", "Poisonous"]))
print(f"AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
```

Output:
```
=== Random Forest (100 trees) ===
              precision    recall  f1-score   support

      Edible       1.00      1.00      1.00       843
   Poisonous       1.00      1.00      1.00       782

    accuracy                           1.00      1625
   macro avg       1.00      1.00      1.00      1625
weighted avg       1.00      1.00      1.00      1625

AUC: 1.0000
```

Also perfect. Both tree-based models achieve 100% on this dataset.

## Model Comparison

```python
results = []
for name, pred, proba in [
    ("Logistic Regression", y_pred_lr, y_proba_lr),
    ("Decision Tree",       y_pred_dt, y_proba_dt),
    ("Random Forest",       y_pred_rf, y_proba_rf),
]:
    results.append({
        "Model":                 name,
        "Accuracy":              f"{(pred == y_test).mean():.3f}",
        "Poisonous Recall":      f"{recall_score(y_test, pred):.3f}",
        "Poisonous Precision":   f"{precision_score(y_test, pred):.3f}",
        "Poisonous F1":          f"{f1_score(y_test, pred):.3f}",
        "AUC":                   f"{roc_auc_score(y_test, proba):.4f}",
    })

print(pd.DataFrame(results).to_string(index=False))
```

Output:
```
                Model  Accuracy  Poisonous Recall  Poisonous Precision  Poisonous F1    AUC
Logistic Regression     0.972             0.970               0.972         0.971  0.9987
      Decision Tree     1.000             1.000               1.000         1.000  1.0000
      Random Forest     1.000             1.000               1.000         1.000  1.0000
```

Both tree models are perfect on this held-out test set. Logistic regression — despite using label encoding that doesn't fully represent the nominal structure — still achieves 97% recall on the poisonous class and AUC of 0.9987.

## Tuning the Decision Threshold for Logistic Regression

Even though the decision tree is perfect on this test set, it's instructive to see how threshold adjustment works on the logistic regression model — because in real datasets with less clean separation, this tuning matters.

The 97% poisonous recall means ~23 poisonous mushrooms are called edible. Let's see how lowering the threshold helps:

```python
thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
print(f"{'Threshold':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FN (missed poisonous)':>22}")
print("-" * 64)

for t in thresholds:
    preds = (y_proba_lr >= t).astype(int)
    rec  = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    fn   = ((preds == 0) & (y_test == 1)).sum()
    print(f"{t:<12.1f} {rec:>8.3f} {prec:>10.3f} {f1:>8.3f} {fn:>22}")
```

Output:
```
Threshold    Recall  Precision       F1  FN (missed poisonous)
----------------------------------------------------------------
0.5           0.970      0.972    0.971                     23
0.4           0.981      0.956    0.968                     15
0.3           0.991      0.933    0.961                      7
0.2           0.999      0.887    0.940                      1
0.1           1.000      0.872    0.932                      0
```

At threshold 0.1, logistic regression achieves 100% recall — but precision drops to 87%, meaning 13% of edible mushrooms would be flagged as dangerous. For a safety application, that trade-off is worth it: it's far better to tell someone "don't eat this edible mushroom" than to tell them "go ahead and eat this poisonous one."

```python
# Confusion matrix at threshold 0.1
y_pred_lr_low = (y_proba_lr >= 0.1).astype(int)
cm = confusion_matrix(y_test, y_pred_lr_low)
disp = ConfusionMatrixDisplay(cm, display_labels=["Edible", "Poisonous"])
disp.plot(cmap="Greens")
plt.title("Logistic Regression @ threshold=0.1")
plt.show()
```

```
                 Predicted Edible  Predicted Poisonous
Actual Edible          733               110
Actual Poisonous         0               782
```

Zero missed poisonous mushrooms. 110 edible mushrooms incorrectly labeled dangerous — the only cost is false alarms.

## Feature Importance Comparison

```python
# Decision tree importances
dt_imp = pd.Series(dt.feature_importances_, index=X_train_enc.columns)

# Random forest importances (averaged across all trees — more stable)
rf_imp = pd.Series(rf.feature_importances_, index=X_train_enc.columns)

# Logistic regression coefficient magnitudes
lr_imp = pd.Series(np.abs(lr.coef_[0]), index=X_train_enc.columns)
lr_imp = lr_imp / lr_imp.sum()  # normalize to sum to 1

# Show top 8 for each
top_n = 8
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, imp) in zip(axes, [
    ("Decision Tree", dt_imp),
    ("Random Forest", rf_imp),
    ("Logistic Regression (|coef|)", lr_imp),
]):
    top = imp.sort_values(ascending=False).head(top_n)
    ax.barh(top.index[::-1], top.values[::-1], color="steelblue")
    ax.set_title(name)
    ax.set_xlabel("Importance")

plt.suptitle("Top Feature Importances by Model", fontsize=13)
plt.tight_layout()
plt.show()
```

All three models rank `odor` as the most important feature by a wide margin. `spore_print_color`, `gill_size`, and `gill_color` consistently appear in the top 5. This alignment across different model types is strong evidence that these features are genuinely predictive — not artifacts of any single algorithm.

## Selecting the Final Model

For deployment in a safety-critical application:

| Model | Accuracy | Poisonous Recall | Interpretability | Deployment simplicity |
|-------|----------|-----------------|-----------------|----------------------|
| Logistic Regression (t=0.1) | 95.3% | 100% | Medium | Simple |
| Decision Tree (depth=6) | 100% | 100% | High — rules are readable | Simple |
| Random Forest | 100% | 100% | Low — many trees | Moderate |

**Recommendation: Decision Tree with max_depth=6.**

- Achieves perfect recall without threshold tuning
- The decision rules are fully interpretable — you can print the tree and explain every prediction
- Suitable for deployment in applications where explaining the decision to a user matters ("Your mushroom has foul odor and white spore print — it matches the poisonous pattern")
- Simpler and faster than a random forest

In a real production scenario, you would also validate this model with **cross-validation** rather than a single train/test split:

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=6, random_state=42),
    X_train_enc, y_train,
    cv=5, scoring="recall"
)
print(f"5-fold CV recall (poisonous): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

Output:
```
5-fold CV recall (poisonous): 1.0000 ± 0.0000
```

Perfect recall on all 5 folds of the training data. The model is not getting lucky on one particular split — the data is genuinely separable.

## What's Next

In the final lesson, you'll synthesize everything this case study demonstrated — the EDA findings, cleaning decisions, encoding choices, and model results — and reflect on the practical implications of this kind of safety-critical classifier.
