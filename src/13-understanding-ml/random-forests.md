# Random Forests

## Overview

In the previous lesson, SVM achieved 98.2% accuracy by finding the single optimal boundary between classes. Random Forests take a fundamentally different approach: instead of one perfect model, build hundreds of imperfect models and let them vote. This "wisdom of crowds" strategy — called **bagging** — reduces variance dramatically while keeping bias low, and produces one of the most reliable general-purpose classifiers in machine learning.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Build ensemble models using bagged decision trees (Random Forests).
- Interpret feature importances and tune key hyperparameters (`n_estimators`, `max_features`, `max_depth`).

## Key Terms

**Ensemble method:** A model that combines multiple individual models (base learners) to produce better predictions than any single model alone. Random forests are the canonical ensemble via bagging.

**Bagging (Bootstrap Aggregating):** Training multiple models on different random subsets of the training data (sampled with replacement), then averaging their predictions. Reduces variance without increasing bias.

**Bootstrap sample:** A random sample drawn with replacement from the training set, the same size as the original. Each bootstrap sample includes roughly 63% of unique training examples (the rest are "out-of-bag").

**Out-of-bag (OOB) error:** An automatic validation estimate: each tree is evaluated on the training examples that were not in its bootstrap sample. Provides a free cross-validation estimate without a separate validation set.

**Feature randomness:** At each split in a random forest tree, only a random subset of `max_features` features is considered. This decorrelates the trees — different trees focus on different features — making the ensemble more robust than averaging correlated trees.

**Feature importance:** A measure of how much each feature reduces impurity (e.g., Gini impurity) across all splits in all trees. Averaged over the forest. Provides a useful ranking of predictive features.

**Gini impurity:** A measure of node impurity: `Gini = 1 − Σ(pᵢ²)`. A perfectly pure node (all one class) has Gini=0. The forest selects splits that minimize weighted Gini impurity.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/04_random-forests_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Why Single Trees Fail

Before building the ensemble, establish the single-tree baseline:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Note: Random forests internally scale is unnecessary (tree-based splits
# are scale-invariant). We use unscaled X here.
depths = [None, 2, 5, 10]
print(f"{'max_depth':<12} {'Train acc':>10} {'Test acc':>10}")
print("-" * 34)
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    tr = dt.score(X_train, y_train)
    te = dt.score(X_test, y_test)
    print(f"{str(d):<12} {tr:>10.3f} {te:>10.3f}")
```

Output:
```
max_depth    Train acc   Test acc
----------------------------------
None              1.000      0.930
2                 0.967      0.939
5                 0.991      0.930
10                1.000      0.930
```

The single decision tree plateaus at ~93% test accuracy regardless of depth. Unconstrained trees overfit (100% train, 93% test). Shallow trees underfit. This ceiling is a fundamental limitation of a single tree's high variance.

## How Random Forests Work

```
Training data (455 samples):

Bootstrap sample 1  →  Tree 1  →  prediction₁
Bootstrap sample 2  →  Tree 2  →  prediction₂
Bootstrap sample 3  →  Tree 3  →  prediction₃
       ...                              ...
Bootstrap sample N  →  Tree N  →  predictionₙ

                    MAJORITY VOTE
                    ─────────────
                    Final prediction
```

Two sources of randomness make trees different from each other:
1. **Different bootstrap samples** — each tree sees a different 63% of the data.
2. **Random feature subsets at each split** — each tree considers only `sqrt(n_features)` features at each split (default for classification).

These two sources of randomness make the trees decorrelated — their errors don't all occur on the same examples, so averaging cancels out much of the noise.

## Basic Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Random Forest (100 trees):")
print(f"  Train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"  Test accuracy:  {rf.score(X_test, y_test):.3f}")
print(f"  OOB score:      {RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42).fit(X_train, y_train).oob_score_:.3f}")
```

Output:
```
Random Forest (100 trees):
  Train accuracy: 1.000
  Test accuracy:  0.965
  OOB score:      0.960
```

The forest still shows 100% training accuracy (trees are grown to full depth by default), but the test accuracy (96.5%) is substantially better than a single tree (93.0%) — the ensemble effect at work. The OOB score (96.0%) is a free estimate of generalization performance, close to the held-out test score.

## Effect of n_estimators

How many trees do you need?

```python
n_values = [1, 5, 10, 25, 50, 100, 200, 500]
test_accs = []

for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    test_accs.append(rf.score(X_test, y_test))

plt.figure(figsize=(9, 4))
plt.plot(n_values, test_accs, marker="o", color="steelblue")
plt.xlabel("n_estimators (number of trees)")
plt.ylabel("Test accuracy")
plt.title("Random Forest: Accuracy vs. Number of Trees")
plt.xscale("log")
plt.tight_layout()
plt.show()

print(f"{'n_estimators':>14} {'Test acc':>10}")
for n, acc in zip(n_values, test_accs):
    print(f"{n:>14} {acc:>10.3f}")
```

Output:
```
 n_estimators   Test acc
             1      0.930
             5      0.956
            10      0.956
            25      0.965
            50      0.965
           100      0.965
           200      0.965
           500      0.965
```

Accuracy improves rapidly up to ~25 trees, then plateaus. More trees never hurt (they don't overfit), but beyond ~100 the improvement is negligible. The practical recommendation: **start with 100–200 trees** and increase only if you see instability across runs.

## Feature Importances

Random forests automatically rank features by how much they reduce impurity across all splits in all trees:

```python
importances = pd.Series(rf.feature_importances_, index=cancer.feature_names)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 5))
top_features.sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Mean impurity decrease")
plt.title("Random Forest: Top 10 Feature Importances")
plt.tight_layout()
plt.show()

print("Top 10 features:")
print(top_features.round(4))
```

Output:
```
Top 10 features:
worst concave points      0.1521
worst perimeter           0.1278
worst radius              0.1124
mean concave points       0.0994
worst area                0.0863
mean perimeter            0.0563
mean radius               0.0548
worst texture             0.0402
mean concavity            0.0348
mean area                 0.0326
```

The top features (`worst concave points`, `worst perimeter`, `worst radius`) are measurements of the most severe cell nucleus features — which aligns with medical intuition that the most extreme cells in a tumor sample are most diagnostically significant.

Feature importances from random forests are useful for:
- Understanding which features drive predictions
- Feature selection (dropping the lowest-importance features)
- Communicating results to stakeholders

**Caveat:** Random forest importances can be misleading when features are correlated. If two features carry the same information, their importance is split between them, underestimating each one's true contribution.

## Tuning max_features and max_depth

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_features": ["sqrt", "log2", None],   # None = all features
    "max_depth":    [None, 10, 20],
    "min_samples_leaf": [1, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

print(f"Best params: {grid_rf.best_params_}")
print(f"Best CV AUC: {grid_rf.best_score_:.3f}")

best_rf = grid_rf.best_estimator_
y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]
print(f"Test accuracy: {best_rf.score(X_test, y_test):.3f}")
print(f"Test AUC:      {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
Best params: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 200}
Best CV AUC: 0.994
Test accuracy: 0.974
Test AUC:      0.994
```

**Key hyperparameters:**

| Hyperparameter | Default | Effect |
|----------------|---------|--------|
| `n_estimators` | 100 | More trees = more stable, diminishing returns past ~200 |
| `max_features` | `"sqrt"` | Fraction of features considered per split. `"sqrt"` is standard for classification |
| `max_depth` | None (full) | Limiting depth reduces variance but can increase bias |
| `min_samples_leaf` | 1 | Minimum samples at a leaf. Increasing reduces overfitting |

## Full Evaluation

```python
best_rf = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", random_state=42
)
best_rf.fit(X_train, y_train)
y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
              precision    recall  f1-score   support

   malignant       0.975      0.907      0.940        43
      benign       0.950      0.986      0.968        71

    accuracy                           0.956       114

   macro avg       0.963      0.947      0.954       114
weighted avg       0.960      0.956      0.957       114

AUC: 0.994
```

Random Forest achieves 95.6% accuracy and AUC 0.994, matching logistic regression's accuracy but with 39.5% recall on malignant — somewhat lower than SVM (97.7%) and kNN (95.3%). The AUC (0.994), however, shows the forest is an excellent ranker even if the default threshold isn't optimally tuned for recall.

**Note on recall:** The default threshold of 0.5 is not optimal for the safety-critical task of detecting malignant tumors. Lowering the threshold (as covered in module 09) would improve malignant recall at a cost to precision — the AUC of 0.994 confirms that with the right threshold, the forest is a strong classifier.

## Comparison: Single Tree vs. Random Forest

| Property | Single Decision Tree | Random Forest |
|----------|---------------------|---------------|
| Training time | Fast | Slower (×n_estimators) |
| Prediction time | Fast | Slower (×n_estimators) |
| Interpretability | High (visual tree) | Low (hundreds of trees) |
| Overfitting tendency | High | Low |
| Accuracy (breast cancer) | 93.0% | 97.4% |
| Feature importances | Yes | Yes (averaged, more stable) |
| Handles missing values | No (sklearn) | No (sklearn) |

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Excellent out-of-the-box accuracy | Less interpretable than a single tree |
| Robust to outliers and noisy features | Slower to train and predict than single trees |
| Handles mixed data types (numeric + categorical) | Larger memory footprint (stores all trees) |
| Provides feature importances | Feature importances can mislead with correlated features |
| OOB error as free validation estimate | Not ideal for very high-dimensional sparse data (text) |
| Low tuning effort — defaults work well | Predictions are hard to explain to individuals |

## When to Use Random Forests

**Use Random Forests when:**
- Dataset is medium to large (1,000–1,000,000 samples)
- You need a strong general-purpose baseline with minimal tuning
- You want interpretable feature importances
- The dataset has mixed data types or outliers
- You don't know which algorithm to try first — random forests are an excellent default

**Avoid Random Forests when:**
- Interpretability requires a single set of explicit rules (use a single decision tree)
- Dataset is very high-dimensional and sparse (use logistic regression or SVM)
- Prediction latency is critical (each prediction requires all trees — use logistic regression)
- Maximum accuracy on tabular data is needed (gradient boosting typically beats random forests)

## Conclusion

Random Forests demonstrate that combining many imperfect models can surpass any single model. On the breast cancer dataset, 200 trees achieve 95.6% accuracy and AUC 0.994 — better than a single decision tree (93.0%) and competitive with logistic regression (95.6% / AUC 0.993), with the added benefit of feature importances. The key insight is **bagging + feature randomness**: each tree sees different data and considers different features, so their errors are largely independent and average out. In the next lesson, you'll learn **Gradient Boosting** — which also uses ensembles of trees but builds them sequentially rather than in parallel, targeting the specific mistakes of previous trees.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/04_random-forests_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="When sampling n examples with replacement from n examples, the probability any single example is NOT selected in a given draw is (1 − 1/n). After n draws, the probability it&#039;s never selected is (1 − 1/n)ⁿ → 1/e ≈ 0.368. So about 36.8% of training examples are left out of each bootstrap sample. These &quot;out-of-bag&quot; examples can be used as a validation set for that tree, and averaging OOB error across all trees gives a free cross-validation estimate.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A random forest uses bootstrap sampling. What does this mean, and what is an "out-of-bag" sample?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Bootstrap sampling means the forest randomly removes features that are not useful. Out-of-bag samples are features that were removed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Bootstrap sampling means each tree is trained on a random sample of training examples drawn with replacement (the same size as the original dataset). This means roughly 37% of training examples are not in each bootstrap sample — these are "out-of-bag" and can be used to estimate generalization performance without a separate validation set.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Bootstrap sampling means the forest randomly selects the number of trees to use. Out-of-bag samples are trees that were not selected.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Bootstrap sampling is a statistical technique that ensures all training examples appear equally often across all trees.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="If you trained 100 trees all on the same data with the same features, they&#039;d produce nearly identical predictions — averaging them would give no benefit. The two randomization sources solve this: different bootstrap samples give different training sets, and limiting `max_features` per split prevents all trees from always choosing the same dominant features. The result is a diverse ensemble where tree disagreements expose genuine uncertainty and tree agreements signal confident predictions.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why do random forests typically outperform single decision trees, and what role does feature randomness play?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Random forests outperform single trees because they use more data — each tree sees all training examples, while a single tree sees only a subset.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Random forests outperform single trees because they ensemble many trees trained on different bootstrap samples. Feature randomness (considering only sqrt(n_features) features per split) decorrelates the trees — different trees specialize on different features, so their errors are largely independent. Averaging independent errors reduces variance without increasing bias.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Random forests outperform single trees because they use a more sophisticated splitting criterion (Gini impurity vs. entropy).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Random forests outperform single trees because they automatically tune the max_depth hyperparameter for each tree using cross-validation.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When two features carry redundant information, a random forest at each split picks whichever version it sees first (or whichever is in the current random feature subset). Over many trees, the importance is split between the correlated features — both appear less important than either would be alone. Additionally, tree-based importance (mean impurity decrease) can favor high-cardinality features. A safer approach: compute permutation importance (shuffling each feature and measuring accuracy drop) which is less susceptible to these biases, or test model performance with and without the feature.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Random forest feature importances show that feature A has importance 0.35 and feature B has importance 0.02. A colleague concludes that feature B can be safely removed. What concern should you raise?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Feature importances in random forests are not normalized, so 0.02 doesn't mean feature B is unimportant.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Feature importances can be misleading when features are correlated. If feature B is highly correlated with feature A, the forest may have distributed their shared importance primarily to A. Feature B might still contain predictive information. Before dropping it, verify with a model trained without feature B, or check correlation with A.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Random forest importances always underestimate the importance of categorical features, so feature B is probably more important than it appears.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The threshold for removing features should be 0.05, not 0.02 — so feature B is borderline important.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

