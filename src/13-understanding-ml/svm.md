# Support Vector Machines

## Overview

In the previous lesson, kNN achieved 97.4% accuracy by looking at the 7 most similar training points. Support Vector Machines (SVM) take the opposite approach: instead of consulting many neighbors, an SVM finds the single **optimal decision boundary** — the hyperplane that separates classes with the widest possible margin. Understanding SVMs builds geometric intuition for decision boundaries that carries forward to understanding why deep learning works in high-dimensional spaces.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Train SVMs for linear and non-linear classification using scikit-learn.
- Tune the C and gamma hyperparameters and understand their effect on the decision boundary.

## Key Terms

**Support Vector Machine (SVM):** A supervised learning algorithm that finds the hyperplane separating two classes with the maximum margin. Only the training points closest to the boundary (support vectors) influence the final model.

**Hyperplane:** In a 2D feature space, a line. In 3D, a plane. In d dimensions, a (d−1)-dimensional surface. The hyperplane is the decision boundary an SVM places between classes.

**Maximum margin:** The goal of SVM training — maximize the distance between the decision boundary and the nearest training point from each class. Larger margins generally generalize better.

**Support vectors:** The training examples that lie exactly on the margin boundary. These are the only points that define the hyperplane — removing any other training point would not change the model.

**C (regularization parameter):** Controls the trade-off between a wide margin and correctly classifying all training points. Small C allows margin violations (softer boundary, higher bias, lower variance). Large C penalizes violations heavily (narrower margin, lower bias, higher variance).

**Kernel:** A function that measures similarity between two points. The kernel trick allows SVM to fit non-linear boundaries without explicitly constructing new features. Common kernels: linear, RBF (Radial Basis Function), polynomial.

**RBF kernel (Gaussian kernel):** `K(a,b) = exp(−γ‖a−b‖²)`. Maps data into infinite-dimensional space. The most commonly used SVM kernel for non-linear problems.

**Gamma (γ):** Controls the influence radius of each training point in the RBF kernel. Low γ: wide influence, smoother boundary. High γ: narrow influence, more complex boundary.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## How SVMs Work

kNN classifies by majority vote among neighbors — it has no explicit boundary. SVM asks: **what is the best boundary?**

```
Two classes in 2D:

   ■ ■           ■ = malignant
■  ■  ■          ● = benign
   ■  ●●●
      ● ●●●
         ●●●

kNN: "classify by the nearest k points"
SVM: "find the widest street between the classes"

        ■ ■
     ■  ■  ■
        ■  | margin | ●●●
           |  max   |● ●●●
           |        |   ●●●
        boundary
```

The support vectors are the points on the edges of this "street." The SVM finds the boundary that makes the street as wide as possible.

**Why maximum margin?** A boundary placed far from both classes generalizes better — new points that fall close to (but not on) the boundary are still classified correctly.

**What if the data isn't linearly separable?** Two solutions:
1. **Soft margin (C parameter):** Allow some points to be misclassified — use a penalty for violations instead of requiring perfect separation.
2. **Kernel trick:** Map the data into a higher-dimensional space where it *is* linearly separable, without explicitly computing the transformation.

## Linear SVM on Breast Cancer

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm_linear.fit(X_train_s, y_train)

acc   = svm_linear.score(X_test_s, y_test)
proba = svm_linear.predict_proba(X_test_s)[:, 1]
auc   = roc_auc_score(y_test, proba)

print(f"Linear SVM (C=1):  accuracy={acc:.3f},  AUC={auc:.3f}")
print(f"Number of support vectors: {svm_linear.n_support_}")
```

Output:
```
Linear SVM (C=1):  accuracy=0.974,  AUC=0.997
Number of support vectors: [22 30]
```

With 30 features, the linear SVM finds a 30-dimensional hyperplane. Only 52 of the 455 training points (22 from the malignant class, 30 from benign) are support vectors — these alone define the entire decision boundary.

## Effect of C: The Bias-Variance Trade-off

```python
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accs, test_accs = [], []

for C in C_values:
    svm = SVC(kernel="linear", C=C, random_state=42)
    svm.fit(X_train_s, y_train)
    train_accs.append(svm.score(X_train_s, y_train))
    test_accs.append(svm.score(X_test_s, y_test))

plt.figure(figsize=(9, 4))
plt.semilogx(C_values, train_accs, label="Train accuracy", color="steelblue", marker="o")
plt.semilogx(C_values, test_accs,  label="Test accuracy",  color="coral",    marker="o")
plt.xlabel("C (log scale)")
plt.ylabel("Accuracy")
plt.title("Linear SVM: Effect of C")
plt.legend()
plt.tight_layout()
plt.show()
```

- **Very small C (0.001):** The margin is wide and many training points are allowed to be on the wrong side — high bias, low variance. The model underfits.
- **Large C (1000):** The model penalizes every misclassification heavily, fitting the training data tightly — low bias, high variance. May overfit.
- **C=1:** A reasonable default that balances the two.

## The RBF Kernel: Non-Linear Boundaries

The RBF kernel maps data into a space where non-linear boundaries become linear. The parameter γ (gamma) controls how far each training point's influence reaches:

```python
# Compare kernels
for kernel in ["linear", "rbf", "poly"]:
    svm = SVC(kernel=kernel, C=1.0, probability=True, random_state=42)
    svm.fit(X_train_s, y_train)
    acc = svm.score(X_test_s, y_test)
    auc = roc_auc_score(y_test, svm.predict_proba(X_test_s)[:, 1])
    print(f"kernel={kernel:<8}  accuracy={acc:.3f}  AUC={auc:.3f}")
```

Output:
```
kernel=linear    accuracy=0.974  AUC=0.997
kernel=rbf       accuracy=0.982  AUC=0.998
kernel=poly      accuracy=0.974  AUC=0.995
```

The RBF kernel achieves the highest accuracy (98.2%) on this dataset. With 30 features that are already continuous and standardized, RBF can capture non-linear relationships that the linear kernel misses.

## Tuning C and Gamma Together

For RBF SVMs, C and gamma must be tuned jointly:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C":     [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, "scale"]
}

grid_svm = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid_svm.fit(X_train_s, y_train)

print(f"Best params: {grid_svm.best_params_}")
print(f"Best CV AUC: {grid_svm.best_score_:.3f}")

# Evaluate on held-out test set
best_svm = grid_svm.best_estimator_
y_pred  = best_svm.predict(X_test_s)
y_proba = best_svm.predict_proba(X_test_s)[:, 1]
print(f"Test accuracy: {best_svm.score(X_test_s, y_test):.3f}")
print(f"Test AUC:      {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
Best params: {'C': 10, 'gamma': 0.01}
Best CV AUC: 0.997
Test accuracy: 0.982
Test AUC:      0.998
```

**Interpretation of gamma:**
- `"scale"` (default in scikit-learn): γ = 1 / (n_features × X.var()). A sensible data-adaptive default.
- Small γ (0.001): Each training point influences a wide region — smoother boundary, more bias.
- Large γ (1): Each training point influences only its immediate neighborhood — complex boundary, more variance.

## Full Evaluation

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
              precision    recall  f1-score   support

   malignant       0.977      0.977      0.977        43
      benign       0.986      0.986      0.986        71

    accuracy                           0.982       114

   macro avg       0.982      0.982      0.982       114
weighted avg       0.982      0.982      0.982       114

AUC: 0.998
```

SVM (RBF, C=10, γ=0.01) achieves 98.2% accuracy and 97.7% recall on malignant — better than kNN (97.4% / 95.3%) and logistic regression (95.6% / 92.9%).

## SVM for Regression

SVM extends to regression via SVR (Support Vector Regression). Instead of finding a boundary between classes, SVR fits a tube around the predictions and only penalizes points that fall outside the tube:

```python
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)
Xh = housing.frame.drop(columns=["MedHouseVal"])
yh = housing.frame["MedHouseVal"]

Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)

scaler_h = StandardScaler()
Xh_tr_s = scaler_h.fit_transform(Xh_tr)
Xh_te_s = scaler_h.transform(Xh_te)

svr = SVR(kernel="rbf", C=10, gamma="scale")
svr.fit(Xh_tr_s, yh_tr)
r2 = r2_score(yh_te, svr.predict(Xh_te_s))
print(f"SVR (RBF):  R²={r2:.3f}")
```

Output:
```
SVR (RBF):  R²=0.736
```

SVR (R²=0.736) outperforms kNN regression (R²=0.703) and linear regression (R²=0.576) on the California Housing dataset. Note: SVR is slower on large datasets — the California Housing dataset (16,512 training samples) takes noticeably longer than classification tasks on the 455-sample breast cancer dataset.

## Why Scaling Is Critical for SVM

Like kNN, SVM computes distances between points. Features on different scales dominate:

```python
# Unscaled SVM
svm_unscaled = SVC(kernel="rbf", C=10, random_state=42)
svm_unscaled.fit(X_train, y_train)   # no scaling

# Scaled SVM
svm_scaled = SVC(kernel="rbf", C=10, random_state=42)
svm_scaled.fit(X_train_s, y_train)

print(f"SVM unscaled:  {svm_unscaled.score(X_test, y_test):.3f}")
print(f"SVM scaled:    {svm_scaled.score(X_test_s, y_test):.3f}")
```

Output:
```
SVM unscaled:  0.632
SVM scaled:    0.982
```

A 35% accuracy gap from one missing preprocessing step. Without scaling, the `mean area` feature (range 143–2501) completely dominates the kernel distance calculation.

**Rule: always scale features before using SVM.**

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Effective in high-dimensional spaces (text, images) | Slow on large datasets — O(n² to n³) training time |
| Works well with small to medium datasets | Sensitive to feature scaling |
| Memory-efficient (only support vectors stored) | No natural probability estimates (requires calibration) |
| Robust to outliers (only support vectors matter) | Kernel and C/gamma must be tuned together |
| Non-linear classification via kernels without explicit feature construction | No interpretable feature importances |

## When to Use SVM

**Use SVM when:**
- Dataset is small to medium (< 10,000 samples)
- Feature dimensionality is high relative to sample size (text classification, genomics)
- You need a strong baseline non-linear classifier with a small number of hyperparameters
- The classes are well-separated (clear margin exists)

**Avoid SVM when:**
- Dataset is large (> 100,000 samples) — training becomes prohibitively slow
- You need probability estimates directly (use logistic regression or gradient boosting)
- Features are highly redundant (tree-based methods handle this better)
- Interpretability is required (SVM provides no feature importances)

## Conclusion

SVM with an RBF kernel achieves 98.2% accuracy and AUC 0.998 on the breast cancer dataset — the best result so far, surpassing logistic regression (95.6%), decision trees (93.0%), and kNN (97.4%). The critical lessons are the same as kNN: **scaling is non-negotiable** (a 35% accuracy gap without it), and hyperparameters (C and gamma) must be tuned together. SVM's geometric foundation — maximizing the margin, storing only support vectors — makes it memory-efficient and robust to outliers, but expensive to train on large datasets. In the next lesson, you'll learn **Random Forests**, which achieve competitive performance by a completely different mechanism: averaging hundreds of decision trees.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="C is the penalty for misclassifying training points. A large C says &quot;misclassifications are very costly — squeeze the margin as narrow as necessary to classify everything correctly.&quot; The result is a boundary that fits training data tightly, potentially including noise, which hurts generalization. A small C says &quot;a wide margin is more valuable than perfect training accuracy,&quot; accepting some misclassifications to achieve a more robust boundary.">
  <div class="quiz-question">
    <strong>Question 1:</strong> An SVM is trained with a very large C value (e.g., C=10,000). What is the likely effect on the model?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The model will have a very wide margin and may underfit by ignoring some correctly-classified points near the boundary.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The model will be forced to correctly classify every training point, resulting in a very narrow margin and a decision boundary that closely fits the training data. This reduces bias but increases variance — the model may overfit.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The model will behave identically to a model with C=1 because SVM is insensitive to the regularization parameter.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The model will automatically switch from a linear kernel to an RBF kernel to accommodate the large C.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="A polynomial kernel of degree 3 with 30 features would require constructing roughly 5,000 explicit polynomial features. The RBF kernel implicitly maps data into an infinite-dimensional space — but because the SVM&#039;s math only requires dot products between points (not explicit coordinates), the kernel function computes these dot products directly in the original space. This is the &quot;trick&quot;: infinite-dimensional transformations computed with a simple exponential function.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does the kernel trick allow an SVM to do, and why is it computationally valuable?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The kernel trick allows an SVM to use gradient descent instead of quadratic programming, making training faster.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The kernel trick allows an SVM to compute similarities between points in a higher-dimensional feature space (where they may be linearly separable) without explicitly constructing that high-dimensional representation. This makes non-linear SVM tractable — computing a dot product in the original space instead of constructing millions of new features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The kernel trick is a method for automatically tuning the C parameter using cross-validation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The kernel trick allows an SVM to make probabilistic predictions by transforming the decision function output into a probability score.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="With 50,000 samples, SVM becomes slow (O(n²) to O(n³) training time). kNN requires computing distances across 100,000-dimensional vectors for every prediction — prohibitively expensive and ineffective in very high dimensions (curse of dimensionality). Decision trees split on one feature at a time and struggle with 100,000 nearly-equally-important features. Logistic regression with L1 regularization is the standard for sparse text: it trains in seconds via stochastic gradient descent, naturally handles high dimensionality, and produces interpretable coefficients. (Linear SVM is also excellent for text, but logistic regression&#039;s probability outputs are often more useful.)">
  <div class="quiz-question">
    <strong>Question 3:</strong> You are building a text classification model with 50,000 training documents and 100,000 vocabulary features (TF-IDF sparse matrix). Which algorithm is likely most appropriate, and why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>kNN — it requires no training and handles high-dimensional data well.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Logistic regression — it is fast to train and handles high-dimensional sparse features efficiently with L1/L2 regularization.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Neural network — it always outperforms simpler methods on text data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Decision tree — it selects the most discriminative features automatically.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

