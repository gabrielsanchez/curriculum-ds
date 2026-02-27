# k-Nearest Neighbors

## Overview

In the previous lesson, you established that logistic regression achieves 95.6% accuracy on the breast cancer dataset. k-Nearest Neighbors (kNN) is the most conceptually direct algorithm in this module: it makes no assumptions about the data's distribution, learns nothing during training, and classifies each new point by asking "what are the k closest training examples, and what labels do they have?" Understanding kNN deeply — including where it excels and where it breaks down — builds intuition for distance-based learning that carries through to SVMs and neural networks.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Implement kNN for both classification and regression.
- Tune the hyperparameter k and understand the effect of distance metrics and feature scaling.

## Key Terms

**k-Nearest Neighbors (kNN):** A non-parametric algorithm that classifies a new point by majority vote of its k closest training points (classification) or their average target value (regression).

**Instance-based learning:** A learning paradigm where the model stores training examples directly and defers computation to prediction time. kNN is the canonical instance-based learner.

**Euclidean distance:** The straight-line distance between two points: `d(a,b) = √(Σ(aᵢ−bᵢ)²)`. The default distance metric in scikit-learn's kNN.

**Manhattan distance:** The sum of absolute differences: `d(a,b) = Σ|aᵢ−bᵢ|`. Less sensitive to outliers in individual dimensions than Euclidean.

**Curse of dimensionality:** In high-dimensional spaces, all points become approximately equidistant from each other, making nearest-neighbor concepts meaningless. kNN degrades sharply in very high dimensions.

**Lazy learner:** An algorithm that does no generalization at training time — it simply memorizes the training data. All computation happens at prediction time.

**Weighted kNN:** A variant where closer neighbors contribute more to the vote, weighted by the inverse of their distance.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/02_knn_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## How kNN Works

kNN has the simplest possible training step: **store all training examples**.

At prediction time for a new point `x`:
1. Compute the distance from `x` to every training point
2. Find the k training points with the smallest distances
3. **Classification:** return the majority class among those k neighbors
4. **Regression:** return the mean target value among those k neighbors

```
Training data:
  ■ = malignant  ● = benign

         ●  ●
       ●   ●  ●
     ■ ● ■ ●  ●
       ■ ■  ●
         ■

New point ★: k=1 → look at 1 nearest → benign
             k=5 → look at 5 nearest → 3 benign, 2 malignant → benign
             k=9 → look at 9 nearest → 5 benign, 4 malignant → benign
```

The value of k determines how much of the local neighborhood influences the prediction.

## Effect of k: The Bias-Variance Trade-off

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Sweep k from 1 to 50
k_values = range(1, 51)
train_accs, test_accs = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    train_accs.append(knn.score(X_train_s, y_train))
    test_accs.append(knn.score(X_test_s,   y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_values, train_accs, label="Train accuracy", color="steelblue")
plt.plot(k_values, test_accs,  label="Test accuracy",  color="coral")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("kNN: Accuracy vs. k")
plt.legend()
plt.tight_layout()
plt.show()

best_k    = k_values[np.argmax(list(test_accs))]
best_acc  = max(test_accs)
print(f"Best k: {best_k},  Test accuracy: {best_acc:.3f}")
```

Output:
```
Best k: 7,  Test accuracy: 0.974
```

The pattern:
- **k=1:** Perfect training accuracy (each point is its own nearest neighbor) but high variance — the model memorizes every example including noise. **Overfitting.**
- **k=50+:** Very smooth decision boundary, but too much averaging smooths out real signal. **Underfitting.**
- **k≈7:** Best test accuracy for this dataset.

## Why Feature Scaling Is Critical for kNN

kNN measures distance between points. Features on different scales dominate the distance calculation:

```python
# Unscaled kNN
knn_unscaled = KNeighborsClassifier(n_neighbors=7)
knn_unscaled.fit(X_train, y_train)   # no scaling
acc_unscaled = knn_unscaled.score(X_test, y_test)

# Scaled kNN
knn_scaled = KNeighborsClassifier(n_neighbors=7)
knn_scaled.fit(X_train_s, y_train)   # StandardScaler
acc_scaled = knn_scaled.score(X_test_s, y_test)

print(f"kNN (k=7) unscaled:  {acc_unscaled:.3f}")
print(f"kNN (k=7) scaled:    {acc_scaled:.3f}")
```

Output:
```
kNN (k=7) unscaled:  0.921
kNN (k=7) scaled:    0.974
```

The 5% accuracy gap comes entirely from scaling. The breast cancer dataset has features like `mean area` (ranging ~140–2501) and `mean smoothness` (ranging ~0.05–0.16). Without scaling, area differences of 100 sq pixels completely dominate smoothness differences of 0.01 — the model effectively ignores smoothness.

**Rule: always scale features before using kNN.**

## Distance Metrics

The default is Euclidean distance. Manhattan distance (`p=1`) is an alternative:

```python
from sklearn.neighbors import KNeighborsClassifier

for metric, p in [("Euclidean", 2), ("Manhattan", 1)]:
    knn = KNeighborsClassifier(n_neighbors=7, p=p)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f"k=7, {metric} distance:  {acc:.3f}")
```

Output:
```
k=7, Euclidean distance:  0.974
k=7, Manhattan distance:  0.974
```

For this dataset, both give the same result. In practice, Euclidean is the default and works well for most standardized datasets. Manhattan can be preferable for high-dimensional data with many sparse features.

## kNN for Regression

kNN extends naturally to regression — predict the mean of the k nearest neighbors' target values:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)
Xh = housing.frame.drop(columns=["MedHouseVal"])
yh = housing.frame["MedHouseVal"]

Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)

scaler_h = StandardScaler()
Xh_tr_s = scaler_h.fit_transform(Xh_tr)
Xh_te_s = scaler_h.transform(Xh_te)

for k in [3, 7, 15, 30]:
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(Xh_tr_s, yh_tr)
    r2 = r2_score(yh_te, knn_reg.predict(Xh_te_s))
    print(f"k={k:3d}  R²={r2:.3f}")
```

Output:
```
k=  3  R²=0.685
k=  7  R²=0.703
k= 15  R²=0.692
k= 30  R²=0.659
```

kNN regression (R²≈0.70) is competitive with linear regression (R²=0.576) and even polynomial regression (R²=0.686) from module 11 on the California Housing dataset — without any feature engineering, simply by averaging nearby training examples.

## Full Evaluation on Breast Cancer

```python
from sklearn.metrics import classification_report

best_knn = KNeighborsClassifier(n_neighbors=7)
best_knn.fit(X_train_s, y_train)
y_pred = best_knn.predict(X_test_s)
y_proba = best_knn.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred,
      target_names=cancer.target_names))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
              precision    recall  f1-score   support

   malignant       0.974      0.953      0.963        43
      benign       0.974      0.986      0.980        71

    accuracy                           0.974       114
   macro avg       0.974      0.969      0.971       114
weighted avg       0.974      0.974      0.974       114

AUC: 0.993
```

kNN (k=7) achieves 97.4% accuracy and recall of 95.3% on malignant — better than the decision tree (93.0%) and matching logistic regression's AUC of 0.993.

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Zero training time — just store the data | Slow prediction: O(n × d) per sample |
| No assumptions about data distribution | Memory-intensive: must store all training data |
| Naturally handles non-linear boundaries | Degrades in high dimensions (curse of dimensionality) |
| Immediate adaptation to new training data | Sensitive to irrelevant features |
| Works for both classification and regression | No feature importances or interpretable model |

## When to Use kNN

**Use kNN when:**
- Dataset is small to medium (< 50,000 samples)
- You need a quick non-parametric baseline
- The local structure of the data is meaningful (e.g., recommendation systems where "similar users" is a sensible concept)
- Training time is unlimited but prediction latency isn't critical

**Avoid kNN when:**
- Dataset is large (prediction time grows linearly with training size)
- Features are high-dimensional and mostly irrelevant (use SVM or tree-based models instead)
- Prediction latency is critical (kNN requires scanning all training points)
- You need to explain individual predictions

## Conclusion

kNN is one of the most transparent algorithms in machine learning — it's easy to explain ("we looked at the 7 most similar patients in our training data, 6 of them were benign, so we predict benign"). On the breast cancer dataset, kNN with k=7 and standard scaling achieves 97.4% accuracy and AUC 0.993, matching logistic regression. The critical lesson here is scaling: unscaled kNN drops to 92.1%, a 5% penalty for one missing preprocessing step. In the next lesson, you'll learn **Support Vector Machines** — which share kNN's focus on geometry but find the single most informative boundary rather than looking at all neighbors.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/02_knn_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="With k=1, a new point is classified by its single nearest neighbor — which is always itself when predicting on training data, giving 100% training accuracy. On the test set, the model relies on whichever training point happens to be closest, which may not be representative of the true class boundary. Larger k smooths out this sensitivity by averaging over multiple neighbors, reducing variance at a small cost in bias.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A kNN classifier (k=1) achieves 100% accuracy on the training set but 72% on the test set. What is happening and how would you fix it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>The model is underfitting — increase k to allow the model to learn more patterns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The model is overfitting — k=1 makes each training point its own nearest neighbor, so it memorizes all training labels perfectly but doesn't generalize. Increase k (e.g., try values from 5 to 30 with cross-validation) to smooth the decision boundary.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The model needs more training data — 100% training accuracy means the dataset is too small.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>kNN cannot overfit because it has no parameters to tune.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="A $10,000 income difference contributes (10,000)² = 10⁸ to the squared Euclidean distance. A 10-year age difference contributes only 100. Without scaling, income differences are a billion times more influential than age differences in the distance calculation. If age is actually an important predictor, the model will perform poorly. `StandardScaler` maps both features to mean=0, std=1, ensuring each feature contributes proportionally to distance.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You apply kNN (k=5) to a dataset with features `income` (range $20,000–$200,000) and `age` (range 18–90) without scaling. What is the likely effect on predictions?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>No effect — kNN computes the majority vote which is scale-invariant.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Age will dominate the distance calculation because it has a larger range.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Income will dominate the distance calculation because it has a much larger range. The model will effectively ignore age, finding "nearest neighbors" based almost entirely on income similarity.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The algorithm will automatically normalize features internally.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="&quot;Lazy&quot; refers to deferred generalization: logistic regression eagerly summarizes the training data into a set of weights at training time, then uses only those weights (not the training data) for prediction. kNN stores every training point and consults them all at prediction time. With 1 million training samples and 100 features, each prediction requires 100 million distance calculations. This makes kNN impractical for large-scale production systems unless approximate nearest-neighbor structures (e.g., KD-trees, ball trees) are used.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Why is kNN called a "lazy learner," and what are the computational consequences of this approach?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>kNN is called lazy because it uses heuristic rules rather than optimization, which makes it inaccurate.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>kNN is lazy because it stores all training data without learning any explicit model at training time. The consequence is zero training cost but O(n × d) prediction cost per sample, where n is training set size and d is the number of features — making it slow for large datasets at inference time.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>kNN is lazy because it requires fewer hyperparameters than other algorithms.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>kNN is lazy because it uses default scikit-learn settings without any configuration.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

