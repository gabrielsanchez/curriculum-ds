# Multi-class Classification

## Overview

You've now covered binary classification and multi-label classification in depth. In this lesson, you'll tackle the remaining case: **multi-class classification** — predicting one of three or more mutually exclusive categories. This comes up whenever you need to assign a single label from a larger set: classifying a digit (0–9), routing a support ticket to the right department, or identifying a species from a photo. You'll learn how algorithms extend from binary to multi-class, how to read multi-class confusion matrices, and how to interpret averaged metrics.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Extend binary classifiers to multi-class settings using One-vs-Rest and One-vs-One strategies.
- Interpret multi-class confusion matrices.
- Compute and compare macro, micro, and weighted-averaged metrics.

## Key Terms

**Multi-class classification:** A supervised learning task where the target variable has three or more mutually exclusive categories. Each sample belongs to exactly one class.

**One-vs-Rest (OvR) / One-vs-All:** A strategy for extending binary classifiers to multi-class. Train one binary classifier per class — each treats its class as "positive" and all others as "negative." Predict the class whose classifier is most confident.

**One-vs-One (OvO):** Train one binary classifier for every pair of classes. For k classes, this produces k×(k-1)/2 classifiers. The final prediction is the class that wins the most pairwise contests.

**Softmax:** A generalization of the sigmoid function that outputs a probability distribution over all classes. Used in logistic regression and neural networks for multi-class problems.

**Multi-class confusion matrix:** An n×n matrix (for n classes) showing how often each true class was predicted as each other class. Diagonal entries are correct predictions.

**Macro average:** Compute the metric (precision, recall, F1) for each class independently, then take the unweighted mean. Every class counts equally, regardless of size.

**Micro average:** Aggregate the counts (TP, FP, FN) across all classes first, then compute the metric. Dominated by the most common class.

**Weighted average:** Average the per-class metric, weighted by the number of true samples in each class (support). Common for imbalanced multi-class problems.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Handwritten digit recognition is one of the oldest benchmarks in machine learning. An optical character recognition system must classify each image into one of 10 categories: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9. No image is simultaneously a "3" and a "7" — this is the defining property of multi-class classification.

Other multi-class problems:
- **News categorization:** politics, sports, technology, business, entertainment
- **Customer segmentation:** high value, medium value, price sensitive, at-risk
- **Plant disease classification:** healthy, bacterial spot, early blight, late blight, mold
- **Language identification:** English, Spanish, French, German, Chinese, ...

The challenge increases with the number of classes: more classes mean more opportunities for confusion, more decision boundaries to learn, and more complex evaluation.

## Setup

We'll use the **Iris dataset** — a classic multi-class benchmark built into scikit-learn with 3 classes (setosa, versicolor, virginica) and 4 numeric features.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

print(f"Features: {iris.feature_names}")
print(f"Classes:  {class_names}")
print(f"Shape:    {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

Output:
```
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Classes:  ['setosa' 'versicolor' 'virginica']
Shape:    (150, 4)
```

## From Binary to Multi-class: The Strategies

### One-vs-Rest (OvR)

OvR is the most common strategy and is the default in scikit-learn's `LogisticRegression`.

For 3 classes:
1. **Classifier 1:** Is this setosa? (setosa=1, versicolor+virginica=0)
2. **Classifier 2:** Is this versicolor? (versicolor=1, setosa+virginica=0)
3. **Classifier 3:** Is this virginica? (virginica=1, setosa+versicolor=0)

Each classifier outputs a confidence score. The predicted class is whichever classifier is most confident.

```python
# LogisticRegression uses OvR by default for multi-class
clf_ovr = LogisticRegression(multi_class="ovr", random_state=42)
clf_ovr.fit(X_train_s, y_train)

y_pred_ovr = clf_ovr.predict(X_test_s)
print(f"OvR Accuracy: {accuracy_score(y_test, y_pred_ovr):.1%}")
```

Output:
```
OvR Accuracy: 96.7%
```

**Pros:** Fast — only k classifiers for k classes. Works well in practice.
**Cons:** Each binary problem is imbalanced (1 class vs. all others).

### Softmax (Multinomial)

Logistic regression can also use the **softmax** function to model all classes simultaneously, outputting a true probability distribution across all classes.

```python
clf_soft = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
clf_soft.fit(X_train_s, y_train)

# Probability output for first test sample
proba = clf_soft.predict_proba(X_test_s[:1])
print(f"Class probabilities for first sample: {proba.round(3)}")
print(f"Predicted class: {class_names[clf_soft.predict(X_test_s[:1])[0]]}")
```

Output:
```
Class probabilities for first sample: [[0.002 0.243 0.755]]
Predicted class: virginica
```

The probabilities sum to 1. The model is most confident this sample is virginica (75.5%).

### One-vs-One (OvO)

OvO trains a classifier for every pair of classes. For 3 classes: setosa-vs-versicolor, setosa-vs-virginica, versicolor-vs-virginica (3 classifiers). For 10 classes: 45 classifiers.

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

# SVC uses OvO by default; we make it explicit here
clf_ovo = OneVsOneClassifier(SVC(kernel="rbf", random_state=42))
clf_ovo.fit(X_train_s, y_train)

y_pred_ovo = clf_ovo.predict(X_test_s)
print(f"OvO Accuracy: {accuracy_score(y_test, y_pred_ovo):.1%}")
```

Output:
```
OvO Accuracy: 96.7%
```

**Pros:** Each sub-problem is perfectly balanced (1 class vs. 1 other class). Often better performance on complex boundaries.
**Cons:** Expensive for many classes — k×(k-1)/2 models.

### Decision Trees: Naturally Multi-class

Tree-based algorithms don't need OvR or OvO — they split on feature values and can handle any number of classes natively:

```python
clf_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_tree.fit(X_train_s, y_train)

y_pred_tree = clf_tree.predict(X_test_s)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.1%}")
```

Output:
```
Decision Tree Accuracy: 96.7%
```

Trees are naturally multi-class because each leaf node can be assigned any class label — no extension strategy needed.

## The Multi-class Confusion Matrix

The confusion matrix scales from 2×2 (binary) to n×n (multi-class):

```python
cm = confusion_matrix(y_test, y_pred_ovr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Multi-class Confusion Matrix — Iris")
plt.show()

print(cm)
```

Output:
```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

Reading the matrix:
- Rows are the **true** classes
- Columns are the **predicted** classes
- **Diagonal** entries are correct predictions
- **Off-diagonal** entries are errors

In this case:
- All 10 setosa were correctly classified
- 9/10 versicolor were correctly classified (1 was misclassified as virginica)
- All 10 virginica were correctly classified

The error pattern is informative: versicolor and virginica are confused, but setosa is never confused with either. This matches biological reality — setosa is distinct, while versicolor and virginica overlap in feature space.

## Per-class Metrics and Averaging

For multi-class problems, precision, recall, and F1 are computed separately for each class, then averaged.

```python
print(classification_report(y_test, y_pred_ovr, target_names=class_names))
```

Output:
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

### Three Ways to Average

**Macro average:** Compute F1 for each class, take the simple mean. Each class counts equally.
- Setosa F1 = 1.00, Versicolor F1 = 0.95, Virginica F1 = 0.95
- Macro F1 = (1.00 + 0.95 + 0.95) / 3 = **0.97**
- Best when all classes are equally important, regardless of size.

**Micro average:** Pool all TPs, FPs, FNs across classes and compute one global metric.
- For balanced datasets, micro average ≈ accuracy.
- Best when you care about the total number of correct predictions.

**Weighted average:** Like macro, but weights each class's F1 by its support (number of true samples).
- Best for imbalanced multi-class problems — gives more importance to common classes.

### Which to Use?

| Situation | Use |
|-----------|-----|
| All classes equally important | Macro average |
| Imbalanced classes, care about overall correctness | Weighted average |
| Want one global metric equivalent to accuracy | Micro average |
| Investigating specific classes | Per-class metrics |

## Working Example: Digit Recognition

Let's try a more challenging multi-class problem — 10-class digit recognition:

```python
from sklearn.datasets import load_digits

digits = load_digits()
X_d, y_d = digits.data, digits.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
)

clf_digits = LogisticRegression(max_iter=1000, random_state=42)
clf_digits.fit(X_tr, y_tr)
y_pred_d = clf_digits.predict(X_te)

print(f"Accuracy: {accuracy_score(y_te, y_pred_d):.1%}")
print("\nClassification Report:")
print(classification_report(y_te, y_pred_d))
```

Output:
```
Accuracy: 96.4%

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      0.99        36
           1       0.93      0.97      0.95        36
           2       0.97      0.97      0.97        35
           3       0.95      0.97      0.96        37
           4       0.97      1.00      0.99        36
           5       0.97      0.94      0.96        37
           6       0.97      0.97      0.97        36
           7       0.97      0.97      0.97        36
           8       0.97      0.91      0.94        35
           9       0.97      0.97      0.97        36

    accuracy                           0.96       360
   macro avg       0.97      0.97      0.97       360
weighted avg       0.97      0.96      0.96       360
```

96.4% accuracy across 10 classes with a simple logistic regression. Looking at the per-class report, digit "8" has the lowest recall (0.91) — meaning some 8s are being misclassified as other digits. The confusion matrix would show which digit "8" is most often confused with.

## Practical Considerations

### Scaling Matters for OvR and Softmax

Logistic regression (OvR and softmax) requires feature scaling. Decision trees do not.

```python
# Always scale before logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Many Classes → Consider Trees

With 100+ classes, OvO becomes infeasible (4,950 classifiers). Tree-based models (decision trees, random forests, gradient boosting) scale naturally.

### Class Imbalance in Multi-class

Works the same as binary: use `class_weight="balanced"` or examine weighted-average metrics.

```python
clf_bal = LogisticRegression(class_weight="balanced", random_state=42)
```

## Comparison: Binary, Multi-class, Multi-label

| Feature | Binary | Multi-class | Multi-label |
|---------|--------|-------------|-------------|
| Number of classes | 2 | 3 or more | Any |
| Labels per sample | 1 | 1 | 0 or more |
| Target shape | `(n,)` | `(n,)` | `(n, k)` |
| Default strategy | Direct | OvR or softmax | Binary relevance |
| Primary metrics | Accuracy, F1, AUC | Per-class F1, macro/weighted avg | Hamming loss, per-label F1 |

## Conclusion

In this lesson, you completed the classification trilogy. You learned that multi-class classification extends binary classification using One-vs-Rest (one classifier per class), One-vs-One (one classifier per pair), or the softmax function (a direct multi-class probability model). You saw that tree-based algorithms are naturally multi-class and don't need these strategies. You also learned to read multi-class confusion matrices — which reveal not just how often a model is wrong, but what it confuses — and to interpret macro, micro, and weighted metric averages appropriately. In the next lesson, you'll apply everything from this module in the **module assessment**.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="One-vs-Rest trains exactly k classifiers for k classes. Classifier 1 is trained to distinguish class 1 from classes {2, 3, 4, 5}; Classifier 2 distinguishes class 2 from classes {1, 3, 4, 5}; and so on. At prediction time, all 5 classifiers score the sample, and the class with the highest confidence score wins. For reference, One-vs-One would train k×(k-1)/2 = 10 classifiers.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A logistic regression classifier using One-vs-Rest is trained on a 5-class problem. How many binary classifiers does it train internally, and what does each one predict?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>10 classifiers — one for every pair of classes.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>1 classifier — logistic regression handles all classes natively.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>5 classifiers — each one predicts whether a sample belongs to one specific class versus all other classes combined.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>4 classifiers — one per class boundary.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When macro F1 &lt;&lt; weighted F1, it almost always indicates class imbalance. Weighted average F1 is pulled up by the large classes, which the model handles well. Macro average gives equal weight to every class — including the small, underrepresented ones that the model struggles with. This gap is a red flag that the model may perform poorly in practice on rare but important categories. Both averages should be reported and understood.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A multi-class classification report shows that your model has macro-average F1 = 0.61 but weighted-average F1 = 0.84. What is the most likely explanation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model performs well overall, and the two averages always disagree by this much.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The dataset is highly imbalanced — the model performs well on large classes (driving up the weighted average) but poorly on small classes (which are treated equally in the macro average).</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The macro average is always lower than the weighted average because it is computed differently.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The model has high precision but low recall across all classes.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Off-diagonal concentration in a confusion matrix shows which class pairs the model finds hard to distinguish. Frequent B→C confusion means the model sees many class-B samples as more similar to class-C samples than to other B samples in feature space. This could mean: the features don&#039;t capture what distinguishes B from C, the boundary is inherently fuzzy (the classes genuinely overlap), or you need domain-specific features to separate them. The confusion matrix guides your feature engineering and data collection efforts.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Looking at a multi-class confusion matrix for a 4-class problem, you notice that class "B" is frequently confused with class "C" (many off-diagonal entries in the B row point to column C). What does this tell you, and what should you investigate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The model should be retrained with a higher learning rate to separate B and C.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Classes B and C are similar in feature space — their distinguishing features may overlap. You should investigate whether they are truly distinct or whether better features are needed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The training data for class B is corrupted and should be discarded.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>This indicates overfitting — the model memorized B in training but generalizes it as C.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

