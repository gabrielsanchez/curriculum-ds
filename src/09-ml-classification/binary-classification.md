# Binary Classification

## Overview

In the previous lesson, you set up the classification pipeline and built your first classifier in a few lines of code. In this lesson, you'll go deep on **binary classification** — predicting one of two possible outcomes. You'll learn the primary algorithm (logistic regression), how to interpret its output as a probability, and — most importantly — how to measure whether your classifier is actually good using a comprehensive set of evaluation metrics.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Train and interpret a logistic regression classifier.
- Evaluate binary classifiers using accuracy, precision, recall, F1-score, and ROC-AUC.
- Select the right metric for the problem at hand.

## Key Terms

**Logistic regression:** A classification algorithm that models the probability that a sample belongs to the positive class using the sigmoid function. Despite the word "regression," it is used for classification.

**Sigmoid function:** A mathematical function that maps any real number to a value between 0 and 1. Used by logistic regression to output a probability.

**Threshold:** The probability cutoff used to convert a probability score into a class label. The default is 0.5: if `P(positive) ≥ 0.5`, predict positive.

**Confusion matrix:** A table that summarizes the four possible outcomes of a binary classifier: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).

**Accuracy:** The fraction of all predictions that were correct. `(TP + TN) / (TP + TN + FP + FN)`. Misleading when classes are imbalanced.

**Precision:** Of all samples predicted positive, what fraction actually were positive? `TP / (TP + FP)`. High precision means few false alarms.

**Recall (Sensitivity):** Of all samples that are actually positive, what fraction did the model correctly identify? `TP / (TP + FN)`. High recall means few missed positives.

**F1-score:** The harmonic mean of precision and recall. `2 × (Precision × Recall) / (Precision + Recall)`. Balances the two when you care about both.

**ROC curve:** Receiver Operating Characteristic curve — a plot of True Positive Rate vs. False Positive Rate across all possible thresholds.

**AUC:** Area Under the ROC Curve. A single number (0 to 1) summarizing overall classifier performance across all thresholds. A random classifier scores 0.5; a perfect classifier scores 1.0.

**Class imbalance:** When one class is much more common than the other (e.g., 99% negative, 1% positive). Requires care in choosing metrics and potentially in training.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Binary classification is the most common type of ML problem in industry. Nearly every "should we or shouldn't we?" business question is a binary classification problem:

- Should we approve this loan?
- Will this customer churn?
- Is this transaction fraudulent?
- Does this scan show a tumor?

A key insight: **not all mistakes are equal.** In fraud detection, letting a fraud through (false negative) might cost the company thousands of dollars. Flagging a legitimate transaction as fraud (false positive) inconveniences a customer. These costs are different — and the right metric depends on which error matters more in your domain.

In this lesson, you'll use the Titanic survival dataset (familiar from module 07) to illustrate every concept, so you can focus on the methods rather than the data.

## Setup

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

# Load and clean
df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].dropna()
df["sex"] = df["sex"].map({"male": 0, "female": 1})

X = df.drop(columns="survived")
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

## Logistic Regression

### The Core Idea

Linear regression predicts a continuous value. To predict a probability (which must be between 0 and 1), logistic regression passes a linear combination of features through the **sigmoid function**:

```
P(y=1 | X) = 1 / (1 + e^(-(w₀ + w₁x₁ + w₂x₂ + ...)))
```

The sigmoid stretches any real number to the range (0, 1):

```
Input:   -5    -2    0    2    5
Output: 0.007  0.12  0.5  0.88  0.993
```

Inputs far below 0 produce probabilities close to 0. Inputs far above 0 produce probabilities close to 1. The model learns the weights (`w`) during training to maximize the separation between classes.

### Training and Predicting

```python
clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)

# Hard predictions (class labels: 0 or 1)
y_pred = clf.predict(X_test_scaled)

# Probability estimates
y_proba = clf.predict_proba(X_test_scaled)
# y_proba has two columns: P(survived=0) and P(survived=1)
print(y_proba[:5])
```

Output:
```
[[0.74  0.26]
 [0.15  0.85]
 [0.63  0.37]
 [0.52  0.48]
 [0.34  0.66]]
```

The model assigns probabilities to each class. By default, it predicts the class with the higher probability — equivalent to using 0.5 as the threshold.

### Interpreting Coefficients

```python
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": clf.coef_[0]
}).sort_values("coefficient")

print(coef_df)
```

Output (approximate):
```
  feature  coefficient
     pclass      -0.93   ← higher class number (3rd class) → lower survival
      sibsp      -0.26
      parch      -0.08
        age      -0.40
       fare       0.21
        sex       1.12   ← female (=1) → much higher survival probability
```

Positive coefficients increase the probability of survival; negative decrease it. This interpretability is one of logistic regression's strengths.

## The Confusion Matrix

The confusion matrix is the foundation of all binary classification metrics. It shows the four possible prediction outcomes:

```
                    Predicted
                 Negative  Positive
Actual Negative    TN        FP
Actual Positive    FN        TP
```

- **True Negative (TN):** Predicted negative, actually negative. Correct.
- **False Positive (FP):** Predicted positive, actually negative. Wrong — a "false alarm."
- **False Negative (FN):** Predicted negative, actually positive. Wrong — a "miss."
- **True Positive (TP):** Predicted positive, actually positive. Correct.

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not survive", "Survived"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression — Titanic")
plt.show()

print(cm)
```

Output:
```
[[71  14]
 [18  40]]

TN=71  FP=14
FN=18  TP=40
```

From this matrix, we can compute every metric.

## Evaluation Metrics

### Accuracy

```
Accuracy = (TP + TN) / Total = (40 + 71) / 143 = 77.6%
```

```python
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
```

**When accuracy fails:** Suppose 95% of transactions are legitimate. A model that always predicts "legitimate" achieves 95% accuracy while catching zero fraud cases. Accuracy hides this because the dominant class overwhelms the calculation.

### Precision and Recall

These metrics focus on the positive class — typically the rarer, more important outcome.

```
Precision = TP / (TP + FP) = 40 / (40 + 14) = 74.1%
Recall    = TP / (TP + FN) = 40 / (40 + 18) = 69.0%
```

```python
print(f"Precision: {precision_score(y_test, y_pred):.1%}")
print(f"Recall:    {recall_score(y_test, y_pred):.1%}")
```

**Precision answers:** "When I predict positive, how often am I right?" High precision means few false alarms.

**Recall answers:** "Of all actual positives, how many did I find?" High recall means few missed cases.

### The Precision-Recall Trade-off

Precision and recall are in tension: you can almost always increase one by decreasing the other.

- **Lower the threshold (e.g., 0.3):** More positives predicted → more TPs and more FPs → recall increases, precision decreases.
- **Raise the threshold (e.g., 0.7):** Fewer positives predicted → fewer FPs but also fewer TPs → precision increases, recall decreases.

Which to optimize depends on your domain:

| Domain | Optimize for | Reason |
|--------|-------------|--------|
| Cancer screening | Recall | Missing a cancer (FN) is far worse than a false alarm |
| Email spam filter | Precision | Deleting a real email (FP) is worse than letting spam through |
| Fraud detection | Recall | Missing fraud costs more than a false block |
| Resume screening | Precision | Wasting interviews (FP) is costly at scale |

### F1-Score

The F1-score is the harmonic mean of precision and recall. It's useful when you want a single number that balances both:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.741 × 0.690) / (0.741 + 0.690)
   = 71.4%
```

```python
print(f"F1-score: {f1_score(y_test, y_pred):.1%}")
```

The harmonic mean penalizes extreme imbalances between precision and recall. A model with precision=0.99 and recall=0.01 gets an F1 of only 2%, signaling it's useless despite high precision.

### Complete Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))
```

Output:
```
                 precision    recall  f1-score   support

Did not survive       0.80      0.84      0.82        85
       Survived       0.74      0.69      0.71        58

       accuracy                           0.78       143
      macro avg       0.77      0.76      0.77       143
   weighted avg       0.77      0.78      0.77       143
```

The report shows metrics for each class separately. This is much more informative than a single accuracy number.

## ROC Curve and AUC

The ROC curve shows how the trade-off between true positive rate (recall) and false positive rate changes as you vary the prediction threshold from 0 to 1.

```python
y_proba_positive = clf.predict_proba(X_test_scaled)[:, 1]  # P(survived=1)

fpr, tpr, thresholds = roc_curve(y_test, y_proba_positive)
auc = roc_auc_score(y_test, y_proba_positive)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

print(f"AUC: {auc:.3f}")
```

Output:
```
AUC: 0.841
```

**Interpreting AUC:**
- **AUC = 1.0** — perfect classifier; all positives rank higher than all negatives
- **AUC = 0.5** — no better than random guessing
- **AUC = 0.84** — the model correctly ranks a random positive above a random negative 84% of the time

AUC is threshold-independent, making it useful for comparing classifiers without committing to a specific threshold.

## Handling Class Imbalance

Real datasets are often imbalanced — one class is far more common than the other.

```python
print(y.value_counts(normalize=True))
```

Output:
```
0    0.594    ← 59% did not survive
1    0.406    ← 41% survived
```

The Titanic dataset has mild imbalance. More extreme imbalance requires special treatment:

### Strategy 1: Use the Right Metric

Stop using accuracy. Use F1-score, AUC, or precision-recall AUC instead.

### Strategy 2: Adjust Class Weights

Most scikit-learn classifiers accept `class_weight="balanced"`, which upweights the minority class during training:

```python
clf_balanced = LogisticRegression(class_weight="balanced", random_state=42)
clf_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = clf_balanced.predict(X_test_scaled)

print(f"Recall (balanced):  {recall_score(y_test, y_pred_balanced):.1%}")
print(f"Recall (standard):  {recall_score(y_test, y_pred):.1%}")
```

The balanced model typically improves recall at the cost of some precision. Whether that trade-off is worth it depends on your use case.

### Strategy 3: Adjust the Decision Threshold

Instead of using 0.5, lower the threshold to catch more positives:

```python
# Lower threshold to 0.35 to increase recall
y_pred_low_thresh = (y_proba_positive >= 0.35).astype(int)

print(f"Precision @ 0.35: {precision_score(y_test, y_pred_low_thresh):.1%}")
print(f"Recall    @ 0.35: {recall_score(y_test, y_pred_low_thresh):.1%}")
```

## Putting It All Together

Here is the full evaluation workflow:

```python
from sklearn.metrics import classification_report, roc_auc_score

# Train
clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred  = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
=== Classification Report ===
                 precision    recall  f1-score   support

Did not survive       0.80      0.84      0.82        85
       Survived       0.74      0.69      0.71        58

       accuracy                           0.78       143
      macro avg       0.77      0.76      0.77       143
   weighted avg       0.77      0.78      0.77       143

AUC: 0.841
```

## Metric Selection Guide

| Situation | Recommended metric |
|-----------|-------------------|
| Balanced classes, equal error cost | Accuracy or F1 |
| Imbalanced classes | F1, AUC, or Precision-Recall AUC |
| Missing positives is very costly (cancer, fraud) | Recall |
| False alarms are very costly (spam, loan denial) | Precision |
| Comparing models without fixing a threshold | AUC |
| Reporting to stakeholders | Accuracy (simplest to explain) + F1 |

## Conclusion

In this lesson, you learned how logistic regression works — modeling class probability through the sigmoid function — and why it's the standard starting point for binary classification. More importantly, you built a full evaluation toolkit: the confusion matrix, accuracy, precision, recall, F1-score, and AUC. You saw that accuracy alone is misleading with imbalanced classes, and you learned how to use `class_weight="balanced"` and threshold adjustment to tune the trade-off between precision and recall for your specific domain. In the next lesson, you'll explore **multi-label classification**, where each sample can simultaneously belong to multiple classes.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](#). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

#### **Question 1: A fraud detection model catches 95% of all fraudulent transactions but incorrectly flags 30% of legitimate transactions. Which metrics best describe this situation?**

1. The model has high accuracy and high precision.
2. The model has high recall (95%) but low precision (many false positives from flagging 30% of legitimate transactions).
3. The model has high precision and low recall.
4. Accuracy is the best metric to report here since it combines all outcomes.

**Correct Answer:**
2. The model has high recall (95%) but low precision (many false positives from flagging 30% of legitimate transactions).

**Explanation:**
Recall measures how many actual positives are correctly identified — catching 95% of fraud is high recall. Precision measures how many predicted positives are truly positive — flagging 30% of legitimate transactions means many false positives, which corresponds to low precision. Accuracy would be misleading here because it includes the large number of correct negatives, masking how many legitimate transactions are being wrongly flagged.

---

#### **Question 2: A logistic regression model is trained to predict hospital readmission within 30 days. You want to catch as many at-risk patients as possible, even at the cost of some unnecessary follow-up calls. Which action best achieves this?**

1. Raise the prediction threshold from 0.5 to 0.7.
2. Optimize for precision using `class_weight="balanced"`.
3. Lower the prediction threshold from 0.5 to 0.3 to predict more patients as high-risk.
4. Remove low-probability samples from the training set.

**Correct Answer:**
3. Lower the prediction threshold from 0.5 to 0.3 to predict more patients as high-risk.

**Explanation:**
Lowering the threshold means the model predicts "high risk" whenever `P(readmission) ≥ 0.3` instead of `≥ 0.5`. This flags more patients — including more true positives (improving recall) but also more false positives (unnecessary calls). Since the goal is to catch as many at-risk patients as possible (high recall), at the acceptable cost of some false alarms, lowering the threshold is the right lever. Raising the threshold does the opposite.

---

#### **Question 3: Two classifiers are compared on a fraud detection dataset (1% fraud, 99% legitimate). Classifier A has 99.1% accuracy. Classifier B has 72% accuracy but an AUC of 0.94. Which should you prefer, and why?**

1. Classifier A, because higher accuracy always means a better classifier.
2. Classifier B, because AUC of 0.94 indicates it separates fraudulent from legitimate transactions far better than chance, while Classifier A may simply be predicting "legitimate" for every sample.
3. Classifier A, because accuracy is the standard metric and 99.1% is excellent.
4. Neither — both classifiers fail because the AUC should equal the accuracy.

**Correct Answer:**
2. Classifier B, because AUC of 0.94 indicates it separates fraudulent from legitimate transactions far better than chance, while Classifier A may simply be predicting "legitimate" for every sample.

**Explanation:**
With 99% of cases being legitimate, a trivial classifier that predicts "legitimate" for everything achieves 99% accuracy while catching zero fraud — useless. Classifier A's 99.1% accuracy is suspiciously close to this naive baseline. Classifier B's AUC of 0.94 means it correctly ranks a random fraudulent transaction above a random legitimate one 94% of the time, demonstrating genuine discriminative power. On highly imbalanced datasets, AUC (or precision-recall metrics) are far more informative than accuracy.
