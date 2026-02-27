# Introduction to Classification

## Overview

In the previous modules, you built the full data preparation toolkit — cleaning data, engineering features, and exploring it visually. In module 07, you learned that supervised learning divides into two families: **regression** (predicting a continuous number) and **classification** (predicting a discrete category). This module focuses entirely on classification. You've already seen this concept at a high level; now you'll learn how to actually build classifiers, evaluate them rigorously, and apply the right algorithm to the right problem.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Define classification and distinguish it from regression.
- Identify binary, multi-class, and multi-label classification problems.
- Describe the end-to-end classification pipeline in scikit-learn.

## Key Terms

**Classification:** A supervised learning task in which the goal is to assign each input to one of a fixed set of discrete categories (classes).

**Class:** One of the possible output categories. In spam detection, the classes are "spam" and "not spam."

**Binary classification:** A classification problem with exactly two possible output classes.

**Multi-class classification:** A classification problem with three or more mutually exclusive output classes (each sample belongs to exactly one class).

**Multi-label classification:** A classification problem where each sample can simultaneously belong to multiple classes (e.g., a movie can be both "action" and "comedy").

**Decision boundary:** The surface in feature space that separates one class from another. A classifier's predictions are determined by which side of the decision boundary a new point falls on.

**Probability score:** Instead of a hard label, many classifiers output a probability that a sample belongs to each class. The label is then the class with the highest probability (or, for binary classification, whichever class exceeds a threshold like 0.5).

**Stratification:** Ensuring that the class distribution in the training and test sets mirrors the class distribution in the full dataset. Applied via `stratify=y` in `train_test_split`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/09-ml-classification/01_introduction-to-classification_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Imagine you're building a system to filter emails. You don't want to predict a number — you want to predict a **category**: is this email spam, or not? That's classification.

Classification problems are everywhere:

- A doctor reviews a scan: **malignant** or **benign**?
- A bank reviews a transaction: **fraudulent** or **legitimate**?
- A streaming service evaluates a new user: **will churn** or **will stay**?
- A wildlife camera captures an image: **wolf**, **deer**, **bear**, or **fox**?

What these problems share is a **discrete, finite set of possible answers**. You're not predicting *how much* — you're predicting *which one*.

## Classification vs. Regression: A Reminder

| Question | Type | Example Target |
|----------|------|----------------|
| How much will this house sell for? | Regression | $435,000 |
| Will this customer cancel their subscription? | Classification | yes / no |
| What species is this plant? | Classification | oak / pine / maple |
| How many units will we sell next month? | Regression | 12,400 |

The rule of thumb: if you can naturally ask "which category?", it's classification. If you'd naturally ask "how much?" or "how many?", it's regression.

## Three Types of Classification

### Binary Classification

Two possible outputs. One is often called the **positive class** (the outcome you're most interested in detecting), and the other the **negative class**.

| Problem | Positive class | Negative class |
|---------|---------------|----------------|
| Spam detection | spam | not spam |
| Fraud detection | fraud | legitimate |
| Medical test | disease present | disease absent |
| Customer churn | will churn | will stay |

Binary classification is the most common type. Most classification algorithms are designed for the binary case first, with extensions to handle more classes.

### Multi-class Classification

Three or more mutually exclusive categories. Each sample belongs to **exactly one** class.

Examples:
- Handwritten digit recognition: 0 through 9 (10 classes)
- News article topic: politics, sports, technology, business, entertainment
- Product category: electronics, clothing, books, furniture, food
- Plant disease: healthy, bacterial_spot, early_blight, late_blight, leaf_mold

The key word is **mutually exclusive** — a digit is either a 3 or a 7, never both.

### Multi-label Classification

Each sample can have **multiple labels at the same time**. This is structurally different from both binary and multi-class classification.

Examples:
- A movie can be simultaneously "action", "comedy", and "thriller"
- A news article can cover "politics" and "economics" at the same time
- A medical image can show signs of multiple conditions

Multi-label is covered in its own dedicated lesson. For now, understand that it exists and is different from multi-class.

## The Classification Pipeline

Every classification project in scikit-learn follows the same sequence:

```
1. Load and explore your data
        ↓
2. Split into features (X) and target (y)
        ↓
3. Split into train/test sets (with stratification)
        ↓
4. Preprocess features (encode categoricals, scale numerics)
        ↓
5. Choose and train a classifier
        ↓
6. Evaluate on the test set
        ↓
7. Iterate (tune, try other algorithms)
```

Steps 1–4 are familiar from modules 07 and 08. In this module, you'll focus on steps 5 and 6.

## The scikit-learn Classifier API

Every classifier in scikit-learn follows the same three-method pattern you learned in module 07:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()     # 1. Instantiate
model.fit(X_train, y_train)          # 2. Train
y_pred = model.predict(X_test)       # 3. Predict labels

# Many classifiers also support probability estimates
y_proba = model.predict_proba(X_test)   # probabilities for each class
```

This consistent interface means you can swap one algorithm for another by changing a single line — the rest of your code stays identical.

## Your First Classifier

Let's build a complete classification pipeline end-to-end using the Titanic dataset (already familiar from module 07). The task: predict whether a passenger survived.

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].dropna()

# ── 2. Features and target ───────────────────────────────────────────────────
X = df.drop(columns="survived")
y = df["survived"]

# ── 3. Encode categorical column ─────────────────────────────────────────────
X["sex"] = X["sex"].map({"male": 0, "female": 1})

# ── 4. Train/test split — use stratify to preserve class ratio ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Survival rate in train: {y_train.mean():.1%}")
print(f"Survival rate in test:  {y_test.mean():.1%}")
```

Output:
```
Train: (571, 6), Test: (143, 6)
Survival rate in train: 40.6%
Survival rate in test:  40.6%
```

The `stratify=y` argument ensures both splits have the same class distribution — crucial when one class is rarer than another.

```python
# ── 5. Train a decision tree ──────────────────────────────────────────────────
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
```

Output:
```
Accuracy: 79.7%
```

You just built a working classifier in about 20 lines of code. But accuracy alone is rarely enough to judge a model — you'll learn a full battery of metrics in the next lesson.

## How Classifiers Make Decisions

Different algorithms draw the decision boundary in different ways:

| Algorithm | How it splits the space |
|-----------|------------------------|
| Logistic Regression | A single straight line (linear boundary) |
| Decision Tree | Axis-aligned rectangular splits |
| k-Nearest Neighbors | Regions defined by nearest training points |
| SVM | The widest possible margin between classes |
| Random Forest | Average of many decision trees |

No algorithm wins on every problem. The right choice depends on your data's structure, size, and the degree of interpretability you need. In later modules, you'll study each of these in depth.

## What Makes a Good Classifier?

A good classifier:
1. **Generalizes** — performs well on unseen data, not just the training set
2. **Is calibrated** — its probability scores reflect real probabilities (a 70% prediction is right about 70% of the time)
3. **Is appropriate for the problem** — considers the cost of different errors (a missed cancer diagnosis is worse than a false alarm)
4. **Can be explained** — ideally, you can justify why it made a prediction

Point 3 is especially important. **Accuracy** — the fraction of predictions that were correct — sounds like the obvious metric, but it can be deeply misleading. If 98% of emails are legitimate, a model that labels everything as "not spam" achieves 98% accuracy while being completely useless. The next lesson covers the right metrics for classification.

## Conclusion

In this lesson, you established the foundation for the classification module. You learned that classification assigns inputs to discrete categories, distinguished binary (two classes), multi-class (three or more exclusive classes), and multi-label (multiple simultaneous labels) problems, and traced the full classification pipeline from data loading to evaluation. You also ran a complete binary classifier on the Titanic dataset using scikit-learn's consistent three-method API. In the next lesson, you'll go deep on **binary classification** — the most common case — including logistic regression and the full suite of evaluation metrics.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="There are three possible output categories (pneumonia, tuberculosis, healthy), and each X-ray belongs to exactly one of them. That&#039;s the definition of multi-class classification. It would be binary if there were only two classes (e.g., pneumonia / not pneumonia). It would be multi-label if a single X-ray could show both pneumonia and tuberculosis simultaneously.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A hospital wants to build a model that reads a chest X-ray and predicts whether the patient has pneumonia, tuberculosis, or is healthy. What type of classification problem is this?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Binary classification — because the model outputs one label per image.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Multi-label classification — because the image shows a chest, which can have many features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Multi-class classification — because there are three mutually exclusive possible diagnoses.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Regression — because medical predictions involve uncertainty.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="If a dataset has 90% negative examples and 10% positive, a random split might put most positive examples in one set. `stratify=y` preserves the class ratio in both splits. This matters most with imbalanced classes — if the model never sees many positive examples in training, it can&#039;t learn to predict them, but evaluation on a test set without enough positives won&#039;t reveal the problem.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why should you use `stratify=y` when splitting a classification dataset?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It makes training faster by organizing the data more efficiently.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It ensures the proportion of each class in the training set matches the test set, preventing misleading evaluation results when one class is rare.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It randomly shuffles the labels to prevent the model from memorizing the order.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It is required by scikit-learn for all classifiers that use `predict_proba`.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Without a fixed `random_state`, `train_test_split` produces a different random split each run. Some splits are harder or easier than others by chance — you might get unlucky and have more difficult test cases. Setting `random_state=42` (or any fixed integer) guarantees the same split every time, making results reproducible and comparable.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You train a decision tree classifier with `max_depth=4` and get 80% accuracy on the test set. Your colleague trains the same model on the same data and gets 73% accuracy. What is the most likely explanation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>One of you has a different version of scikit-learn.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>One of you used a different value for `random_state` in `train_test_split`, resulting in a different split with a different difficulty distribution.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Decision trees are non-deterministic and produce different results every time.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The `max_depth` parameter controls accuracy directly.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

