# Module Assessment

## Overview

This assessment brings together everything you've learned in module 09: binary classification, model evaluation metrics, and extending classifiers to multi-class settings. You'll work with the **mushroom dataset** — a preview of the case study in module 10. Your goal is to build and compare multiple classifiers, evaluate them rigorously, and justify your choices.

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/09-ml-classification/09-ml-classification-module-assessment_starter.ipynb).

## Learning Objective

By the end of this assessment, you will have demonstrated the ability to:

- Build and evaluate binary classifiers end-to-end using appropriate metrics.
- Compare multiple classification algorithms on the same dataset.
- Tune a model's decision threshold based on domain requirements.
- Interpret confusion matrices and classification reports.

## The Dataset

The mushroom dataset contains physical descriptions of hypothetical mushrooms, each labeled as either **edible** or **poisonous**. This is a high-stakes binary classification problem: a false negative (predicting "edible" when a mushroom is actually poisonous) has severe real-world consequences.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, f1_score, recall_score
)
import matplotlib.pyplot as plt

# Load the mushroom dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = [
    "class", "cap_shape", "cap_surface", "cap_color", "bruises",
    "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color",
    "stalk_shape", "stalk_root", "stalk_surface_above", "stalk_surface_below",
    "stalk_color_above", "stalk_color_below", "veil_type", "veil_color",
    "ring_number", "ring_type", "spore_print_color", "population", "habitat"
]
df = pd.read_csv(url, header=None, names=columns)

print(df.shape)
print(df["class"].value_counts())
```

Output:
```
(8124, 23)
e    4208    ← edible
p    3916    ← poisonous
```

## Coding Assessment

Practice the concepts from this module using this [notebook](#). After completing all exercises, save your notebook to GitHub and [submit the link for grading](https://ai-grader-production-07a3.up.railway.app/).

The notebook walks you through the following tasks:

### Task 1: Data Preparation

Encode all categorical features and split the data into training and test sets with stratification.

```python
# Encode target
df["label"] = (df["class"] == "p").astype(int)   # 1 = poisonous, 0 = edible

# Encode features
X = df.drop(columns=["class", "label"])
y = df["label"]

# Encode all categorical columns to integers
for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### Task 2: Train and Evaluate a Logistic Regression Baseline

Train a logistic regression model and generate a full classification report. Note which metric matters most for this domain.

*Hint: In mushroom safety prediction, what is the cost of a false negative?*

### Task 3: Train a Decision Tree

Train a `DecisionTreeClassifier` with `max_depth=5`. Compare its confusion matrix to logistic regression. Which model makes fewer false negatives (predicting edible when the mushroom is actually poisonous)?

### Task 4: Adjust the Decision Threshold

For the logistic regression model, lower the classification threshold to 0.3. Does this improve recall for the poisonous class? What happens to precision?

```python
# Example: adjusting threshold
y_proba = clf.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_proba >= 0.3).astype(int)
```

### Task 5: Comparison Table

Fill in this table with your results:

| Model | Accuracy | Precision (poisonous) | Recall (poisonous) | F1 (poisonous) | AUC |
|-------|----------|----------------------|--------------------|----------------|-----|
| Logistic Regression (threshold=0.5) | | | | | |
| Logistic Regression (threshold=0.3) | | | | | |
| Decision Tree (max_depth=5) | | | | | |

### Task 6: Recommendation

Write 2–3 sentences justifying which model and threshold you would deploy in a real application that helps hikers identify safe mushrooms, and why.

---

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="In mushroom safety prediction, a false negative (predicting &quot;edible&quot; when the mushroom is poisonous) can be life-threatening. High accuracy masks this problem because the dataset is fairly balanced. A recall of 62% on the poisonous class means more than one-third of poisonous mushrooms would be cleared as safe to eat. For safety-critical applications, recall on the dangerous class must be optimized — typically approaching 99–100% — even at the cost of reduced precision (more false alarms telling users to avoid mushrooms that are actually edible).">
  <div class="quiz-question">
    <strong>Question 1:</strong> For the mushroom classification problem, you train a model that achieves 96% accuracy but has a recall of only 62% for the poisonous class. Is this model acceptable for a real application? Why or why not?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Yes — 96% accuracy is excellent and means most predictions are correct.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>No — 62% recall for the poisonous class means 38% of poisonous mushrooms are predicted as edible, which is dangerous. For this application, recall on the poisonous class should be near-perfect.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Yes — precision is more important than recall for safety-critical applications.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>No — you should always achieve above 98% accuracy before deploying a model.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="Model C has the highest accuracy but the lowest recall for the poisonous class (0.74) — meaning 26% of poisonous mushrooms are misclassified as edible. This is unacceptable for a safety application. Model B achieves high accuracy (97%), the highest recall (0.96 — only 4% of poisonous mushrooms missed), and the best AUC (0.99). Model A has the lowest recall (0.89), missing 11% of poisonous cases. Model B is the best choice because it excels on the metric that matters most for this domain.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You've trained three classifiers on the mushroom dataset with the following results:
| Model | Accuracy | Recall (poisonous) | AUC |
|-------|----------|--------------------|-----|
| A | 94% | 0.89 | 0.97 |
| B | 97% | 0.96 | 0.99 |
| C | 99% | 0.74 | 0.98 |
**Which model should you prefer for deployment, and why?**
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Model C — highest accuracy means fewest total mistakes.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Model A — lowest accuracy means the model generalizes better without overfitting.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Model B — best balance of all three metrics, with the highest recall for the dangerous class.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Model C — highest accuracy and AUC indicates the best overall classifier.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="A model that always predicts &quot;edible&quot; on a 95% edible test set achieves 95% accuracy — appearing excellent while being completely useless. When evaluation data has a different class distribution from training data, you must rely on class-specific metrics (precision, recall, F1 for the minority class) rather than overall accuracy. Additionally, if the real-world deployment environment truly has 5% poisonous cases (rather than the training distribution of ~48%), you may need to recalibrate the model&#039;s probability scores or adjust the decision threshold to maintain adequate recall on the rare but dangerous class.">
  <div class="quiz-question">
    <strong>Question 3:</strong> After training your final mushroom classifier, you discover that the test set contains 95% edible mushrooms and only 5% poisonous ones — very different from the 52%/48% split in the original dataset. How does this affect your evaluation, and what should you do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Nothing changes — the model's test performance is always representative of real-world performance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The highly imbalanced test set means accuracy will be misleadingly high. You should report precision, recall, and F1 for the poisonous class specifically, and consider whether the test set's class distribution matches real-world deployment conditions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>You should retrain the model on the imbalanced test set to improve generalization.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>You should use the micro-average F1 since it is not affected by class distribution.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

