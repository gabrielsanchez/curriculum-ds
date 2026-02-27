# Introduction to Advanced ML Models

## Overview

You've now built a solid foundation: logistic regression and decision trees for classification (module 09), linear regression and regularized models for regression (module 11), and an independent capstone project (module 12). These tools are genuinely useful — logistic regression powers production systems at major tech companies, and regularized regression is the standard in many scientific fields. But there's a large class of problems where these methods leave meaningful performance on the table. This module introduces five additional algorithms — kNN, SVM, Random Forests, Gradient Boosting, and Neural Networks — and, crucially, teaches you when each one earns its additional complexity.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Characterize the strengths and limitations of each advanced algorithm covered in this module.
- Select an appropriate algorithm based on dataset properties and problem constraints.

## Key Terms

**Ensemble method:** A model that combines multiple individual models (base learners) to produce better predictions than any single model alone. Random forests and gradient boosting are ensemble methods.

**Bagging (Bootstrap Aggregating):** An ensemble technique that trains multiple models on different random subsets of the training data and averages their predictions. Reduces variance. Used in random forests.

**Boosting:** An ensemble technique that trains models sequentially, with each new model focusing on correcting the errors of its predecessors. Reduces both bias and variance. Used in gradient boosting.

**Kernel method:** A technique for operating in a high-dimensional (or infinite-dimensional) feature space without explicitly computing transformations, using a kernel function to measure similarity between points. Used in SVM.

**Hyperparameter:** A configuration value set before training (e.g., the number of trees in a forest, the kernel type of an SVM). Distinguished from model *parameters* (e.g., weights), which are learned from data.

**No Free Lunch theorem:** The theoretical result that no single learning algorithm outperforms all others across all possible problems. Every algorithm has strengths and weaknesses, and the best model depends on the data.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/01_introduction-to-advanced-ml_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Why Learn More Algorithms?

You've already used two algorithm families:

- **Linear models** (logistic regression, linear regression, Ridge, Lasso) — fit a weighted sum of features. Fast, interpretable, and strong on clean tabular data with linear relationships.
- **Decision trees** — recursively partition the feature space. Interpretable, handle mixed data types, but prone to overfitting unless depth is constrained.

These fail in common scenarios:

| Scenario | Why linear models fail | Why decision trees fail |
|----------|----------------------|-------------------------|
| Complex non-linear boundaries | Can't fit curves without polynomial features | Individual trees overfit; ensembles help |
| Very high-dimensional data | May work, but regularization needed | Each split considers few features |
| Small datasets with many features | Overfits | Very sensitive to small data changes |
| Noisy tabular data | Underfits if relationships are non-linear | Memorizes noise with deep trees |

The algorithms in this module address these gaps in different ways.

## The Dataset for This Module

All algorithm lessons use the **breast cancer dataset** from scikit-learn — a binary classification task predicting whether a tumor is malignant (cancerous) or benign, based on 30 numeric features derived from digitized cell nucleus images.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)   # 0 = malignant, 1 = benign

print(f"Shape:  {X.shape}")
print(f"Classes: {cancer.target_names}  (0=malignant, 1=benign)")
print(f"\nClass distribution:")
print(y.value_counts().rename({0: "malignant", 1: "benign"}))
print(f"\nFeature sample: {list(cancer.feature_names[:5])}")
```

Output:
```
Shape:  (569, 30)
Classes: ['malignant' 'benign']  (0=malignant, 1=benign)

Class distribution:
benign       357
malignant    212

Feature sample: ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
```

569 samples, 30 numeric features, 63% benign / 37% malignant. The medical context is deliberate — recall that in safety-critical classification (module 09), missing a malignant tumor (false negative) is far more costly than a false alarm. This will matter when we evaluate models.

```python
# Standard split and scale — used in every lesson in this module
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
```

Output:
```
Train: (455, 30),  Test: (114, 30)
```

## Establishing the Baselines

Before learning any new algorithm, establish what the algorithms you already know can do:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

baselines = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
}

print(f"{'Model':<30} {'Accuracy':>10} {'Recall (malignant)':>20} {'AUC':>8}")
print("-" * 72)

for name, clf in baselines.items():
    clf.fit(X_train_s, y_train)
    y_pred  = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    acc    = (y_pred == y_test).mean()
    recall = (y_pred[y_test == 0] == 0).mean()   # recall on malignant (class 0)
    auc    = roc_auc_score(y_test, y_proba)

    print(f"{name:<30} {acc:>10.3f} {recall:>20.3f} {auc:>8.3f}")
```

Output:
```
Model                          Accuracy  Recall (malignant)      AUC
------------------------------------------------------------------------
Logistic Regression               0.956               0.929    0.993
Decision Tree (depth=5)           0.930               0.857    0.951
```

Logistic regression already achieves 95.6% accuracy and AUC of 0.993 on this dataset. The question this module answers is: **can more advanced methods do better, and does the improvement justify the added complexity?**

## A Map of the Algorithms

| Algorithm | Core idea | Best for | Key weakness |
|-----------|-----------|---------|-------------|
| **k-Nearest Neighbors** | Classify by the majority label of the k closest training points | Simple, intuitive problems; small-medium datasets | Slow prediction; sensitive to scale and irrelevant features |
| **Support Vector Machine** | Find the hyperplane with the widest margin between classes | High-dimensional data; small-medium datasets; non-linear via kernels | Slow on large data; sensitive to scale; fewer probability outputs |
| **Random Forest** | Average many decision trees, each trained on a bootstrap sample | General-purpose; handles mixed data; robust to outliers | Less interpretable than a single tree; slow on very large data |
| **Gradient Boosting** | Sequentially correct errors with shallow trees | Tabular data competitions; highest accuracy on structured data | Slower to train; more hyperparameters to tune |
| **Neural Networks** | Layers of neurons learn hierarchical representations | Image, text, audio; complex non-linear relationships | Needs large data; expensive to train; hardest to interpret |

## The Algorithm Selection Framework

When starting a new project, use this decision process:

```
How much data do you have?
├── Small (<1,000 samples)
│     → Try Logistic Regression, SVM, kNN first
│     → Neural networks likely overfit
│
├── Medium (1,000–100,000 samples)
│     → Random Forests and Gradient Boosting are typically the best choices
│     → SVM may work but gets slow
│
└── Large (>100,000 samples)
      → Gradient Boosting (XGBoost/LightGBM) or Neural Networks
      → Logistic Regression as a fast baseline

Is interpretability required?
├── Yes → Logistic Regression (coefficients), Decision Tree (rules),
│         or Random Forest (feature importances)
└── No  → Gradient Boosting or Neural Networks for maximum accuracy

Are features high-dimensional (image, text, audio)?
├── Yes → Neural Networks (especially CNNs, RNNs — module 14)
└── No  → Tree ensembles or SVM
```

## A Caution About "More Complex = Better"

The most common mistake is reaching for the most sophisticated algorithm first. In practice:

1. **Start simple.** Logistic regression is fast to train, easy to interpret, and reveals whether the problem is actually hard. A sophisticated model that barely beats logistic regression may not be worth its additional complexity.

2. **Complexity has costs.** Neural networks require more data, more computation, and more tuning. Gradient boosting has a dozen hyperparameters. More complex models take longer to train, harder to debug, and can fail silently.

3. **Data quality matters more than algorithm choice.** A gradient boosting model on messy data will underperform a logistic regression on well-engineered features. The biggest wins come from EDA, cleaning, and feature engineering — not algorithm selection.

## Conclusion

In this lesson, you established the baseline performance on the breast cancer dataset (logistic regression: 95.6% accuracy, AUC 0.993), got an overview of the five advanced algorithms this module covers, and learned the framework for deciding when each is worth trying. In the next lesson, you'll start with the most conceptually transparent of the five: **k-Nearest Neighbors**, an algorithm with almost no training step and predictions driven entirely by similarity to known examples.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/01_introduction-to-advanced-ml_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="With only 800 samples, complex models like neural networks and gradient boosting have limited data to learn from and risk overfitting. Logistic regression&#039;s coefficients directly show each feature&#039;s contribution to the prediction — straightforwardly explainable. A shallow decision tree provides literal if-then rules that non-technical stakeholders can follow. Both are also fast to train and debug, making them good starting points before trying anything more sophisticated.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A data scientist has 800 samples and needs a model they can explain to non-technical stakeholders (showing which features drove a prediction). Which algorithm family is most appropriate to start with?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Neural Networks — they achieve the highest accuracy and can be explained with attention maps.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Gradient Boosting — it's the most accurate algorithm for tabular data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Logistic Regression or a shallow Decision Tree — both are interpretable and appropriate for small datasets where neural networks and gradient boosting would likely overfit.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>SVM — it always outperforms logistic regression with the right kernel.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The key difference is the relationship between models. In bagging, each model is trained independently and contributes equally to the final vote — the ensemble&#039;s power comes from diversity. In boosting, each new model explicitly targets the mistakes of the existing ensemble, building a progressively better combined predictor. This makes boosting more powerful but also more sensitive to overfitting and noisy data.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What is the key distinction between bagging (used in Random Forests) and boosting (used in Gradient Boosting)?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Bagging uses decision trees; boosting uses neural networks.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Bagging trains multiple models in parallel on random subsets of data and averages their predictions, reducing variance. Boosting trains models sequentially, with each model correcting the previous model's errors, reducing both bias and variance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Bagging is used for regression; boosting is used for classification.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Bagging requires more data than boosting because it samples with replacement.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Model selection involves trade-offs beyond accuracy. A 2% accuracy gain that takes 10× longer to train, requires a PhD to tune, and can&#039;t be explained to a regulator may not be worth it. On the other hand, in high-stakes domains (medical diagnosis, fraud detection), 2% can translate to thousands of lives or millions of dollars — and the complexity is justified. The right choice is always context-dependent.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Logistic regression achieves 94% accuracy on a classification problem. A gradient boosting model achieves 96% accuracy. Should you always deploy the gradient boosting model?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Yes — higher accuracy always justifies the more complex model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Not necessarily — the 2% improvement must be weighed against the costs of gradient boosting: slower training and inference, more hyperparameters to tune, harder to interpret, and more infrastructure complexity. If the improvement is not practically meaningful (e.g., in a low-stakes application), the simpler model is often preferable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Yes — gradient boosting is always more reliable than logistic regression.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>No — you should always use the simplest possible model regardless of performance differences.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

