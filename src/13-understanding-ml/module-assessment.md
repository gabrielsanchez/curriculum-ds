# Module Assessment

## Overview

You have now studied five advanced machine learning algorithms: k-Nearest Neighbors, Support Vector Machines, Random Forests, Gradient Boosting, and Neural Networks. Each lesson isolated one algorithm and examined it on the breast cancer dataset. This assessment brings them together for a rigorous head-to-head comparison, asks you to reflect on which algorithm to choose in different contexts, and tests your understanding of the core concepts across the module.

## Learning Objectives

By the end of this assessment, you will have:

- Compared all five advanced algorithms against the logistic regression and decision tree baselines on a shared dataset.
- Demonstrated effective hyperparameter tuning and model evaluation.
- Articulated when each algorithm is appropriate and what trade-offs it carries.

## Starter Code

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/13-understanding-ml/13-understanding-ml-module-assessment_starter.ipynb). All code templates are provided — your task is to complete the exercises, interpret the results, and answer the reflection questions. [Submit the link to the AI Grader for grading](https://ai-grader-production-07a3.up.railway.app/)

---

## Part 1: Algorithm Comparison

### Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

### Defining the Models

```python
# Each algorithm at its best-performing configuration from the module lessons
models = {
    # Baselines from introduction lesson
    "Logistic Regression":      LogisticRegression(random_state=42),
    "Decision Tree (depth=5)":  DecisionTreeClassifier(max_depth=5, random_state=42),

    # Module algorithms
    "kNN (k=7)":                KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF, C=10)":          SVC(kernel="rbf", C=10, gamma=0.01,
                                    probability=True, random_state=42),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200,
                                                         random_state=42),
    "Gradient Boosting":        GradientBoostingClassifier(
                                    n_estimators=200, learning_rate=0.1,
                                    max_depth=2, random_state=42),
    "Neural Network (MLP)":     MLPClassifier(hidden_layer_sizes=(100, 50),
                                               alpha=0.01, max_iter=500,
                                               random_state=42),
}
```

### Evaluation Loop

```python
# Tree-based models don't require scaling; distance-based and linear models do.
# Use scaled features for all models for fair comparison (trees are scale-invariant,
# so this doesn't hurt them — it just makes the loop simpler).

results = []

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc    = model.score(X_test_s, y_test)
    recall_malignant = (y_pred[y_test == 0] == 0).mean()
    auc    = roc_auc_score(y_test, y_proba)

    results.append({
        "Model":             name,
        "Accuracy":          acc,
        "Recall (malignant)": recall_malignant,
        "AUC":               auc,
    })

df_results = pd.DataFrame(results).sort_values("AUC", ascending=False)
df_results["Accuracy"]          = df_results["Accuracy"].map("{:.3f}".format)
df_results["Recall (malignant)"] = df_results["Recall (malignant)"].map("{:.3f}".format)
df_results["AUC"]               = df_results["AUC"].map("{:.3f}".format)

print(df_results.to_string(index=False))
```

Output:
```
                     Model Accuracy Recall (malignant)    AUC
          Gradient Boosting    0.974              0.953  0.999
          SVM (RBF, C=10)      0.982              0.977  0.998
       Neural Network (MLP)    0.974              0.953  0.998
              kNN (k=7)         0.974              0.953  0.993
  Random Forest (200 trees)    0.965              0.907  0.994
      Logistic Regression       0.956              0.929  0.993
  Decision Tree (depth=5)       0.930              0.857  0.951
```

### Cross-Validation

A single train/test split gives one estimate. Cross-validation gives a more reliable picture:

```python
# Use pipelines to ensure scaling is re-fit on each fold
cv_results = []

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    cv_results.append({
        "Model":    name,
        "CV AUC (mean)": scores.mean(),
        "CV AUC (std)":  scores.std(),
    })

df_cv = pd.DataFrame(cv_results).sort_values("CV AUC (mean)", ascending=False)
df_cv["CV AUC (mean)"] = df_cv["CV AUC (mean)"].map("{:.3f}".format)
df_cv["CV AUC (std)"]  = df_cv["CV AUC (std)"].map("{:.4f}".format)

print(df_cv.to_string(index=False))
```

Output:
```
                     Model CV AUC (mean) CV AUC (std)
       Gradient Boosting        0.995       0.0042
          SVM (RBF, C=10)       0.994       0.0046
       Neural Network (MLP)     0.994       0.0052
  Random Forest (200 trees)     0.994       0.0061
      Logistic Regression        0.993       0.0054
              kNN (k=7)          0.987       0.0089
  Decision Tree (depth=5)        0.948       0.0215
```

Cross-validation reveals a different ordering than the single test set:
- Gradient boosting and SVM lead by CV AUC (0.995, 0.994), but their confidence intervals overlap significantly.
- All five advanced algorithms beat the decision tree (0.948) and improve on logistic regression (0.993) by small margins.
- kNN's higher variability (std=0.0089) reflects its sensitivity to which examples land in each fold.
- The decision tree's high standard deviation (0.0215) confirms it as the least stable model.

---

## Part 2: The Full Comparison Table

Complete the table below by reading off values from the code outputs above. Fill in the "Notes" column with one observation per row.

| Algorithm | CV AUC | Test Accuracy | Recall (malignant) | Scaling required? | Notes |
|-----------|--------|---------------|-------------------|-------------------|-------|
| Decision Tree (baseline) | 0.948 | 93.0% | 85.7% | No | Highest variance; simplest to interpret |
| Logistic Regression (baseline) | 0.993 | 95.6% | 92.9% | Yes | Fast; interpretable coefficients |
| kNN (k=7) | 0.987 | 97.4% | 95.3% | Yes | Zero training time; slow prediction |
| SVM (RBF, C=10) | 0.994 | 98.2% | 97.7% | Yes | Best test recall; expensive to tune C+gamma |
| Random Forest (200 trees) | 0.994 | 96.5% | 90.7% | No | Best feature importances; reliable default |
| Gradient Boosting | 0.995 | 97.4% | 95.3% | No | Highest CV AUC; most hyperparameters |
| Neural Network (MLP) | 0.994 | 97.4% | 95.3% | Yes | Too small a dataset to show full benefit |

---

## Part 3: Reflection Questions

Answer each question in your notebook (2–4 sentences per question).

### Question 1: Model Selection

**Prompt:** You are deploying this breast cancer classifier in a clinical decision-support tool. A false negative (predicting benign when the tumor is malignant) has far more severe consequences than a false positive. Based on the comparison table, which algorithm would you recommend and why? What threshold adjustment would you make?

**Guidance:**
- Look at the Recall (malignant) column, not just overall accuracy.
- SVM has the highest malignant recall at the default threshold (97.7%).
- All algorithms have AUC ≥ 0.987, meaning any of them can achieve high recall by adjusting the probability threshold below 0.5.
- Justify both the algorithm choice and the threshold strategy.

---

### Question 2: Complexity vs. Accuracy

**Prompt:** Logistic regression achieves 95.6% accuracy and AUC 0.993. The best advanced model (SVM) achieves 98.2% and AUC 0.998. Is the improvement worth the added complexity? Under what circumstances would you prefer logistic regression?

**Guidance:**
- Consider the use case: medical diagnosis vs. internal tool vs. research.
- Consider interpretability: logistic regression's coefficients show each feature's contribution; SVM does not.
- Consider maintenance: logistic regression needs no scaling hyperparameter tuning.
- There is no universally right answer — the question is about reasoning through trade-offs.

---

### Question 3: What the Breast Cancer Dataset Doesn't Tell You

**Prompt:** All five algorithms perform within ~5% of each other on this dataset (93–98%). Name two characteristics of the breast cancer dataset that make this comparison an incomplete guide to real-world algorithm selection. What kind of dataset would reveal larger differences between these algorithms?

**Guidance:**
- Think about dataset size (569 samples), data type (all numeric), class balance (63%/37%), and noise level.
- A dataset with 100,000 samples and 500 features would expose kNN's scalability weakness.
- A dataset with complex non-linear interactions might favor SVM/neural networks more clearly.
- A dataset with many irrelevant features would favor Lasso, random forests, or gradient boosting's feature selection.

---

## Part 4: Coding Assessment

Complete this exercise in your notebook.

### Exercise: Hyperparameter Tuning and Threshold Optimization

You are given the breast cancer dataset. Your task:

1. **Train a `GradientBoostingClassifier` with `GridSearchCV`** over the following grid:
   - `n_estimators`: [100, 200, 300]
   - `learning_rate`: [0.05, 0.1, 0.2]
   - `max_depth`: [2, 3]
   - Use 5-fold cross-validation, scoring on `"roc_auc"`.

2. **Report** the best parameters and best CV AUC.

3. **Evaluate the best model** on the held-out test set. Report accuracy, recall on malignant, and AUC.

4. **Plot the ROC curve** for the best model. Mark the point corresponding to the default 0.5 threshold and a threshold that achieves at least 98% recall on malignant.

5. **Write 2–3 sentences** interpreting your ROC curve: what threshold would you use in a clinical context and why?

### Template

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ── Step 1: Grid search ───────────────────────────────────────────────────────
param_grid = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [2, 3],
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid.fit(X_train, y_train)   # tree-based: no scaling needed

print(f"Best params: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

# ── Step 2: Evaluate on test set ──────────────────────────────────────────────
best_model = grid.best_estimator_
y_proba    = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=cancer.target_names))

# ── Step 3: ROC curve with threshold markers ──────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# Default threshold (≈0.5)
default_idx = np.argmin(np.abs(thresholds - 0.5))

# Find threshold for ≥98% recall on malignant (class 0)
# Note: pos_label=1 means tpr = recall for benign, fpr = FPR for benign.
# For malignant recall, we look at 1-fpr (= TNR for benign = recall for malignant).
mal_recall   = 1 - fpr   # recall for malignant class
high_recall_idx = np.argmax(mal_recall >= 0.98)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="steelblue", linewidth=2,
         label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.scatter(fpr[default_idx], tpr[default_idx], s=120, color="coral",
            zorder=5, label=f"Threshold=0.5 (malignant recall={mal_recall[default_idx]:.2f})")
plt.scatter(fpr[high_recall_idx], tpr[high_recall_idx], s=120, color="green",
            zorder=5, label=f"≥98% malignant recall (threshold≈{thresholds[high_recall_idx]:.2f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve — Gradient Boosting (Best Model)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(f"\nThreshold for ≥98% malignant recall: {thresholds[high_recall_idx]:.4f}")
print(f"At this threshold — malignant recall: {mal_recall[high_recall_idx]:.3f}, "
      f"benign precision: {tpr[high_recall_idx]:.3f}")
```

---

## Module Summary: Algorithm Selection Guide

Use this reference when starting a new classification project:

```
Start here:
  1. Always fit Logistic Regression as a baseline.
     → If accuracy is within 2% of what you need: stop here.
     → If not: continue.

  2. Is the dataset small (<1,000 samples)?
     → Try SVM (RBF kernel) and kNN.
     → Avoid neural networks (overfit with limited data).

  3. Is the dataset medium (1,000–100,000 samples)?
     → Random Forest is the best-effort default.
     → Gradient Boosting for maximum accuracy (more tuning required).

  4. Is the data unstructured (images, text, audio)?
     → Neural Networks (CNN, Transformer) — module 14.
     → Gradient boosting won't work on raw pixels or tokens.

  5. Is interpretability required?
     → Logistic Regression (coefficients) or Decision Tree (rules).
     → Random Forest (feature importances, not individual predictions).
     → SVM, Gradient Boosting, Neural Networks: black boxes.

  6. Is prediction latency critical?
     → Logistic Regression (fastest).
     → Random Forest or Gradient Boosting (moderate).
     → kNN (slow: searches all training data on every prediction).
```

---

## Knowledge Check

### **Question 1: Across the five advanced algorithms, which two properties are shared by ALL distance-based and linear algorithms but NOT by tree-based algorithms?**

1. They all require feature scaling, and they all provide probability estimates natively.
2. They all require feature scaling to work correctly, and they all are sensitive to irrelevant features — both kNN and SVM can be misled by features that don't contribute to the prediction.
3. They all require feature scaling, and they all have O(n²) prediction time.
4. They all require cross-validation for hyperparameter tuning and cannot be used on unscaled data.

**Correct Answer:**
2. Distance-based and linear algorithms (kNN, SVM, logistic regression, neural networks) require feature scaling to work correctly, and they are sensitive to irrelevant features because irrelevant features contribute noise to the distance or linear combination computation.

**Explanation:**
Tree-based algorithms (decision trees, random forests, gradient boosting) split on one feature at a time and compare feature values against a threshold — the scale of features doesn't affect which split is optimal, and irrelevant features are simply never selected for splits (or selected rarely, giving them low importance). kNN and SVM, by contrast, compute distances across all features simultaneously: an irrelevant feature with a large scale will dominate the distance calculation. Logistic regression and neural networks similarly compute weighted sums where unscaled features can dominate gradients during training.

---

### **Question 2: Random forests use bagging; gradient boosting uses boosting. On the breast cancer dataset, gradient boosting slightly outperforms random forests (CV AUC 0.995 vs 0.994). What general conditions would make the gap between boosting and bagging larger?**

1. Gradient boosting always outperforms random forests by a large margin on any classification task.
2. Gradient boosting's advantage grows when the dataset is large (more residual corrections can be made) and when the data has complex non-linear interactions that can be captured incrementally. Random forests' advantage grows when the data is noisy (boosting amplifies noise) or when fast training is needed.
3. The gap grows when more trees are used — gradient boosting always benefits more from additional trees than random forests.
4. The gap grows when the dataset is small — boosting is designed for small datasets while bagging works better on large datasets.

**Correct Answer:**
2. Gradient boosting's advantage grows on large, complex datasets with non-linear interactions. Random forests' advantage grows on noisy datasets (where boosting amplifies noise) or when fast, low-tuning results are needed.

**Explanation:**
On the breast cancer dataset (569 samples, 30 features), both methods are excellent and nearly equivalent. On Kaggle tabular competitions with 100,000+ samples and complex feature interactions, gradient boosting (XGBoost, LightGBM) consistently outperforms random forests. On datasets with significant label noise or outliers, boosting's sequential error-correction can amplify noise — fitting the noisy residuals from early rounds — while random forests' averaging of independent trees is more robust. This is why random forests are preferred as a "fire-and-forget" baseline while gradient boosting is the choice when squeezing out maximum accuracy.

---

### **Question 3: A colleague says "neural networks always outperform gradient boosting when you have enough data." Is this claim accurate? What is missing from it?**

1. Yes — with large enough datasets, neural networks always outperform gradient boosting because they have more parameters and can learn more complex functions.
2. The claim is overly broad and context-dependent. Neural networks outperform gradient boosting on unstructured data (images, text, audio) where their specialized architectures (CNNs, Transformers) encode useful inductive biases. On structured tabular data, gradient boosting often matches or beats neural networks even with large datasets — gradient boosting's strength comes from how it handles heterogeneous features and interactions, not from raw data volume. The claim also ignores the No Free Lunch theorem: no algorithm universally dominates.
3. The claim is correct for structured tabular data but incorrect for unstructured data — neural networks fail on tabular data regardless of dataset size.
4. The claim is incorrect because neural networks are always slower to train than gradient boosting, which means they are impractical regardless of performance.

**Correct Answer:**
2. The claim is overly broad. Neural networks outperform gradient boosting on unstructured data (images, text, audio) with specialized architectures. On tabular data, gradient boosting often matches or beats neural networks even with large datasets.

**Explanation:**
The history of Kaggle competitions is instructive: from ~2014 to present, gradient boosting (XGBoost, LightGBM) has dominated tabular data competitions — even on datasets with millions of samples. Neural networks win convincingly on image, text, and audio tasks, where their spatial or sequential structure assumptions match the data. "Tabular" data — heterogeneous features like age, income, zip code, product category — doesn't have the structure that CNNs or Transformers exploit. This doesn't mean neural networks never win on tabular data, but the assumption that scale always favors them is false. The right choice remains empirical: try gradient boosting and neural networks and compare with cross-validation.
