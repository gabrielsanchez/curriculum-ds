# Multi-label Classification

## Overview

In the previous lesson, you built binary classifiers that predict one of two possible outcomes and learned how to evaluate them rigorously. Both binary and multi-class classification assume that each sample belongs to **exactly one** category. In this lesson, you'll explore a structurally different problem: **multi-label classification**, where a single sample can simultaneously belong to multiple categories. This comes up constantly in real-world applications and requires a different modeling approach.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Recognize problems where samples can carry multiple labels simultaneously.
- Implement multi-label classification using scikit-learn's `MultiOutputClassifier`.
- Evaluate multi-label models using Hamming loss and label-level metrics.

## Key Terms

**Multi-label classification:** A classification task where each sample can be assigned zero, one, or multiple labels from a predefined set simultaneously.

**Label set:** The complete collection of possible labels. In a movie genre problem, the label set might be {action, comedy, drama, horror, romance, thriller}.

**Binary relevance:** The simplest multi-label approach — train one independent binary classifier per label and combine their outputs.

**`MultiOutputClassifier`:** A scikit-learn wrapper that applies a binary classifier independently to each output column, implementing binary relevance.

**Hamming loss:** The fraction of label-sample pairs that are incorrectly predicted. Lower is better. A Hamming loss of 0.1 means 10% of individual label predictions are wrong.

**Label-level precision/recall:** Per-label precision and recall scores, averaged across labels using macro, micro, or weighted averaging.

**Jaccard similarity (subset accuracy):** The fraction of samples where the predicted label set exactly matches the true label set. Very strict — any single wrong label counts as a full failure.

**Classifier chain:** An approach that trains classifiers sequentially, where each successive classifier uses the previous classifiers' predictions as additional features.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/09-ml-classification/04_multi-label-classification_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

Consider a movie on a streaming platform. It doesn't fit into just one genre:

- *The Dark Knight* is action, crime, and thriller
- *Toy Story* is animation, adventure, and comedy
- *Parasite* is drama, thriller, and dark comedy

Forcing it into a single category loses information. The same is true for:

- **Document tagging:** A news article about climate legislation is tagged "environment", "politics", and "policy"
- **Medical diagnosis:** A patient's chart may indicate hypertension, diabetes, and sleep apnea simultaneously
- **Image labeling:** A photo can contain a dog, a park, and a child at the same time
- **Music categorization:** A song can be "jazz", "instrumental", and "relaxing"

Multi-label classification preserves this richness. Instead of predicting a single class, the model outputs a **binary vector** — one entry per possible label, where 1 means "this label applies" and 0 means it doesn't.

## How the Target Variable Changes

In binary and multi-class classification, `y` is a 1D array of class labels:

```python
# Binary / multi-class
y = [0, 1, 0, 1, 2, 1, ...]   # one integer per sample
```

In multi-label classification, `y` is a **2D matrix** — one row per sample, one column per label:

```python
# Multi-label: [action, comedy, drama, thriller, romance]
y = [
    [1, 0, 0, 1, 0],   # sample 0: action + thriller
    [0, 1, 1, 0, 0],   # sample 1: comedy + drama
    [0, 0, 1, 1, 0],   # sample 2: drama + thriller
    [0, 0, 0, 0, 1],   # sample 3: romance only
]
```

This matrix format is what scikit-learn expects for multi-label tasks.

## Building a Multi-label Dataset

Let's create a synthetic movie dataset to make this concrete:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

# Generate a synthetic multi-label dataset
X, y = make_multilabel_classification(
    n_samples=1000,
    n_features=20,
    n_classes=5,       # 5 possible labels
    n_labels=2,        # average of 2 labels per sample
    random_state=42
)

label_names = ["action", "comedy", "drama", "thriller", "romance"]
y_df = pd.DataFrame(y, columns=label_names)

print("Label distribution:")
print(y_df.mean().round(2))

print(f"\nSample label vectors:")
print(y_df.head())
```

Output:
```
Label distribution:
action      0.44
comedy      0.39
drama       0.52
thriller    0.41
romance     0.36

Sample label vectors:
   action  comedy  drama  thriller  romance
0       1       1      1         0        0
1       0       0      1         1        0
2       0       0      0         1        1
3       1       0      1         0        0
4       1       1      0         0        0
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Approach 1: Binary Relevance with `MultiOutputClassifier`

The simplest and most common approach is **binary relevance**: train one independent binary classifier for each label. If you have 5 labels, you train 5 separate binary classifiers.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Wrap a binary classifier to handle multiple outputs
base_clf = LogisticRegression(random_state=42)
multi_clf = MultiOutputClassifier(base_clf)

multi_clf.fit(X_train, y_train)
y_pred = multi_clf.predict(X_test)

print(f"Predictions shape: {y_pred.shape}")  # (200, 5)
print("First 5 predictions:")
print(pd.DataFrame(y_pred[:5], columns=label_names))
```

Output:
```
Predictions shape: (200, 5)
First 5 predictions:
   action  comedy  drama  thriller  romance
0       1       1      1         0        0
1       0       0      1         1        0
2       0       0      0         0        1
3       1       0      1         0        0
4       1       1      0         0        0
```

The wrapper trains 5 logistic regression models — one for each genre — and combines their predictions into a single output matrix.

**Advantage:** Simple, interpretable, parallelizable.

**Limitation:** The independent classifiers don't know about label correlations. In reality, "action" and "thriller" often co-occur, but binary relevance treats each label as isolated.

## Approach 2: Classifier Chains

Classifier chains improve on binary relevance by propagating predictions from earlier classifiers as features for later ones:

- Classifier 1 predicts "action" using original features
- Classifier 2 predicts "comedy" using original features + predicted "action"
- Classifier 3 predicts "drama" using original features + predicted "action" + predicted "comedy"
- …and so on

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(LogisticRegression(random_state=42), random_state=42)
chain_clf.fit(X_train, y_train)
y_pred_chain = chain_clf.predict(X_test)
```

The chain allows later classifiers to leverage label correlations — if "action" is predicted, the probability of "thriller" may increase. The trade-off is that early prediction errors propagate through the chain.

## Evaluating Multi-label Models

Because each prediction is now a vector, standard binary metrics need adaptation.

### Hamming Loss

The most common multi-label metric. It measures the fraction of label-sample pairs that are wrong:

```python
from sklearn.metrics import hamming_loss

hl_br    = hamming_loss(y_test, y_pred)
hl_chain = hamming_loss(y_test, y_pred_chain)

print(f"Hamming Loss (Binary Relevance): {hl_br:.3f}")
print(f"Hamming Loss (Classifier Chain): {hl_chain:.3f}")
```

Output:
```
Hamming Loss (Binary Relevance): 0.213
Hamming Loss (Classifier Chain): 0.198
```

A Hamming loss of 0.213 means 21.3% of individual label predictions are incorrect. Lower is better; 0 is perfect.

### Exact Match (Subset Accuracy)

The strictest metric — what fraction of samples have every label exactly correct?

```python
from sklearn.metrics import accuracy_score

exact_match = accuracy_score(y_test, y_pred)
print(f"Exact match (subset accuracy): {exact_match:.1%}")
```

Output:
```
Exact match (subset accuracy): 37.0%
```

This is strict. Getting 4 out of 5 labels right still counts as a complete failure. Useful when partial predictions are unacceptable, but overly harsh for most applications.

### Label-level Classification Report

You can compute precision, recall, and F1 for each label independently:

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    target_names=label_names,
    zero_division=0
))
```

Output:
```
              precision    recall  f1-score   support

      action       0.73      0.71      0.72        84
      comedy       0.71      0.70      0.71        76
       drama       0.74      0.75      0.74       102
    thriller       0.75      0.72      0.73        80
     romance       0.72      0.69      0.70        73

   micro avg       0.73      0.72      0.72       415
   macro avg       0.73      0.71      0.72       415
weighted avg       0.73      0.72      0.72       415
```

This lets you identify which labels your model handles well or poorly.

## Metric Summary

| Metric | What it measures | Strict? |
|--------|-----------------|---------|
| Hamming loss | Fraction of individual label errors | No — counts each label independently |
| Exact match | Fraction of fully correct label vectors | Yes — all-or-nothing per sample |
| Per-label F1 | F1 for each label separately | No |
| Macro-avg F1 | Average F1 across all labels (equal weight per label) | No |
| Weighted-avg F1 | Average F1 weighted by label frequency | No |

**Which to use?**
- **Hamming loss** for a quick overall health check
- **Per-label metrics** to identify weak spots
- **Exact match** only when partial correctness is unacceptable

## When to Use Multi-label vs. Multi-class

| Situation | Type |
|-----------|------|
| A product belongs to exactly one category | Multi-class |
| A product can belong to several categories simultaneously | Multi-label |
| Each customer has exactly one risk tier | Multi-class |
| A document can address multiple topics at once | Multi-label |
| An image shows one dominant object | Multi-class |
| An image can contain multiple tagged objects | Multi-label |

The key question: **can a single sample belong to more than one category at the same time?** If yes, you need multi-label classification.

## Conclusion

In this lesson, you learned that multi-label classification addresses problems where a single sample can have multiple simultaneous labels — common in document tagging, image labeling, and medical diagnosis. You implemented the binary relevance approach using `MultiOutputClassifier`, which trains independent classifiers per label, and saw how classifier chains improve on this by modeling label correlations. You evaluated models with Hamming loss, exact match, and per-label F1 scores. In the next lesson, you'll study **multi-class classification** — where you have three or more mutually exclusive categories and need strategies like One-vs-Rest to extend binary classifiers.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="Since a single ticket can be tagged with multiple topics at the same time (e.g., both &quot;billing&quot; and &quot;technical&quot;), each ticket needs a binary vector of labels, not a single class. Multi-label classification is the right framework. Multi-class would force exactly one label per ticket, losing the information that it covers multiple topics. Binary would only work if there were exactly two possible outcomes.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A developer is building a system that tags customer support tickets. Each ticket can be about billing, technical issues, shipping, or returns — and many tickets are about more than one topic. What type of classification is this?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Binary classification — the model predicts whether a ticket requires escalation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Multi-class classification — the model assigns the primary topic to each ticket.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Multi-label classification — the model assigns all applicable topic tags to each ticket simultaneously.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Regression — because the number of topics per ticket can vary.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Hamming loss = 0.08 means 8% of individual label-sample pairs are incorrectly predicted — so 92% are right at the individual label level, which is good performance. Exact match = 22% means only 22% of samples have every label exactly correct. This gap is expected: if a model correctly predicts 5 out of 6 labels per sample, Hamming loss would be low (≈17%), but exact match would be 0% because no sample is fully correct. These metrics measure different things and should both be reported.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You train a multi-label classifier on a 6-label dataset and compute Hamming loss = 0.08 and exact match accuracy = 22%. Which statement best interprets these results?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model is performing poorly on all metrics and needs significant improvement.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The model correctly predicts individual labels most of the time (92% of label predictions are right), but rarely predicts the entire label set for a sample perfectly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The model's Hamming loss of 0.08 means it gets 8% of samples entirely correct.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Exact match accuracy should always equal 1 minus Hamming loss.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Binary relevance treats each label independently, ignoring the fact that some labels often appear together (e.g., &quot;action&quot; and &quot;thriller&quot;). Classifier chains pass the prediction of each label as an additional feature to the next classifier, allowing it to learn &quot;if action=1, thriller is more likely.&quot; This can improve performance when labels are correlated. The trade-off is that early prediction errors propagate through the chain, and training is sequential rather than parallelizable.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What is the main advantage of the Classifier Chain approach over Binary Relevance for multi-label classification?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Classifier chains are faster to train because they use fewer models.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Classifier chains model label correlations by using earlier predictions as features for later classifiers, potentially improving accuracy when labels co-occur.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Classifier chains always produce lower Hamming loss regardless of the dataset.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Classifier chains use only one model instead of one per label.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

