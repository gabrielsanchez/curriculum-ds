# Supervised vs. Unsupervised Learning

## Overview

In the previous lesson, you learned what machine learning is and how it differs from traditional programming. Not all ML problems are the same, however — the type of data you have and the question you're asking determine which family of algorithms you should use. In this lesson, you'll learn to distinguish between **supervised** and **unsupervised** learning, understand the key sub-types of each, and develop the judgment to match a real-world problem to the right ML approach.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Differentiate supervised learning (regression, classification) from unsupervised learning (clustering, dimensionality reduction).
- Match a given problem to the correct ML approach.

## Key terms

**Supervised learning:** A type of machine learning where the training data includes labeled examples — each input has a corresponding correct output that the algorithm learns to predict.

**Unsupervised learning:** A type of machine learning where the training data has no labels — the algorithm finds structure, patterns, or groupings in the data on its own.

**Classification:** A supervised learning task where the target variable is a discrete category (e.g., spam/not spam, disease/no disease, cat/dog/bird).

**Regression:** A supervised learning task where the target variable is a continuous numeric value (e.g., house price, temperature, sales amount).

**Clustering:** An unsupervised learning task that groups similar data points together without predefined labels (e.g., customer segments, document topics).

**Dimensionality reduction:** An unsupervised technique that reduces the number of features in a dataset while preserving as much information as possible (e.g., PCA).

**Reinforcement learning:** A type of ML where an agent learns by taking actions in an environment and receiving rewards or penalties — not covered in depth in this module.

**Train/test split:** The practice of dividing a labeled dataset into a portion used to train the model and a separate portion held out to evaluate it.

## Introduction

The most important question to ask before choosing an algorithm is: **do I have labels?**

- If yes — you're doing **supervised learning**. The model learns from examples with known correct answers.
- If no — you're doing **unsupervised learning**. The model finds structure in the data without guidance.

This single distinction narrows your choice of algorithm dramatically and shapes the entire modeling workflow.

## Supervised Learning

In supervised learning, every training example consists of:
- **Features (X):** the input variables (what you know)
- **Label (y):** the correct answer (what you want to predict)

The model learns a function `f` such that `f(X) ≈ y`. After training, you can give it new X values (without labels) and it will produce predictions.

Think of supervised learning as learning from a **teacher** — the labels are the teacher's correct answers, and the algorithm adjusts itself to minimize its mistakes on those answers.

### Classification

The target variable is a **discrete category**. The model assigns each input to one of a fixed set of classes.

**Binary classification** — two possible outputs:
- Email: spam or not spam
- Transaction: fraudulent or legitimate
- Patient: disease present or absent
- Customer: will churn or won't churn

**Multi-class classification** — three or more possible outputs:
- Handwritten digit recognition: 0 through 9
- News article topic: politics, sports, technology, entertainment
- Plant disease severity: healthy, mild, moderate, severe

**How to recognize it:** The target is a category (a label or a yes/no). You're asking "which group does this belong to?"

### Regression

The target variable is a **continuous number**. The model predicts a specific value on a continuous scale.

Examples:
- Predicting a house's sale price given its features
- Estimating a patient's blood pressure from health indicators
- Forecasting next month's sales revenue
- Predicting how many minutes a delivery will take

**How to recognize it:** The target is a number that can take any value in a range. You're asking "how much?" or "how many?"

### The Train/Test Split

Because supervised models learn from labeled data, evaluation is straightforward: hide a portion of your labeled data, train on the rest, and measure performance on the hidden portion.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% of data for training, 20% held out for evaluation
```

You'll use this pattern in every supervised learning project. The `random_state` parameter ensures reproducibility.

## Unsupervised Learning

In unsupervised learning, you only have features — no labels. The algorithm explores the data and finds patterns, structure, or compact representations on its own.

Think of unsupervised learning as learning without a teacher — you're exploring the data and asking "what's in here?" rather than "can I predict this label?"

### Clustering

Clustering groups similar data points together based on their feature values, without any predefined categories. The algorithm decides both how many groups exist (or you specify a number) and which points belong to each group.

Examples:
- **Customer segmentation:** Group customers by purchasing behavior to tailor marketing campaigns. You don't know the right groups in advance.
- **Document clustering:** Organize news articles by topic without pre-labeling them.
- **Anomaly detection:** Identify data points that don't belong to any cluster as potential fraud or equipment failure.
- **Genomics:** Group genes with similar expression patterns.

The most common clustering algorithm is **k-means**, which you'll encounter in later modules.

### Dimensionality Reduction

Real datasets often have dozens or hundreds of features. Many of these features are correlated or carry redundant information. Dimensionality reduction compresses many features into fewer, more informative ones.

Use cases:
- **Visualization:** Compress 50 features into 2 dimensions so you can plot them.
- **Noise reduction:** Remove low-information features before training another model.
- **Storage and speed:** Smaller feature sets mean faster training and less memory.

The most common technique is **PCA (Principal Component Analysis)**.

### Other Unsupervised Techniques

- **Association rule mining:** Find items that frequently appear together (e.g., "customers who buy X also buy Y" — used in recommendation systems).
- **Generative models:** Learn the underlying distribution of data to generate new examples (e.g., GANs for image synthesis).

## Reinforcement Learning (Brief Overview)

A third category, distinct from both supervised and unsupervised learning, is **reinforcement learning (RL)**. An RL agent:
- Takes actions in an environment
- Receives a reward (positive or negative) based on the outcome
- Learns a policy that maximizes cumulative reward over time

Examples: game-playing AI (Chess, Go, video games), robotic control, autonomous vehicle training, personalized content recommendation.

RL is not covered in depth in this curriculum, but it's worth knowing it exists as a distinct paradigm.

## Matching Problems to Approaches

Use this decision framework when approaching a new ML problem:

```
Do I have labeled training data?
├── YES → Supervised Learning
│         What type of target?
│         ├── Discrete category → Classification
│         │     How many classes?
│         │     ├── 2 → Binary classification
│         │     └── 3+ → Multi-class classification
│         └── Continuous number → Regression
│
└── NO → Unsupervised Learning
          What is the goal?
          ├── Find natural groups → Clustering
          ├── Reduce feature count → Dimensionality reduction
          └── Find patterns/associations → Association mining
```

## Side-by-Side Comparison

| Feature | Supervised | Unsupervised |
|---------|-----------|--------------|
| Requires labels? | Yes | No |
| Goal | Predict a known target | Discover structure |
| Evaluation | Accuracy, error on test labels | Cluster cohesion, interpretability |
| Examples | Email spam filter, price prediction | Customer segments, topic modeling |
| Common algorithms | Linear regression, decision trees, SVM | k-means, PCA, DBSCAN |

## Practical Examples

**Scenario 1:** A hospital has 10,000 patient records with a column indicating whether each patient developed diabetes within five years. They want to flag high-risk patients for early intervention.

→ **Supervised / Binary classification** — labeled data exists (diabetes: yes/no), and the goal is to predict that label for new patients.

**Scenario 2:** A streaming service wants to understand distinct viewer "types" based on watch history, without any predefined categories.

→ **Unsupervised / Clustering** — no labels exist; the goal is to discover natural groups in the behavior data.

**Scenario 3:** A real estate company wants to estimate the sale price of a house given its size, location, and number of bedrooms.

→ **Supervised / Regression** — labeled data exists (past sale prices), and the target is a continuous number.

**Scenario 4:** A marketing team has a dataset with 120 customer features and wants to reduce it to the 10 most informative dimensions before modeling.

→ **Unsupervised / Dimensionality reduction** — no prediction target; the goal is to compress the feature space.

## Conclusion

In this lesson, you learned the core distinction between supervised learning (learning from labeled examples to predict a target) and unsupervised learning (finding structure in unlabeled data). Within supervised learning, you can distinguish regression (continuous target) from classification (discrete target). Within unsupervised learning, clustering finds groups while dimensionality reduction compresses features. These categories are your first filter when approaching any ML problem — getting the right category right means you'll be looking at the right family of algorithms from the start. In the next lesson, you'll focus on the practical challenge of **collecting** the data those algorithms need.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="The dataset has labels (fraudulent / legitimate), making it supervised learning. The target is a discrete category with exactly two possible values, making it binary classification. The goal — predicting the label of a new transaction — is exactly the classification task.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A data scientist has a dataset of 50,000 customer transactions, each labeled as either "fraudulent" or "legitimate." They want to build a system that flags new transactions in real time. What type of ML problem is this?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Unsupervised / Clustering</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Supervised / Binary classification</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Supervised / Regression</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Unsupervised / Dimensionality reduction</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="The defining difference is the presence of labels. Supervised algorithms learn a mapping from inputs to outputs using labeled examples. Unsupervised algorithms explore the data&#039;s structure — finding groups, patterns, or compact representations — without any target variable to guide them.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What is the key characteristic that separates supervised learning from unsupervised learning?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Supervised learning uses more data than unsupervised learning.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Supervised learning requires a GPU; unsupervised learning does not.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Supervised learning requires labeled training data (features paired with correct outputs); unsupervised learning does not use labels.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Supervised learning is used for images; unsupervised learning is used for text.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="There are no predefined labels — the retailer doesn&#039;t know the &quot;correct&quot; product groupings in advance. The goal is to discover natural structure in the data. This is a classic unsupervised clustering problem. Regression and classification require a target variable to predict, and reinforcement learning requires an environment and reward signal — neither applies here.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A retailer has transaction data showing which products were purchased together across 1 million orders, but has no predefined product categories. They want to discover natural groupings of products that tend to be bought by similar customers. Which approach is most appropriate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Supervised / Regression — predict the number of items in each group.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Supervised / Multi-class classification — classify each transaction into a product group.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Unsupervised / Clustering — find natural groups in purchasing behavior without predefined labels.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Reinforcement learning — reward the model when it correctly groups products.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

