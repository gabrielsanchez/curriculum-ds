# Introduction to Machine Learning

## Overview

Welcome to the Introduction to Machine Learning module! The previous modules equipped you with the full data science toolkit: Python, Pandas, SQL, visualization, and exploratory analysis. Now you'll use all of it to build something that can **learn from data**. In this lesson, you'll learn what machine learning is, how it differs from traditional programming, and how it fits into the broader landscape of artificial intelligence and data science.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Define machine learning and explain how it differs from traditional programming.
- Recognize industry use cases and understand the common ML workflow.

## Key terms

**Machine learning (ML):** A branch of artificial intelligence in which systems learn patterns from data and use those patterns to make predictions or decisions — without being explicitly programmed with rules.

**Model:** The output of a machine learning algorithm after training on data. A model is a mathematical function that maps inputs to predictions.

**Training:** The process of exposing a machine learning algorithm to labeled data so it can learn the relationship between features and outcomes.

**Prediction (inference):** Using a trained model on new, unseen data to produce an output — a class label, a numeric value, or a probability.

**Feature:** An individual measurable input variable used by a model. Also called a predictor or independent variable.

**Label (target):** The output variable the model is trying to predict. Also called the dependent variable or response variable.

**Algorithm:** The mathematical procedure used to learn a model from data (e.g., linear regression, decision tree, neural network).

**Generalization:** A model's ability to perform well on new, unseen data — not just the training data it was built on.

## Introduction

For most of history, software worked by **explicit rules**. A programmer studied a problem, formulated rules, and wrote code to enforce them. A spam filter, for example, might check whether an email contains words like "free money" or "click here" and block it if it does.

This approach works when the rules are known and manageable. But what about problems where the rules are too complex, too numerous, or simply unknown?

- How do you write rules that distinguish a cat from a dog in an image?
- How do you write rules that predict whether a loan will default based on 50 financial variables?
- How do you write rules that recommend the next movie a user will enjoy, given their entire watch history?

These problems resist rule-based solutions. **Machine learning** takes a different approach: instead of telling the computer what rules to follow, you give it **data** — many examples of inputs and their correct outputs — and let it discover the rules itself.

## Traditional Programming vs. Machine Learning

The fundamental shift in thinking:

```
Traditional programming:
  Data + Rules → Output

Machine learning:
  Data + Output → Rules (the model)
```

In traditional programming, a developer writes the logic. In machine learning, the algorithm infers the logic from examples. Once trained, the model applies that learned logic to new inputs — just like a human expert who has seen many cases can make judgments on new ones.

**Example:** Email spam detection

| Approach | How it works |
|----------|--------------|
| Traditional | Programmer writes rules: "if subject contains 'FREE' and sender is unknown, mark as spam" |
| Machine learning | Algorithm analyzes thousands of labeled emails (spam/not spam) and learns which combinations of words, senders, and patterns predict spam — without being told which patterns to look for |

The ML approach finds patterns a programmer would never think to encode — and automatically adapts as spam tactics evolve (by retraining on new data).

## The Machine Learning Workflow

Every ML project follows a similar sequence of steps:

```
1. Define the problem
        ↓
2. Collect data
        ↓
3. Explore & clean data (EDA)
        ↓
4. Engineer features
        ↓
5. Select & train a model
        ↓
6. Evaluate the model
        ↓
7. Deploy & monitor
```

You've already practiced steps 2–4 in depth (Pandas, EDA, visualization). This module focuses on building the conceptual and practical foundation for steps 5 and 6. Later modules will take you through the full cycle end-to-end.

### Step 1: Define the Problem

The most important step is often overlooked. Before writing a line of code, clarify:
- What question are you answering?
- What does a "correct" answer look like?
- What data do you have — or need to collect?
- How will model performance be measured?
- How will the model's output be used?

A vague problem ("improve customer experience") produces a vague model. A specific problem ("predict which customers will cancel their subscription within 30 days") leads to a focused, actionable solution.

### Steps 2–4: Data Work

As you've learned, most of the work in a real ML project is in the data:
- **Collecting** enough high-quality examples
- **Cleaning** the data to fix errors and handle missing values
- **Engineering features** to give the model the most useful signal

A rule of thumb in industry: 80% of a data scientist's time is spent on data preparation; 20% on modeling.

### Steps 5–6: Train and Evaluate

The algorithm learns from the training data to produce a model. The model is then evaluated on **held-out test data** it has never seen — this measures how well it *generalizes* to new situations, which is the only thing that matters in practice.

### Step 7: Deploy

A model that isn't used doesn't create value. Deployment means integrating the model into a product or workflow — a web API, a batch prediction job, a real-time scoring system.

## Industry Use Cases

Machine learning powers many of the systems you interact with daily:

| Domain | ML Application |
|--------|----------------|
| **Finance** | Fraud detection, credit risk scoring, algorithmic trading |
| **Healthcare** | Medical image diagnosis, drug discovery, patient risk prediction |
| **Retail / E-commerce** | Product recommendations, demand forecasting, dynamic pricing |
| **Technology** | Search engines, voice assistants, spam filters, content moderation |
| **Transportation** | Autonomous vehicles, route optimization, predictive maintenance |
| **Marketing** | Customer churn prediction, audience segmentation, ad targeting |
| **Natural Language** | Translation, sentiment analysis, document summarization |

In each case, the pattern is the same: a large quantity of historical data is used to train a model that makes predictions or decisions on new data.

## What ML Is Not

A few important clarifications:

- **ML is not magic.** Models are only as good as the data they're trained on. Garbage in, garbage out.
- **ML is not always the right tool.** A well-written set of business rules often outperforms a complex model on a simple, well-understood problem.
- **ML does not replace domain expertise.** Feature engineering, problem framing, and result interpretation all require human judgment. Domain experts and data scientists need to work together.
- **ML models can be wrong — and biased.** A model trained on historical data will reflect historical patterns, including historical biases. Fairness and responsible use are active areas of research and practice.

## The Tools: scikit-learn

Throughout this module and the ones that follow, you'll use **[scikit-learn](https://scikit-learn.org/)** (imported as `sklearn`) — Python's most widely used machine learning library. It provides:

- A consistent API for dozens of algorithms
- Tools for data preprocessing and feature engineering
- Model evaluation utilities
- Built-in sample datasets for practice

```python
# scikit-learn is pre-installed in Google Colab
import sklearn
print(sklearn.__version__)
```

Every model in scikit-learn follows the same three-step pattern:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()      # 1. Instantiate the algorithm
model.fit(X_train, y_train)     # 2. Train on labeled data
predictions = model.predict(X_test)  # 3. Predict on new data
```

You'll use this pattern constantly. Learning it once means you can apply it to any algorithm in the library.

## Conclusion

In this lesson, you learned that machine learning is a paradigm shift — instead of writing rules, you let algorithms discover rules from data. You saw how this applies to real-world problems across many industries, traced the standard ML workflow from problem definition to deployment, and were introduced to scikit-learn as the primary tool for the rest of this course. In the next lesson, you'll learn how to categorize ML problems — supervised versus unsupervised — and match problem types to the right approach.

## Practice

### Knowledge Check

#### **Question 1: What is the fundamental difference between traditional programming and machine learning?**
1. Traditional programming uses Python; machine learning uses R.
2. Traditional programming uses data and rules to produce output; machine learning uses data and output to discover rules (a model).
3. Machine learning is faster than traditional programming for all tasks.
4. Traditional programming can only handle numeric data; machine learning handles all data types.

**Correct Answer:**
2. Traditional programming uses data and rules to produce output; machine learning uses data and output to discover rules (a model).

**Explanation:**
In traditional programming, a developer explicitly codes the logic that transforms inputs into outputs. In machine learning, the algorithm is given many examples of inputs paired with correct outputs, and it infers the underlying rules (the model) on its own. The trained model can then make predictions on new inputs.

---

#### **Question 2: A company wants to predict whether a new loan application will default. This is a machine learning problem because:**
1. The company has a computer, which is required for machine learning.
2. The rules that determine default are too complex and numerous to write manually, but historical loan data with known outcomes can be used to train a model to learn those patterns.
3. Machine learning is always better than rule-based approaches for financial problems.
4. The problem involves numbers, and machine learning only works with numerical data.

**Correct Answer:**
2. The rules that determine default are too complex and numerous to write manually, but historical loan data with known outcomes can be used to train a model to learn those patterns.

**Explanation:**
Machine learning is most valuable when explicit rules are hard to formulate but labeled examples are available. Loan default depends on dozens of interacting factors (income, debt, employment history, economic conditions) in ways too complex for manual rule-writing. Historical loan records with known default outcomes provide exactly the training signal a model needs to learn those patterns.

---

#### **Question 3: Why is model evaluation performed on held-out test data rather than the training data?**
1. Test data is always cleaner than training data, so it gives more accurate results.
2. Training data is too large to evaluate efficiently.
3. A model evaluated on its own training data will appear to perform better than it actually does on new cases — test data measures real-world generalization ability.
4. Scikit-learn requires a separate test set for technical reasons.

**Correct Answer:**
3. A model evaluated on its own training data will appear to perform better than it actually does on new cases — test data measures real-world generalization ability.

**Explanation:**
A model can "memorize" its training data and score perfectly on it without learning any generalizable patterns. Evaluating on held-out test data — examples the model has never seen — measures whether it has truly learned the underlying relationship or just memorized the training examples. This distinction (training performance vs. generalization performance) is one of the most fundamental concepts in machine learning.
