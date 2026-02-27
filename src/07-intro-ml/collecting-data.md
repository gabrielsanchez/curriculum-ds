# Collecting Data

## Overview

In the previous lesson, you learned how to categorize ML problems and match them to the right approach. Every ML model depends entirely on the quality and quantity of its training data — and data has to come from somewhere. In this lesson, you'll explore the main sources of data for ML projects, learn how to access public datasets in Python, and understand the properties that make data useful (or dangerous) for training a model.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Explore methods for data acquisition (public datasets, APIs, web scraping).
- Understand the importance of data quantity and quality for ML.

## Key terms

**Dataset:** A structured collection of data used for analysis or model training, typically organized as rows (observations) and columns (features).

**Public dataset:** A dataset made freely available by governments, research institutions, or companies for public use.

**API (Application Programming Interface):** A standardized interface that allows programs to request data from an external service.

**Web scraping:** Programmatically extracting data from web pages using tools like `requests` and `BeautifulSoup`.

**Data bias:** A systematic skew in a dataset that causes a model trained on it to make unfair or inaccurate predictions for certain groups or scenarios.

**Class imbalance:** A condition where one target class has far fewer examples than another (e.g., 99% non-fraud, 1% fraud), which can cause a model to ignore the minority class.

**Train/test split:** Dividing a dataset into a training portion (used to fit the model) and a held-out test portion (used to evaluate generalization).

**`train_test_split`:** The scikit-learn function that randomly splits a dataset into training and test sets.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/07-intro-ml/03_collecting-data_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Introduction

The saying "garbage in, garbage out" is nowhere more true than in machine learning. A sophisticated algorithm trained on bad data will produce bad predictions. Before worrying about which model to use or how to tune it, you need data — enough of it, representative of the problem you're solving, and labeled correctly.

In practice, data scientists spend significant time sourcing, evaluating, and combining datasets. This lesson surveys the main acquisition methods and the quality considerations that determine whether a dataset is actually fit for training.

## Sources of Data

### 1. Public Datasets

The fastest path to a working ML project is a pre-collected, clean public dataset. Thousands are freely available:

**General repositories:**
- [**Kaggle Datasets**](https://www.kaggle.com/datasets) — Tens of thousands of community-contributed datasets across every domain, with associated notebooks and competitions.
- [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/ml/index.php) — Classic benchmark datasets used in research since the 1980s (Iris, Wine, Adult Income, etc.).
- [**Google Dataset Search**](https://datasetsearch.research.google.com/) — A search engine specifically for datasets across the web.

**Government and institutional data:**
- [**data.gov**](https://www.data.gov) — U.S. government open data (census, health, transportation, education).
- [**World Bank Open Data**](https://data.worldbank.org/) — Global economic and development statistics.
- [**CDC / NIH**](https://www.cdc.gov/datastatistics/) — Public health and biomedical data.

**Built into Python libraries:**
Several Python libraries ship with ready-to-use datasets — perfect for practice and experimentation.

```python
# scikit-learn built-in datasets
from sklearn.datasets import (
    load_iris,           # Classic flower classification (150 rows, 3 classes)
    load_wine,           # Wine variety classification (178 rows, 3 classes)
    load_breast_cancer,  # Binary cancer classification (569 rows)
    load_diabetes,       # Diabetes regression (442 rows)
    fetch_california_housing  # Housing price regression (20,640 rows)
)

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
print(df.shape)      # (150, 5)
print(df.head())
```

```python
# seaborn also ships with several well-known datasets
import seaborn as sns

titanic  = sns.load_dataset("titanic")   # Passenger survival (891 rows)
tips     = sns.load_dataset("tips")      # Restaurant tips (244 rows)
diamonds = sns.load_dataset("diamonds")  # Diamond prices (53,940 rows)
penguins = sns.load_dataset("penguins")  # Palmer penguins (344 rows)

print(titanic.shape)
print(titanic.head())
```

### 2. APIs

An API returns structured data (usually JSON) in response to HTTP requests. Most major platforms — Twitter/X, Spotify, OpenWeather, the U.S. Census Bureau — expose data via APIs.

```python
import requests
import pandas as pd

# Example: Open-Meteo free weather API (no API key required)
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 41.85,
    "longitude": -87.65,
    "daily": "temperature_2m_max,temperature_2m_min",
    "forecast_days": 7,
    "timezone": "America/Chicago"
}

response = requests.get(url, params=params)
data = response.json()

df_weather = pd.DataFrame({
    "date":    data["daily"]["time"],
    "temp_max": data["daily"]["temperature_2m_max"],
    "temp_min": data["daily"]["temperature_2m_min"]
})
print(df_weather)
```

Most commercial APIs require an API key (a token that authenticates your requests). Keep keys secret — never commit them to a public GitHub repository.

### 3. Web Scraping

When data is displayed on a website but not available via an API or download, you can programmatically extract it using `requests` (to fetch the HTML) and `BeautifulSoup` (to parse it):

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the first table on the page
table = soup.find("table", {"class": "wikitable"})
df = pd.read_html(str(table))[0]
print(df.head())
```

> **Important:** Always check a website's `robots.txt` and Terms of Service before scraping. Scraping is not always permitted, and excessive requests can overload servers.

### 4. Internal / Organizational Data

In industry, most ML training data comes from internal systems:
- **Databases** (SQL, as you've learned) — transaction logs, customer records, sensor readings
- **Data warehouses** — aggregated historical data from multiple systems
- **Event logs** — clickstreams, app usage, server logs
- **Manually labeled data** — human annotators who label images, text, or audio for training

For sensitive internal data, privacy regulations (GDPR, HIPAA) govern how data can be collected, stored, and used. Data governance is a real responsibility.

### 5. Synthetic Data

When real data is scarce, too sensitive, or imbalanced, **synthetic data** — artificially generated to mimic real distributions — can supplement training:

```python
from sklearn.datasets import make_classification, make_regression

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_classes=2,
    random_state=42
)
print(f"X shape: {X.shape}, class distribution: {y.mean():.2f}")
```

## Data Quantity and Quality

### How Much Data Do You Need?

There is no universal answer, but general guidance:

| Model complexity | Typical data requirement |
|-----------------|------------------------|
| Linear models (logistic/linear regression) | Hundreds to thousands of rows |
| Tree-based models (random forests, gradient boosting) | Thousands to tens of thousands |
| Deep neural networks | Tens of thousands to millions |

These are rough estimates. The required amount also depends on the number of features, the complexity of the pattern, and the acceptable error rate.

**The danger of too little data:** The model memorizes the training examples (overfitting) rather than learning the underlying pattern. It performs well on training data but poorly on new data.

**The danger of too much noise:** Even large datasets are harmful if they're mislabeled, full of errors, or irrelevant to the problem.

### Data Quality Dimensions

Quantity matters, but quality matters more:

| Dimension | What it means | Why it matters |
|-----------|--------------|----------------|
| **Accuracy** | Values are correct | Mislabeled examples teach the model wrong patterns |
| **Completeness** | Missing values are minimal | Missing features reduce the signal available to the model |
| **Consistency** | Same facts are represented the same way | Inconsistent encoding (e.g., "NY" vs "New York") creates duplicate signals |
| **Representativeness** | Data covers the full range of cases the model will face | Non-representative data leads to poor generalization |
| **Timeliness** | Data reflects current patterns | Stale data may not reflect the present-day distribution |

### Bias and Representativeness

**Data bias** is one of the most serious risks in ML. A model trained on biased data learns those biases and amplifies them.

Examples:
- A hiring algorithm trained on historical hiring decisions that under-represented women will learn to discriminate against women.
- A medical diagnosis model trained only on patients from one demographic will perform poorly for patients from other demographics.
- A facial recognition system trained mostly on light-skinned faces will perform worse on darker-skinned faces.

**How to reduce bias:**
- Ensure your training data represents the full population the model will serve.
- Examine the data distribution across demographic groups before training.
- Audit model performance separately for different subgroups.

### Class Imbalance

In classification problems, one class often has far fewer examples:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
print(df["survived"].value_counts(normalize=True))
```

Output:
```
0    0.617
1    0.383
Name: survived, dtype: float64
```

The Titanic dataset is moderately imbalanced (62% did not survive). A model that always predicted "did not survive" would be 61.7% accurate — without learning anything. Techniques to handle imbalance include oversampling the minority class, undersampling the majority class, or using class weights in the algorithm.

## The Train/Test Split

Before any data work begins, split your labeled dataset into a **training set** and a **test set**. The test set must remain unseen until final evaluation — it is your measure of real-world performance.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = sns.load_dataset("titanic").dropna(subset=["survived"])

X = df[["pclass", "age", "fare", "sibsp", "parch"]].fillna(0)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Reproducibility
    stratify=y           # Preserve class proportions in both splits
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set:     {X_test.shape[0]} rows")
print(f"Train class balance: {y_train.mean():.2f}")
print(f"Test class balance:  {y_test.mean():.2f}")
```

Output:
```
Training set: 712 rows
Test set:     179 rows
Train class balance: 0.38
Test class balance:  0.39
```

`stratify=y` ensures the class proportions in the training and test sets match the overall dataset — important for imbalanced problems.

## Conclusion

In this lesson, you explored the main methods for acquiring ML training data — public datasets (including those built into scikit-learn and seaborn), APIs, web scraping, internal databases, and synthetic data generation. You also learned the data quality dimensions that matter for training — accuracy, completeness, representativeness, and class balance — and why poor-quality data produces poor models regardless of algorithmic sophistication. Finally, you performed your first `train_test_split`. In the next lesson, you'll learn how to prepare raw data for ML through **data cleaning**.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/07-intro-ml/03_collecting-data_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="When one class vastly outnumbers another, a naive model can achieve high overall accuracy by simply predicting the majority class every time. In fraud detection, this would mean 99% accuracy but zero fraud caught — a useless model. This is class imbalance, and it requires special handling (oversampling, class weights, or evaluation metrics beyond accuracy like precision/recall).">
  <div class="quiz-question">
    <strong>Question 1:</strong> You load a dataset to train a fraud detection model. The dataset contains 99,000 legitimate transactions and 1,000 fraudulent ones. What problem does this create, and what term describes it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Overfitting — the model has seen too many examples of fraud.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Class imbalance — the model may learn to always predict "legitimate" and achieve 99% accuracy without detecting any fraud.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Data leakage — the fraud labels have contaminated the training features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Underfitting — the dataset is too large for the algorithm to process.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The test set simulates data the model will encounter after deployment. If you use it to make modeling decisions (tuning parameters, selecting features, choosing algorithms), those decisions are implicitly optimized for the test set — which then no longer measures true unseen performance. The test set should be opened once, at the very end, to report final performance.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why should the test set be kept completely separate and unseen until final evaluation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Test data must be stored in a different file format than training data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>If the model is tuned or evaluated on the test set during development, performance estimates become overly optimistic and no longer reflect true generalization.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Scikit-learn's `train_test_split` automatically hides the test set from all functions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The test set needs to be smaller than the training set, so it must be separated in advance.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="ML models learn whatever patterns exist in their training data — including biased ones. If historical hiring decisions systematically disadvantaged women, the model learns that pattern and perpetuates it at scale. This is a real-world example of data bias with serious ethical consequences. The fix is not a better algorithm — it&#039;s examining and correcting the training data, auditing model outputs across demographic groups, and building fairness constraints into the modeling process.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A company builds a resume-screening ML model trained on 10 years of past hiring decisions. After deployment, it consistently rates female candidates lower than male candidates with equivalent qualifications. What is the most likely cause?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The algorithm selected was not designed for fairness.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The training data reflected historical hiring bias — the model learned and replicated the discriminatory patterns present in past decisions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The model was overfitted to the training data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The dataset was too small to include enough female candidates.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

