# Intro to Capstone I: Regression

## Overview

You've spent the last four modules building a regression toolkit: data preparation (module 07–08), classification concepts that transfer directly to regression (module 09), and in module 11 you learned linear regression, polynomial regression, regularization, and the full evaluation framework. This capstone puts all of it together — on a dataset you choose, for a question you define.

Unlike the case studies in modules 08 and 10, which walked you through a specific dataset with detailed instructions, this capstone is **student-directed**. You'll select a dataset, frame the regression problem, perform EDA, clean and engineer features, train and compare models, and present your findings — just as a practicing data scientist would.

## Learning Objectives

By the end of this capstone, you will have demonstrated the ability to:

- Independently frame and execute a regression project from raw data to evaluated model.
- Apply EDA, cleaning, feature engineering, and model evaluation in an integrated workflow.
- Communicate findings clearly with code, visualizations, and written interpretation.

## What This Capstone Is

The capstone is a **single Jupyter notebook** hosted on GitHub. It is not a set of exercises with predetermined answers — it is an original analysis that you design and carry out. The notebook must tell a coherent story: why this dataset, what you found, what you built, and what it means.

The capstone covers three phases:

| Phase | Lesson | What you produce |
|-------|--------|-----------------|
| Planning & Exploration | Project Planning & Dataset Exploration | Problem statement, EDA with visualizations, cleaned dataset |
| Modeling & Evaluation | Modeling & Evaluation | Three trained models, comparison table, residual analysis, tuned final model |
| Submission | — | GitHub link, 3-paragraph written summary |

## Deliverables

### 1. Jupyter Notebook on GitHub

Your notebook must include:

- **Problem statement** — one paragraph explaining what you're predicting, why it matters, and what dataset you're using
- **Dataset description** — source, size, features, and target variable
- **EDA** — at least five visualizations with written interpretation of each
- **Data cleaning** — documented decisions (what you changed and why)
- **Feature engineering** — at least two engineered or transformed features with justification
- **Three models** — trained and evaluated with the same metrics
- **Model comparison** — a table showing MAE, RMSE, and R² for all models
- **Hyperparameter tuning** — at least one model tuned with cross-validation
- **Residual analysis** — residual vs. predicted plot and histogram for your best model
- **Conclusion** — written summary of findings and honest assessment of limitations

### 2. Written Summary (3 paragraphs)

Submitted alongside the notebook link. Write as if explaining to a technically literate colleague who has not seen your notebook:

- **Paragraph 1:** What dataset did you choose and what question are you answering?
- **Paragraph 2:** What did EDA reveal about the data, and what cleaning decisions did that drive?
- **Paragraph 3:** Which model performed best and why? What does the error mean in practical terms?

## Choosing a Dataset

### Requirements

Your dataset must:
- Have a **continuous numeric target variable** (not a category or count that can't take fractional values)
- Have at least **500 rows** and **5 features** (more is better)
- Require at least **some cleaning** — datasets with no issues provide no opportunity to demonstrate those skills
- Be publicly available and citable

**Avoid:** The California Housing dataset (used throughout module 11) and the Titanic dataset (used in earlier modules).

### Suggested Datasets

The following datasets work well for a regression capstone. All are freely available.

| Dataset | Target variable | Source | Difficulty |
|---------|----------------|--------|------------|
| **Ames Housing** | House sale price | Kaggle (`house-prices-advanced-regression-techniques`) | Medium-High (80 features, requires thoughtful encoding) |
| **Medical Insurance** | Insurance charges | Kaggle (`insurance`) | Low-Medium (clean, 6 features, clear relationships) |
| **Bike Sharing (UCI)** | Daily bike rentals | UCI ML Repository | Medium (time features, seasonality) |
| **Diamonds** | Diamond price | Built into seaborn (`sns.load_dataset("diamonds")`) | Low-Medium (mix of numeric and categorical) |
| **Used Cars** | Car listing price | Kaggle (multiple datasets) | Medium (missing values, text cleaning) |
| **Wine Quality** | Quality score (3–9) | UCI ML Repository | Medium (correlated features, borderline regression/classification) |
| **Energy Efficiency** | Heating/cooling load | UCI ML Repository | Low (clean, 8 features, strong linear signal) |
| **Student Performance** | Final grade | UCI ML Repository | Low-Medium (mix of numeric and categorical) |

You are not limited to this list. If you find a dataset that interests you, discuss it with your instructor before committing.

### Evaluating a Dataset Before Committing

Before choosing, spend 10 minutes doing a quick scan:

```python
import pandas as pd

df = pd.read_csv("your_dataset.csv")

# Is the target continuous?
print(df["target"].describe())
print(df["target"].nunique(), "unique values")

# How much missing data?
print(df.isnull().mean().sort_values(ascending=False).head(10))

# Any categorical columns to encode?
print(df.dtypes.value_counts())

# Is there a reasonable spread in the target?
print(df["target"].hist(bins=30))
```

A good dataset for this capstone has:
- A target with at least 50 unique values (genuinely continuous)
- Some but not excessive missing data (5–30% in some columns is fine)
- A mix of numeric and categorical features (gives you something to encode)
- A target distribution that isn't perfectly uniform or perfectly bimodal

### Red Flags

Avoid datasets where:
- The target is already perfectly predicted by one feature (nothing to model)
- More than 50% of values are missing in key columns (cleaning becomes the whole project)
- Every feature is a free-text string (requires NLP, outside scope of this capstone)
- Fewer than 200 samples after cleaning (too small for reliable train/test splits)

## Assessment Rubric

Your capstone will be evaluated on the following dimensions:

| Dimension | Full credit | Partial credit | No credit |
|-----------|-------------|----------------|-----------|
| **Problem framing** | Clear regression problem with a specific target; motivation explained | Target identified but motivation vague | No clear problem statement |
| **EDA** | 5+ visualizations; each interpreted in writing; findings drive cleaning decisions | Visualizations present but uninterpreted | Only summary statistics; no plots |
| **Data cleaning** | Every decision documented with rationale; no silent changes | Some decisions documented | Changes made without explanation |
| **Feature engineering** | At least 2 engineered features; encoding appropriate for feature type | One engineered feature | No feature engineering; raw data fed to model |
| **Model training** | 3 models trained; all evaluated with same metrics on the same test set | 2 models trained | Only 1 model |
| **Evaluation** | MAE, RMSE, R² reported; cross-validation used; residuals analyzed | Metrics reported; no cross-validation | Only accuracy or a single metric |
| **Hyperparameter tuning** | At least one model tuned with GridSearchCV or equivalent CV-based search | Manual tuning without CV | No tuning |
| **Conclusion** | Honest interpretation of best model; limitations acknowledged | Results stated without interpretation | No conclusion |
| **Code quality** | Notebook runs top-to-bottom without errors; clear section structure | Minor errors; mostly readable | Does not run |

## Timeline Guidance

This capstone spans two lessons. A suggested pace:

**During Lesson 2 (Project Planning & Dataset Exploration):**
- Choose dataset and write problem statement
- Complete EDA (all visualizations and interpretations)
- Complete data cleaning
- Complete feature engineering and train/test split

**During Lesson 3 (Modeling & Evaluation):**
- Train and evaluate all three models
- Tune your best model with cross-validation
- Analyze residuals
- Write conclusion and three-paragraph summary

## What's Next

In the next lesson, you'll plan your project in detail — framing the regression problem, selecting features, performing EDA, and building a clean, engineered feature matrix ready for modeling.
