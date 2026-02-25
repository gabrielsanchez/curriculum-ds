# Introduction to the Case Study

## Overview

You've spent the last two modules building the foundational skills of the machine learning pipeline: sourcing and evaluating data, cleaning it systematically, and engineering features that algorithms can use. Now you'll apply all of those skills together on a real-world dataset — without a guided checklist to follow. This case study walks you through the complete data preparation workflow on the **Kaggle House Rent Prediction Dataset**, a property rental dataset covering six major Indian cities. By the end of the module, you'll have a clean, engineered feature matrix ready for a regression model.

## Learning Objectives

By the end of this case study, you will have learned how to:

- Understand the context and goals of the house rental dataset.
- Outline the questions to be answered or hypotheses to test.

## What Is a Case Study?

The previous modules taught individual skills in isolation — one lesson on missing values, one on encoding, one on scaling. Real data science work doesn't look like that. A case study gives you an end-to-end problem: messy raw data and a business question. Your job is to figure out what the data needs, in what order, using the skills you've built.

This module is structured as a four-step walkthrough:

1. **Exploratory Data Analysis** — Understand the data's shape, distributions, and relationships before touching anything.
2. **Data Cleaning** — Fix quality issues discovered during EDA: parse complex columns, handle outliers, and validate logical constraints.
3. **Feature Engineering** — Create new variables, encode categoricals, and scale numerics for downstream modeling.
4. **Summary** — Synthesize findings, assess the final dataset, and define next steps for a regression model.

## The Dataset

The **House Rent Prediction Dataset** is published on Kaggle and contains 4,746 rental listings scraped from an Indian property platform. Each row describes a single rental property.

### Columns

| Column | Description | Type |
|--------|-------------|------|
| `Posted On` | Date the listing was posted | String (date) |
| `BHK` | Number of bedrooms, halls, and kitchens | Integer |
| `Rent` | Monthly rent in Indian Rupees (INR) | Integer — **this is the target** |
| `Size` | Size of the property in square feet | Integer |
| `Floor` | Floor of the unit and total floors in building | String (e.g., `"Ground out of 2"`, `"1 out of 3"`) |
| `Area Type` | How the size is measured (Super Area, Carpet Area, Built Area) | Categorical |
| `Area Locality` | Specific neighborhood or locality | Categorical (high cardinality) |
| `City` | City where the property is located | Categorical (6 cities) |
| `Furnishing Status` | Furnished / Semi-Furnished / Unfurnished | Categorical |
| `Tenant Preferred` | Bachelors / Family / Bachelors/Family | Categorical |
| `Bathroom` | Number of bathrooms | Integer |
| `Point of Contact` | Contact Owner / Contact Agent / Contact Builder | Categorical |

### The Business Question

> **What factors drive rental prices, and how can we prepare this data to accurately predict the monthly rent for a new listing?**

Answering this question requires understanding the data first — which features correlate with rent, which contain noise or errors, and which need transformation before a model can use them.

## Starter Code

Use the included [*Colaboratory notebook*](#) to follow along with the case study. The notebook contains all code blocks from all four lessons — you can run each cell as you read, modify values to experiment, and add your own analysis.

## Loading the Dataset

Download the dataset from Kaggle (`House_Rent_Dataset.csv`) and upload it to your Colab session, or mount Google Drive:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("House_Rent_Dataset.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
df.head()
```

Output:
```
Shape: (4746, 12)

Columns:
Posted On             object
BHK                    int64
Rent                   int64
Size                   int64
Floor                 object
Area Type             object
Area Locality         object
City                  object
Furnishing Status     object
Tenant Preferred      object
Bathroom               int64
Point of Contact      object
dtype: object
```

The dataset has 4,746 rental listings across 12 columns — a manageable size that still captures the full complexity of a real-world data preparation problem.

## Initial Impressions

Before diving into analysis, it helps to scan the data with fresh eyes:

```python
print(df.isnull().sum())
```

Output:
```
Posted On            0
BHK                  0
Rent                 0
Size                 0
Floor                0
Area Type            0
Area Locality        0
City                 0
Furnishing Status    0
Tenant Preferred     0
Bathroom             0
Point of Contact     0
dtype: int64
```

No missing values — but that doesn't mean the data is clean. Hidden quality issues are often more dangerous than obvious `NaN`s:
- Is the `Rent` column free of implausible extremes?
- What exactly does `"Ground out of 2"` mean in the `Floor` column?
- Does every listing have a logical relationship between `BHK` and `Bathroom`?
- How many unique values does `Area Locality` have — is it usable as a feature?

These are the questions the next three lessons will answer. The process of finding and fixing them is the case study.

## The Six Cities

```python
print(df["City"].value_counts())
print(f"\nCities: {df['City'].unique()}")
```

Output:
```
Mumbai        1093
Chennai        948
Bangalore      884
Hyderabad      838
Delhi          774
Kolkata        209
Name: City, dtype: int64

Cities: ['Kolkata' 'Mumbai' 'Bangalore' 'Hyderabad' 'Delhi' 'Chennai']
```

The dataset spans six major Indian cities, with Mumbai most represented and Kolkata least. City is likely an important feature — rental markets vary significantly by location.

## What's Next

In the next lesson, you'll perform **Exploratory Data Analysis**: examining the distribution of rent, how features relate to price, and what patterns and anomalies exist in the data before any cleaning begins.
