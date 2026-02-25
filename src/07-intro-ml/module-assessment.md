# Module Assessment

## Overview

In this module, you learned the foundational concepts and practical skills of machine learning: what ML is and how it differs from traditional programming, the distinction between supervised and unsupervised learning, how to source and evaluate data, how to clean raw data for ML, and how to engineer features that models can actually use. This assessment asks you to demonstrate all of those skills together — building a complete data preparation pipeline from a raw dataset to a processed feature matrix ready for a model.

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Show readiness for more complex ML modules.
- Demonstrate basic data prep and understanding of ML tasks.

## Assessment Overview

You'll work with the **Palmer Penguins dataset** — a classic ML dataset with realistic data quality issues, categorical variables, and numeric features at different scales. Your task is to take it from raw to ready: cleaned, encoded, scaled, and split for supervised classification.

### Skills Assessed

| Skill | Lessons covered |
|-------|----------------|
| Identify the ML problem type | Supervised vs. Unsupervised |
| Load and inspect a dataset | Collecting Data |
| Handle missing values and duplicates | Cleaning Data |
| Drop irrelevant or leaking columns | Cleaning Data |
| Encode categorical variables | Feature Engineering |
| Scale numeric features | Feature Engineering |
| Create engineered features | Feature Engineering |
| Perform a stratified train/test split | Collecting Data, Feature Engineering |
| Assemble a `ColumnTransformer` pipeline | Feature Engineering |

## Starter Code

Use the included [*Colaboratory notebook*](#) to complete the tasks below. Submit the link to your saved notebook for grading.

## Coding Assessment

### The Dataset

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("penguins")
print(df.shape)
print(df.head())
print(df.isnull().sum())
```

The penguins dataset contains 344 rows and these columns:

| Column | Description | Type |
|--------|-------------|------|
| `species` | Penguin species (Adelie, Chinstrap, Gentoo) | Categorical — **this is the target** |
| `island` | Island where observed (Biscoe, Dream, Torgersen) | Categorical |
| `bill_length_mm` | Bill length in millimeters | Numeric |
| `bill_depth_mm` | Bill depth in millimeters | Numeric |
| `flipper_length_mm` | Flipper length in millimeters | Numeric |
| `body_mass_g` | Body mass in grams | Numeric |
| `sex` | Penguin sex (Male, Female) | Categorical |

---

### Task 1: Identify the ML Problem Type

In a markdown cell, answer the following:

1. Is this a supervised or unsupervised learning problem? Why?
2. Is this a classification or regression task? Why?
3. Is this binary classification or multi-class classification? Why?

---

### Task 2: Inspect and Clean the Data

Perform the following steps:

1. Print the shape, data types, and missing value counts.
2. Drop any duplicate rows.
3. Handle missing values:
   - For numeric columns (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`): impute with the **median**.
   - For the categorical column `sex`: impute with the **mode**.
4. After cleaning, confirm there are zero missing values with `df.isnull().sum()`.

```python
# Your cleaning code here
```

---

### Task 3: Define Features and Target

1. Separate the features (`X`) and target (`y`):
   - `y` = `species`
   - `X` = all other columns
2. Perform a **stratified** train/test split: 80% train, 20% test, `random_state=42`.
3. Print the shape of `X_train` and `X_test`.
4. Print the class distribution in `y_train` using `value_counts(normalize=True)`.

```python
from sklearn.model_selection import train_test_split

# Your code here
```

---

### Task 4: Engineer a New Feature

Before building the preprocessing pipeline, create one new feature on both `X_train` and `X_test`:

- `bill_ratio`: `bill_length_mm` divided by `bill_depth_mm`. This captures the shape of the bill regardless of absolute size.

```python
# Add bill_ratio to X_train and X_test
X_train = X_train.copy()
X_test  = X_test.copy()

X_train["bill_ratio"] = ...
X_test["bill_ratio"]  = ...
```

---

### Task 5: Build a `ColumnTransformer` Pipeline

Build a preprocessing pipeline using `ColumnTransformer` that:

- **Numeric columns** (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `bill_ratio`): apply `StandardScaler`
- **Categorical columns** (`island`, `sex`): apply `OneHotEncoder(drop="first", sparse_output=False)`

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

numeric_features     = ["bill_length_mm", "bill_depth_mm",
                         "flipper_length_mm", "body_mass_g", "bill_ratio"]
categorical_features = ["island", "sex"]

# Build numeric_transformer, categorical_transformer, and preprocessor here
```

Fit the preprocessor on training data and transform both sets:

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape:  {X_test_processed.shape}")
```

---

### Task 6: Verify the Output

After processing, confirm the following:

1. No `NaN` values exist in `X_train_processed` (hint: `np.isnan(X_train_processed).sum()`).
2. The numeric columns have approximately mean = 0 and std = 1. Compute the column-wise mean and standard deviation of the numeric portion of `X_train_processed`.
3. Print how many total features are in the processed output and list what they represent (e.g., 5 scaled numeric + how many one-hot columns?).

---

### Task 7: Reflection

In a markdown cell, answer the following questions:

1. Why must the `ColumnTransformer` be **fitted on training data only** and then used to **transform** the test set — rather than being fitted on the full dataset?
2. You created `bill_ratio` as an engineered feature. Propose one additional feature you could create from the existing columns and explain what signal you think it would capture.
3. The penguins dataset has three species. If you trained a model and it achieved 95% accuracy, but 40% of the rows are Adelie penguins — would accuracy alone be sufficient to evaluate the model? What other metric(s) might you look at?

---

## Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Problem identification | 10 | Correctly identifies supervised multi-class classification with justification |
| Task 2: Cleaning | 20 | Removes duplicates, imputes all missing values correctly, confirms zero NaN |
| Task 3: Train/test split | 15 | Correct stratified split, prints shapes and class balance |
| Task 4: Feature engineering | 15 | `bill_ratio` correctly computed on both train and test sets |
| Task 5: ColumnTransformer | 25 | Pipeline built correctly, fit on train only, transform applied to both sets |
| Task 6: Verification | 10 | Confirms no NaN, verifies scaled statistics, explains output shape |
| Task 7: Reflection | 5 | Thoughtful answers demonstrating conceptual understanding |
| **Total** | **100** | |

## Knowledge Check

#### **Question 1: You are building a model to classify penguin species using body measurements. After completing feature engineering, you realize you accidentally called `preprocessor.fit_transform()` on the combined train + test data instead of training data only. What problem does this cause?**
1. The preprocessor will produce incorrect output shapes.
2. The scaling statistics (mean, standard deviation) will be computed using test set values, which constitutes data leakage — making performance estimates on the test set artificially optimistic.
3. `ColumnTransformer` will raise an error if fitted on more than 80% of the data.
4. One-hot encoding will create different columns for train and test, causing a shape mismatch.

**Correct Answer:**
2. The scaling statistics (mean, standard deviation) will be computed using test set values, which constitutes data leakage — making performance estimates on the test set artificially optimistic.

**Explanation:**
`fit_transform()` on the combined dataset allows test set statistics to influence the scaler's learned mean and standard deviation. In practice, when a model is deployed, you only have training data when fitting preprocessors — future records must be transformed using statistics from the training distribution. Fitting on the full dataset leaks future information into the preprocessing step, making the model appear to perform better than it will in production.

---

#### **Question 2: The penguins dataset has three species: Adelie (44%), Chinstrap (20%), and Gentoo (36%). When you call `train_test_split`, you use `stratify=y`. Why?**
1. `stratify=y` makes the split faster by sorting the data first.
2. Without stratification, the random split might place all Chinstrap penguins in the training set and none in the test set, making evaluation unreliable for underrepresented classes.
3. `stratify=y` ensures the test set is exactly 20% of the data.
4. Stratification is required when the target is a string column rather than a numeric column.

**Correct Answer:**
2. Without stratification, the random split might place all Chinstrap penguins in the training set and none in the test set, making evaluation unreliable for underrepresented classes.

**Explanation:**
When class proportions are unequal, a purely random split can produce a test set where a minority class is underrepresented or even absent. `stratify=y` ensures each class appears in the train and test sets in proportion to its frequency in the full dataset. This guarantees that performance is measured on a representative sample of all classes — especially important for the Chinstrap class, which makes up only 20% of the data.

---

#### **Question 3: A teammate suggests skipping feature scaling because "the model will figure out the right weights anyway." For which of the following algorithms is this reasoning most problematic?**
1. A decision tree classifier — decision trees split on individual feature thresholds.
2. A random forest classifier — random forests average many decision trees.
3. A k-nearest neighbors classifier — KNN computes Euclidean distance between points and will treat `body_mass_g` (in grams, ~3500–6000) as orders of magnitude more important than `bill_depth_mm` (in mm, ~13–21).
4. A gradient boosting classifier — gradient boosting is tree-based and scale-invariant.

**Correct Answer:**
3. A k-nearest neighbors classifier — KNN computes Euclidean distance between points and will treat `body_mass_g` (in grams, ~3500–6000) as orders of magnitude more important than `bill_depth_mm` (in mm, ~13–21).

**Explanation:**
Tree-based algorithms (decision trees, random forests, gradient boosting) split on individual feature values and are unaffected by scale. KNN, however, computes the distance between data points across all features simultaneously. A `body_mass_g` difference of 500 grams dwarfs a `bill_depth_mm` difference of 2 mm in Euclidean distance — not because body mass is more predictive, but simply because it's measured in a larger unit. Without scaling, KNN effectively ignores low-magnitude features. `StandardScaler` ensures each feature contributes proportionally to distance calculations.
