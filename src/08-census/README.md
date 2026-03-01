# Case Study: Census Income

This case study is the first applied project in the course. Rather than introducing new methods, it requires you to use everything from modules 03–07 — Pandas, EDA, data cleaning, and feature engineering — on a real-world census dataset with all the messiness that real data brings.

Working through a case study without step-by-step instructions develops the judgment that distinguishes a practitioner from someone who can only follow a tutorial. Each lesson poses an analytical question and asks you to answer it using the tools you already have.

This dataset also introduces an early opportunity to think about the non-technical dimensions of a data science project: who does the model affect, what features should be in it, and what does "good performance" mean when error costs are asymmetric.

## Lessons

1. **Introduction to the Case Study** — Dataset overview, problem framing, initial data loading, and first impressions from `.info()`, `.describe()`, and a missing value scan.
2. **Exploratory Data Analysis** — Income rate by education, occupation, and workclass; age and hours-per-week distributions; capital gain sparsity; native country cardinality.
3. **Data Cleaning** — Handling `?`-encoded missing values, dropping the non-predictive `fnlwgt` column, removing the redundant `education` string column, and deduplicating.
4. **Feature Engineering** — Encoding the binary target, creating `native_us` and `has_capital` derived features, one-hot encoding five categorical columns, scaling five continuous columns, and assembling the final 49-feature matrix.
5. **Summary** — Synthesizing findings, reflecting on model fairness considerations, and previewing the binary classification steps that follow in module 09.
