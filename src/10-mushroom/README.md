# 10 Case Study: Mushroom Identification

This case study applies the full classification pipeline — EDA, cleaning, feature engineering, and model selection — to the UCI Mushroom Dataset, a benchmark problem with clear real-world stakes: predicting whether a mushroom is edible or poisonous from its physical features.

Unlike most benchmark datasets, every feature in the mushroom dataset is categorical, making encoding the central challenge of preparation. The dataset is also highly separable, which makes it an ideal setting for understanding *why* a model works and what drives its decisions — rather than just tuning until accuracy improves.

The safety-critical framing reinforces a core lesson from module 09: accuracy is not enough. A false negative — predicting "edible" when the mushroom is poisonous — is categorically worse than a false positive. Metric selection, threshold tuning, and domain reasoning all follow from that constraint.

## Lessons

1. **Introduction to the Case Study** — Problem framing, dataset description, loading the data, and initial impressions.
2. **Exploratory Data Analysis** — Feature distributions, poisonous rate by feature value, Cramér's V associations, and identifying the dominant predictors.
3. **Data Cleaning** — Handling `?`-encoded missing values, dropping zero-variance features, and validating the cleaned dataset.
4. **Feature Engineering** — Label encoding all categorical features, train/test split with stratification.
5. **Model Selection** — Training logistic regression, decision tree, and random forest; comparing on safety-relevant metrics; threshold tuning; feature importance analysis.
6. **Summary** — Synthesizing findings, reflecting on safety-critical classification, and outlining next steps.
