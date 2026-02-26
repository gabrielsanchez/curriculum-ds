# 11 Machine Learning: Regression

This module covers regression — predicting a continuous numeric value from features. Building on the data preparation skills from module 08 and the supervised learning foundation from module 09, you'll learn how linear regression works, how to handle non-linearity, how to prevent overfitting with regularization, and how to evaluate models rigorously.

All lessons use the California Housing dataset (built into scikit-learn) so you can focus on the methods rather than data preparation. The module culminates in a complete model comparison showing when and why more complex methods outperform the linear baseline.

## Lessons

1. **Introduction to Regression** — Regression vs. classification, use cases, the regression pipeline, and a first end-to-end model.
2. **Linear Regression** — Ordinary least squares, coefficient interpretation, residual analysis, and the four assumptions of linear regression.
3. **Additional Regression Methods** — Polynomial regression for non-linear relationships; decision tree regression for threshold-based splits.
4. **Regularization** — Ridge (L2), Lasso (L1), and ElasticNet for controlling overfitting; automatic alpha selection with cross-validation.
5. **Model Evaluation for Regression** — MAE, RMSE, R², and adjusted R²; cross-validation for reliable estimates; hyperparameter tuning with `GridSearchCV`; final model comparison.
