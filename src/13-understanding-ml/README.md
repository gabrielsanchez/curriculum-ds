# 13 Understanding ML Models

This module introduces five advanced machine learning algorithms and teaches when each one is worth the additional complexity over logistic regression and decision trees. All algorithm lessons use the breast cancer dataset from scikit-learn — a 569-sample binary classification task — enabling direct head-to-head comparison.

The module concludes with an assessment that consolidates all five algorithms into a single comparison table, a threshold-optimization exercise, and reflection questions on model selection trade-offs.

## Lessons

1. **Introduction to Advanced ML Models** — Why learn more algorithms, the breast cancer dataset, logistic regression and decision tree baselines (LR: 95.6% accuracy, AUC 0.993), algorithm overview table, and the algorithm selection framework.
2. **k-Nearest Neighbors** — Instance-based learning, k sweep from 1–50, critical importance of feature scaling (92.1% unscaled → 97.4% scaled), kNN regression on California Housing, Euclidean vs. Manhattan distance.
3. **Support Vector Machines** — Maximum margin hyperplane, support vectors, C regularization parameter, RBF kernel and gamma, GridSearchCV for C and gamma jointly (achieves 98.2% accuracy / AUC 0.998), SVR for regression.
4. **Random Forests** — Bagging, bootstrap samples, out-of-bag error, feature randomness, n_estimators sweep, feature importances, comparison to single decision trees (93.0% → 96.5%).
5. **Gradient Boosting** — Sequential residual correction, learning rate / n_estimators trade-off, shallow trees (depth 2–3), feature importances, GridSearchCV tuning (AUC 0.999), introduction to XGBoost.
6. **Neural Networks** — Neurons, layers, ReLU activation, backpropagation, Adam optimizer, MLPClassifier architecture sweep, L2 regularization (alpha), loss curve interpretation, scaling requirement (88.6% → 97.4%).
7. **Module Assessment** — Head-to-head comparison of all 7 models (2 baselines + 5 advanced), cross-validation comparison, reflection questions on model selection, coding exercise with ROC curve and threshold optimization.
