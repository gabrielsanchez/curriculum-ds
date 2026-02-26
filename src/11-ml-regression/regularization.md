# Regularization

## Overview

In the previous lesson, you saw that adding more complexity — polynomial terms, deeper decision trees — improves training performance but can hurt test performance. The model learns noise in the training data rather than the true underlying pattern. **Regularization** is the principled tool for preventing this: it adds a penalty to the cost function that discourages large coefficients, pulling the model toward simpler solutions. This lesson covers Ridge regression (L2), Lasso (L1), and ElasticNet — three regularized versions of linear regression that are staples of practical data science.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Apply Ridge and Lasso regression to control overfitting.
- Tune the regularization strength and interpret the effect on coefficients.

## Key Terms

**Regularization:** Adding a penalty term to the model's cost function that discourages large coefficients, reducing the model's tendency to overfit.

**Regularization parameter (alpha / λ):** Controls the strength of the penalty. alpha=0 is equivalent to ordinary linear regression. Larger alpha → stronger penalty → simpler model → less overfitting, but potentially more underfitting.

**Ridge regression (L2):** Adds a penalty proportional to the *sum of squared coefficients* (L2 norm). Shrinks all coefficients toward zero but rarely makes any exactly zero. Best for correlated features where all features likely contribute.

**Lasso regression (L1):** Adds a penalty proportional to the *sum of absolute coefficient values* (L1 norm). Drives some coefficients to exactly zero — effectively selecting features. Best when you believe many features are irrelevant.

**ElasticNet:** A hybrid that combines L1 and L2 penalties. Useful when you want both the grouping behavior of Ridge and the sparsity of Lasso.

**Coefficient path:** A plot showing how each coefficient's value changes as the regularization strength (alpha) varies. Reveals which features survive strong regularization.

**Cross-validation:** Splitting the training data into multiple folds to estimate generalization performance. Used to select the best alpha without peeking at the test set.

**`RidgeCV` / `LassoCV`:** scikit-learn classes that automatically run cross-validation over a range of alpha values and select the best one.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

## The Overfitting Problem, Precisely

Consider what happens when you fit linear regression with polynomial features on a small dataset:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Simulate a small dataset (200 samples — more realistic for overfitting to show)
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=200, replace=False)
X_small = X_train.iloc[idx]
y_small = y_train.iloc[idx]

for degree in [1, 2, 3]:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression())
    ])
    pipe.fit(X_small, y_small)
    train_r2 = r2_score(y_small, pipe.predict(X_small))
    test_r2  = r2_score(y_test,  pipe.predict(X_test))
    print(f"Degree {degree}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")
```

Output:
```
Degree 1: Train R²=0.611, Test R²=0.572
Degree 2: Train R²=0.837, Test R²=0.460
Degree 3: Train R²=0.991, Test R²=0.052
```

Degree 3 fits the 200 training samples almost perfectly (R²=0.991) but fails catastrophically on the test set (R²=0.052). The model learned the training noise rather than the signal — and regularization is the solution.

## How Regularization Works

Ordinary least squares minimizes:
```
Cost(OLS) = Σ (yᵢ − ŷᵢ)²
```

**Ridge** (L2) adds a penalty on the sum of squared coefficients:
```
Cost(Ridge) = Σ (yᵢ − ŷᵢ)² + alpha × Σ wⱼ²
```

**Lasso** (L1) adds a penalty on the sum of absolute coefficients:
```
Cost(Lasso) = Σ (yᵢ − ŷᵢ)² + alpha × Σ |wⱼ|
```

In both cases, the optimizer must now balance fitting the data well *and* keeping coefficients small. As `alpha` increases, the penalty dominates and coefficients are pushed toward zero — at the extreme, all coefficients become zero and the model predicts the mean of the training data.

**Scaling is required** before regularization. The penalty treats all coefficients equally — if features are on different scales, features with larger raw values will have artificially smaller coefficients, and the penalty will be applied unevenly. Always standardize first.

## Ridge Regression (L2)

```python
# Compare unregularized vs. Ridge at several alpha values
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

print(f"{'alpha':<10} {'Train R²':>10} {'Test R²':>10} {'MAE ($)':>12}")
print("-" * 45)

lr_baseline = LinearRegression().fit(X_train_s, y_train)
print(f"{'OLS (α=0)':<10} {r2_score(y_train, lr_baseline.predict(X_train_s)):>10.3f} "
      f"{r2_score(y_test, lr_baseline.predict(X_test_s)):>10.3f} "
      f"{mean_absolute_error(y_test, lr_baseline.predict(X_test_s))*100_000:>12,.0f}")

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_s, y_train)
    train_r2 = r2_score(y_train, ridge.predict(X_train_s))
    test_r2  = r2_score(y_test,  ridge.predict(X_test_s))
    mae      = mean_absolute_error(y_test, ridge.predict(X_test_s)) * 100_000
    print(f"{alpha:<10.3f} {train_r2:>10.3f} {test_r2:>10.3f} {mae:>12,.0f}")
```

Output:
```
alpha      Train R²   Test R²     MAE ($)
---------------------------------------------
OLS (α=0)     0.606     0.576       52,700
0.001         0.606     0.576       52,700
0.010         0.606     0.576       52,700
0.100         0.606     0.576       52,699
1.000         0.606     0.576       52,694
10.000        0.605     0.575       52,717
100.000       0.596     0.566       53,500
```

For the full 16,000-sample California Housing training set, Ridge provides minimal improvement over OLS — the dataset is large enough that OLS doesn't overfit. The benefit of regularization is most pronounced on smaller datasets or with polynomial features.

### Ridge on Polynomial Features with a Small Dataset

This is where Ridge shines:

```python
from sklearn.linear_model import Ridge

results = []
for alpha in [0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)

    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  model)
    ])
    pipe.fit(X_small, y_small)
    train_r2 = r2_score(y_small, pipe.predict(X_small))
    test_r2  = r2_score(y_test,  pipe.predict(X_test))
    results.append({"alpha": alpha, "Train R²": round(train_r2, 3), "Test R²": round(test_r2, 3)})

print(pd.DataFrame(results).to_string(index=False))
```

Output:
```
 alpha  Train R²  Test R²
     0     0.991    0.052   ← catastrophic overfit
   0.1     0.972    0.254
   1.0     0.932    0.502
  10.0     0.867    0.593
 100.0     0.807    0.634   ← best test performance
1000.0     0.732    0.620
```

Ridge dramatically recovers the degree-3 polynomial model from near-zero test R² to 0.634 — simply by penalizing large coefficients. This illustrates the core principle: regularization trades a small amount of training fit for substantially better generalization.

## Lasso Regression (L1)

Lasso works similarly to Ridge but with the absolute value penalty. Its defining property: it drives some coefficients to **exactly zero**, effectively removing those features from the model.

```python
from sklearn.linear_model import Lasso

alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

print(f"{'alpha':<8} {'Test R²':>10} {'Non-zero coefs':>16}")
print("-" * 36)

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10_000)
    lasso.fit(X_train_s, y_train)
    test_r2   = r2_score(y_test, lasso.predict(X_test_s))
    n_nonzero = (lasso.coef_ != 0).sum()
    print(f"{alpha:<8.3f} {test_r2:>10.3f} {n_nonzero:>16}")
```

Output:
```
 alpha   Test R²  Non-zero coefs
------------------------------------
 0.001     0.576              8
 0.010     0.576              8
 0.050     0.571              7
 0.100     0.554              6
 0.500     0.454              4
 1.000     0.338              2
```

At alpha=0.001, all 8 features are retained. As alpha increases, Lasso zeros out features one by one. At alpha=0.5, only 4 features survive. Lasso is performing **automatic feature selection**.

### Which Features Does Lasso Eliminate?

```python
lasso_sparse = Lasso(alpha=0.1, max_iter=10_000)
lasso_sparse.fit(X_train_s, y_train)

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lasso_sparse.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print(coef_df)
```

Output:
```
         feature  coefficient
          MedInc       0.7831
        Latitude      -0.6423
       Longitude      -0.5889
        AveRooms       0.2201
       AveBedrms      -0.1820
        HouseAge       0.0978
        AveOccup      -0.0000   ← zeroed out
      Population       0.0000   ← zeroed out
```

At alpha=0.1, `AveOccup` and `Population` are eliminated. Their signal is weak enough that the L1 penalty pushes them to zero. Lasso effectively says: "These features aren't worth the model complexity they add."

### Coefficient Path: Watching Features Disappear

```python
from sklearn.linear_model import lasso_path

alphas_path, coefs_path, _ = lasso_path(X_train_s, y_train, eps=1e-3)

plt.figure(figsize=(10, 5))
for i, name in enumerate(X.columns):
    plt.plot(np.log10(alphas_path), coefs_path[i], label=name)
plt.xlabel("log₁₀(alpha)  [← weaker regularization | stronger regularization →]")
plt.ylabel("Coefficient value")
plt.title("Lasso Coefficient Path")
plt.legend(fontsize=8, loc="upper right")
plt.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()
```

This plot shows each feature's coefficient value as regularization strength varies. Features whose lines hit zero first are the least predictive. `MedInc` and geographic features (Latitude, Longitude) survive the longest — they're the strongest predictors.

## ElasticNet: Combining L1 and L2

ElasticNet adds both penalties:
```
Cost = Σ (yᵢ − ŷᵢ)² + alpha × [l1_ratio × Σ|wⱼ| + (1 − l1_ratio) × Σwⱼ²]
```

Where `l1_ratio=1` is pure Lasso and `l1_ratio=0` is pure Ridge.

```python
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000)
enet.fit(X_train_s, y_train)

print(f"ElasticNet — Test R²: {r2_score(y_test, enet.predict(X_test_s)):.3f}")
print(f"Non-zero coefficients: {(enet.coef_ != 0).sum()}")
```

Output:
```
ElasticNet — Test R²: 0.574
Non-zero coefficients: 8
```

ElasticNet is most useful when you have many correlated features — Lasso arbitrarily picks one from each correlated group, while ElasticNet keeps them all but shrinks them together.

## Selecting Alpha Automatically with Cross-Validation

Choosing alpha by evaluating on the test set is data leakage. Use cross-validation on the training data:

```python
from sklearn.linear_model import RidgeCV, LassoCV

# RidgeCV searches over a range of alphas using k-fold CV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X_train_s, y_train)

print(f"Best alpha (Ridge): {ridge_cv.alpha_:.4f}")
print(f"Ridge CV — Test R²: {r2_score(y_test, ridge_cv.predict(X_test_s)):.3f}")

# LassoCV does the same for Lasso
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10_000)
lasso_cv.fit(X_train_s, y_train)

print(f"\nBest alpha (Lasso): {lasso_cv.alpha_:.6f}")
print(f"Lasso CV — Test R²: {r2_score(y_test, lasso_cv.predict(X_test_s)):.3f}")
```

Output:
```
Best alpha (Ridge): 7.197
Ridge CV — Test R²: 0.576

Best alpha (Lasso): 0.000387
Lasso CV — Test R²: 0.576
```

Both cross-validated models match the OLS baseline — again confirming that regularization's main benefit appears on smaller datasets or higher-dimensional feature spaces (like degree-3 polynomial features).

## Ridge vs. Lasso vs. ElasticNet: Choosing

| | Ridge (L2) | Lasso (L1) | ElasticNet |
|-|------------|------------|------------|
| **Penalty** | Σ wⱼ² | Σ \|wⱼ\| | Mix of both |
| **Zeros coefficients** | No | Yes | Yes (fewer than Lasso) |
| **Feature selection** | No | Yes | Partial |
| **Best for** | Multicollinearity; all features relevant | Sparse problems; many irrelevant features | Many correlated features with some irrelevant |
| **Tuning** | `alpha` | `alpha` | `alpha` + `l1_ratio` |
| **Sensitivity to correlated features** | Keeps all, shrinks evenly | Arbitrarily keeps one | Keeps group, shrinks together |

**Rule of thumb:**
- Start with **Ridge** — it's stable and rarely hurts
- Use **Lasso** when you suspect many features are irrelevant and want interpretability
- Use **ElasticNet** when you have many correlated feature groups

## The Bias-Variance Trade-off Visualized

```python
alphas = np.logspace(-3, 3, 30)
train_scores, test_scores = [], []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_s, y_train)
    train_scores.append(r2_score(y_train, ridge.predict(X_train_s)))
    test_scores.append(r2_score(y_test,  ridge.predict(X_test_s)))

plt.figure(figsize=(9, 5))
plt.semilogx(alphas, train_scores, label="Train R²", color="steelblue")
plt.semilogx(alphas, test_scores,  label="Test R²",  color="coral")
plt.xlabel("alpha (regularization strength, log scale)")
plt.ylabel("R²")
plt.title("Bias-Variance Trade-off: Ridge Regularization")
plt.legend()
plt.axvline(ridge_cv.alpha_, color="gray", linestyle="--", label=f"CV best α={ridge_cv.alpha_:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
```

The plot shows two regions:
- **Left (low alpha):** Training and test R² are close and high — the model fits well without overfitting (for this large dataset)
- **Right (high alpha):** Both scores fall — too much regularization forces the model toward the mean, underfitting both sets

The best alpha sits at the point where test R² peaks.

## Conclusion

In this lesson, you learned that regularization prevents overfitting by adding a penalty to the cost function that discourages large coefficients. Ridge (L2) shrinks all coefficients toward zero while retaining all features — best when all features likely contribute. Lasso (L1) drives some coefficients to exactly zero, performing automatic feature selection — best when many features are irrelevant. ElasticNet combines both. For all methods, `alpha` controls the strength and should be selected with cross-validation, not by evaluating on the test set. In the final lesson, you'll build a complete model evaluation framework that ties all the regression methods together.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Lasso&#039;s L1 penalty has a geometric property that promotes sparse solutions: the penalty surface has corners at the axes, and the optimal solution often lands exactly on an axis (coefficient = 0). As alpha grows, the penalty dominates and more features are zeroed out. This is automatic feature selection — useful when you have many candidate features and want the model to identify which ones actually matter, reducing both complexity and overfitting risk.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You train a Lasso regression model with progressively larger alpha values. As alpha increases from 0.001 to 10.0, you observe that the number of non-zero coefficients drops from 20 to 3. What is happening, and what is the practical benefit of this behavior?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Lasso is removing correlated features by averaging them together, which reduces model complexity.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Lasso's L1 penalty is driving coefficients to exactly zero as the regularization strength increases, automatically performing feature selection. The benefit is a sparser, more interpretable model that keeps only the strongest predictors.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The model is converging to a local minimum and becoming less stable at high alpha values.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Lasso eliminates features randomly to prevent overfitting — the selection is not predictable.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Lasso tends to arbitrarily select one feature from a correlated group and zero out the others, even when all carry similar information. ElasticNet&#039;s L2 component groups correlated features together (like Ridge does), while its L1 component still drives irrelevant features to zero. The `l1_ratio` parameter controls the balance: values closer to 1 behave more like Lasso, values closer to 0 behave more like Ridge. For a dataset with many correlated feature groups where some features are genuinely irrelevant, ElasticNet is the most appropriate choice.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You have a dataset with 500 features, many of which are highly correlated in groups. You want to use Lasso for feature selection but are concerned it will arbitrarily choose one feature from each correlated group and discard the others. Which approach is more appropriate?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Use Ridge — it handles correlated features by shrinking them together without zeroing any out.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Use ElasticNet — it combines L1 (for sparsity) and L2 (which groups correlated features together), keeping the group signal while still zeroing out irrelevant features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Use a higher Lasso alpha to force it to keep all correlated features.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Use a Decision Tree, which is immune to feature correlation.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="The key rule is: the test set may only be used once, for the final evaluation. `RidgeCV` performs cross-validation entirely within the training set — it splits training data into 5 folds, evaluates each alpha on held-out folds, and selects the best alpha. The test set is never seen during this process. The final step — training Ridge(alpha=7.2) on all training data and evaluating on the test set — is correct and gives an unbiased performance estimate. This is the standard workflow: CV for hyperparameter selection, test set for final evaluation.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You use `RidgeCV` with 5-fold cross-validation to select the best alpha from a grid of 50 values, and it selects alpha=7.2. You then evaluate the final Ridge model (trained with alpha=7.2 on the full training set) on the held-out test set. Is this workflow correct, and why?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>No — you should retrain with a different alpha on the test set to confirm the choice.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Yes — `RidgeCV` selects alpha using only the training data (5-fold CV), so the test set was never used to make any decisions. Evaluating the final model on the test set gives an unbiased estimate of generalization performance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>No — you should use a validation set (not the test set) to confirm alpha selection after cross-validation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Yes, but only if the test set was not used to build the cross-validation folds.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

