# Neural Networks

## Overview

In the previous lesson, gradient boosting achieved AUC 0.999 by sequentially correcting residual errors. Neural networks take a different architectural approach: they learn hierarchical representations by passing data through multiple layers of interconnected neurons, each layer transforming the input into increasingly abstract features. This module introduces neural networks through scikit-learn's `MLPClassifier` — a shallow, fully-connected network — to build intuition before module 14 covers deep learning with dedicated libraries.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Recognize and describe the basic architecture of a neural network (layers, neurons, activation functions, backpropagation).
- Train and tune a multi-layer perceptron (MLP) using scikit-learn, and compare it to the other algorithms in this module.

## Key Terms

**Neuron (node):** The basic computational unit. Takes a weighted sum of its inputs, adds a bias, and passes the result through an activation function.

**Layer:** A collection of neurons that process the same inputs in parallel. Networks have an input layer, one or more hidden layers, and an output layer.

**Activation function:** A non-linear function applied to each neuron's output. Without activation functions, a stack of linear layers is just a linear model. Common activations: ReLU, sigmoid, tanh.

**ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. The default activation for hidden layers. Simple, fast, and avoids the vanishing gradient problem that plagued earlier sigmoid-based networks.

**Backpropagation:** The algorithm for computing gradients of the loss with respect to all weights in the network by applying the chain rule layer by layer, from output back to input.

**Stochastic Gradient Descent (SGD):** Updates weights using the gradient computed on a small batch of training examples (mini-batch), rather than the full dataset. Makes training feasible on large datasets.

**Adam optimizer:** An adaptive SGD variant that adjusts the learning rate for each weight individually. The default optimizer in most deep learning frameworks and scikit-learn's `MLPClassifier`.

**Epoch:** One complete pass through the entire training dataset during optimization.

**Overfitting in neural networks:** Networks are prone to memorizing training data. Common remedies: dropout (randomly zeroing neurons during training), L2 regularization (weight decay), early stopping.

**Multi-layer perceptron (MLP):** A fully-connected neural network where each neuron in one layer connects to every neuron in the next layer. The simplest form of neural network.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## The Neural Network Architecture

```
Input layer     Hidden layer 1   Hidden layer 2   Output layer
(30 features)   (100 neurons)    (50 neurons)     (2 classes)

  x₁ ──┐        ○ ○ ○ ○ ○        ○ ○ ○ ○ ○        ○ benign
  x₂ ──┤─────→  ○ ○ ○ ○ ○  ───→  ○ ○ ○ ○ ○  ───→
  x₃ ──┤        ○ ○ ○ ○ ○        ○ ○ ○ ○ ○        ○ malignant
  ...  │         (more)           (more)
  x₃₀ ─┘

Each ─→ represents a learned weight.
Each ○ applies: output = ReLU(Σ wᵢxᵢ + b)
```

A network with 30 inputs, one hidden layer of 100 neurons, and 2 outputs has:
- 30 × 100 = 3,000 weights in the first layer
- 100 × 2 = 200 weights in the second layer
- Plus biases: 100 + 2 = 102
- **Total: 3,302 learnable parameters** for a simple 2-layer network

Gradient boosting on the same dataset has ~200 trees × ~15 nodes each ≈ 3,000 parameters. The models are more similar in size than they first appear — the difference is in how those parameters are structured and learned.

## Training a Basic MLP

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CRITICAL: neural networks require feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# A simple MLP with one hidden layer of 100 neurons
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_s, y_train)

y_pred  = mlp.predict(X_test_s)
y_proba = mlp.predict_proba(X_test_s)[:, 1]

print(f"MLP (1 hidden layer, 100 neurons):")
print(f"  Train accuracy: {mlp.score(X_train_s, y_train):.3f}")
print(f"  Test accuracy:  {mlp.score(X_test_s, y_test):.3f}")
print(f"  Test AUC:       {roc_auc_score(y_test, y_proba):.3f}")
print(f"  Iterations:     {mlp.n_iter_}")
```

Output:
```
MLP (1 hidden layer, 100 neurons):
  Train accuracy: 1.000
  Test accuracy:  0.974
  Test AUC:       0.997
  Iterations:     113
```

With default settings, a single hidden layer of 100 neurons achieves 97.4% accuracy and AUC 0.997 — competitive with kNN and gradient boosting. The model converged in 113 passes through the training data.

## Hidden Layer Architecture

```python
architectures = [
    (50,),           "Small (50)",
    (100,),          "Medium (100)",
    (200,),          "Large (200)",
    (100, 50),       "Two layers (100→50)",
    (100, 100, 50),  "Three layers (100→100→50)",
]

# Pair up (sizes, label) correctly
configs = [
    ((50,),           "Small (50)"),
    ((100,),          "Medium (100)"),
    ((200,),          "Large (200)"),
    ((100, 50),       "Two layers (100→50)"),
    ((100, 100, 50),  "Three layers (100→100→50)"),
]

print(f"{'Architecture':<28} {'Train':>8} {'Test':>8} {'AUC':>8}")
print("-" * 56)

for hidden_layers, label in configs:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_s, y_train)
    y_proba = mlp.predict_proba(X_test_s)[:, 1]
    print(f"{label:<28} {mlp.score(X_train_s, y_train):>8.3f} "
          f"{mlp.score(X_test_s, y_test):>8.3f} "
          f"{roc_auc_score(y_test, y_proba):>8.3f}")
```

Output:
```
Architecture              Train     Test      AUC
--------------------------------------------------------
Small (50)                1.000    0.965    0.993
Medium (100)              1.000    0.974    0.997
Large (200)               1.000    0.974    0.997
Two layers (100→50)       1.000    0.982    0.997
Three layers (100→100→50) 1.000    0.974    0.997
```

On this small dataset (455 training samples), deeper networks don't consistently outperform a single hidden layer. The two-layer network (100→50) achieves the best test accuracy (98.2%), matching SVM. On very small datasets, deeper networks have more parameters to overfit — the marginal benefit of additional layers diminishes.

**Why "deeper is better" applies mainly to large datasets and unstructured data:**
- Images have local structure that convolutional layers exploit.
- Text has sequential structure that recurrent layers exploit.
- Tabular data with 30 features typically doesn't benefit from more than 1–2 hidden layers.

## Regularization with alpha

Neural networks can overfit, especially on small datasets. The `alpha` parameter adds L2 regularization (weight decay):

```python
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

print(f"{'alpha':>10} {'Train':>10} {'Test':>10}")
print("-" * 32)
for alpha in alphas:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=alpha,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_s, y_train)
    print(f"{alpha:>10.4f} {mlp.score(X_train_s, y_train):>10.3f} "
          f"{mlp.score(X_test_s, y_test):>10.3f}")
```

Output:
```
     alpha      Train       Test
--------------------------------
    0.0001      1.000      0.974
    0.0010      1.000      0.974
    0.0100      1.000      0.982
    0.1000      1.000      0.974
    1.0000      0.991      0.965
```

Moderate regularization (alpha=0.01) achieves the best test accuracy (98.2%). Very strong regularization (alpha=1.0) begins to underfit — the model is constrained too heavily and can't capture all the pattern in the data.

## The Loss Curve

Neural networks minimize loss iteratively. Plotting the loss curve reveals whether training converged:

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_s, y_train)

plt.figure(figsize=(9, 4))
plt.plot(mlp.loss_curve_, color="steelblue")
plt.xlabel("Iteration (epoch)")
plt.ylabel("Training loss")
plt.title("MLP: Training Loss Curve")
plt.tight_layout()
plt.show()

print(f"Final training loss: {mlp.loss_curve_[-1]:.4f}")
print(f"Converged in {mlp.n_iter_} iterations")
```

The loss should:
- Decrease consistently in the early epochs (learning is happening)
- Plateau near the end (training has converged)

If the loss curve is still declining at `max_iter`, increase `max_iter` to let training finish. If the curve oscillates wildly, the learning rate may be too high (use a smaller value or switch to `solver="lbfgs"` for small datasets).

## Full Evaluation

```python
best_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.01,
    max_iter=500,
    random_state=42
)
best_mlp.fit(X_train_s, y_train)
y_pred  = best_mlp.predict(X_test_s)
y_proba = best_mlp.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

Output:
```
              precision    recall  f1-score   support

   malignant       0.976      0.953      0.964        43
      benign       0.972      0.986      0.979        71

    accuracy                           0.974       114

   macro avg       0.974      0.969      0.972       114
weighted avg       0.974      0.974      0.974       114

AUC: 0.998
```

The tuned MLP achieves 97.4% accuracy and AUC 0.998, matching kNN and gradient boosting on this dataset.

## Beyond MLPClassifier: Why Deep Learning Needs Different Tools

`MLPClassifier` is a fully-connected "shallow" network — suitable for structured tabular data with a modest number of features. Real-world deep learning uses specialized architectures:

| Data type | Architecture | Library |
|-----------|-------------|---------|
| Images | Convolutional Neural Network (CNN) | PyTorch, TensorFlow/Keras |
| Text | Transformer (BERT, GPT) | Hugging Face Transformers |
| Sequences | Recurrent Neural Network (LSTM/GRU) | PyTorch, TensorFlow |
| Tabular data | MLP, or gradient boosting | sklearn, XGBoost, PyTorch |
| Audio | CNN or Transformer | PyTorch, TensorFlow |

The fundamental concepts — neurons, layers, activation functions, backpropagation, SGD — are the same across all of these. Module 14 introduces deep learning frameworks (PyTorch or TensorFlow) where these architectures are built.

## Why Scaling Is Critical for Neural Networks

Like SVM and kNN, neural networks compute weighted sums of inputs. Features on different scales create very different gradient magnitudes during training:

```python
# Unscaled MLP
mlp_unscaled = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_unscaled.fit(X_train, y_train)

# Scaled MLP
mlp_scaled = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_scaled.fit(X_train_s, y_train)

print(f"MLP unscaled: {mlp_unscaled.score(X_test, y_test):.3f}")
print(f"MLP scaled:   {mlp_scaled.score(X_test_s, y_test):.3f}")
```

Output:
```
MLP unscaled: 0.886
MLP scaled:   0.974
```

An 8.8% accuracy gap from one missing preprocessing step. Without scaling, the optimizer spends iterations navigating a poorly conditioned loss landscape — weights connected to large-valued features update too aggressively while weights connected to small-valued features barely move.

**Rule: always scale features before training neural networks.**

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Can learn arbitrary complex functions | Requires large amounts of data |
| State-of-the-art on images, text, audio | Slow to train (especially deep networks) |
| Flexible architecture (layers, skip connections) | Many hyperparameters to tune |
| Pre-trained models available for transfer learning | Hard to interpret (no feature importances) |
| Handles raw unstructured data (pixels, tokens) | Sensitive to initialization and learning rate |

## When to Use Neural Networks

**Use Neural Networks when:**
- Data is unstructured: images, audio, text — where convolutional or recurrent architectures excel
- Dataset is large (> 10,000 samples for MLP; > 100,000 for deep learning)
- You have access to significant compute resources (GPU for deep learning)
- Pre-trained models (BERT, ResNet) can be fine-tuned for your task

**Avoid Neural Networks (for tabular data) when:**
- Dataset is small (< 5,000 samples) — gradient boosting typically wins
- Training budget is limited — gradient boosting trains faster
- Interpretability is needed — neural networks provide no natural explanations
- You're building a baseline — start with logistic regression or gradient boosting

## Conclusion

Neural networks offer unmatched flexibility — they can learn arbitrary mappings from inputs to outputs, given enough data. On the breast cancer dataset (455 training samples), the MLP achieves 97.4% accuracy and AUC 0.998, competitive with kNN, SVM, and gradient boosting. The key lessons carry through from previous algorithms: **scaling is non-negotiable** (8.8% accuracy gap without it), and the dataset is too small to demonstrate the true advantage of deep networks. In the next lesson, the module assessment brings together all five algorithms — kNN, SVM, Random Forests, Gradient Boosting, and Neural Networks — for a head-to-head comparison, helping you synthesize which algorithm to reach for in different problem contexts.

## Practice

### Knowledge Check

#### **Question 1: A neural network with one hidden layer of 100 ReLU neurons is trained on the breast cancer dataset. The training loss decreases rapidly for the first 50 epochs, then plateaus. The training accuracy is 100% but test accuracy is 88%. What is happening and what would you try first?**

1. The network has not converged — increase max_iter so training continues past the plateau.
2. The network is overfitting — it has memorized training examples. The 100% training accuracy and significant train-test gap indicate the model has learned noise. Try: increasing the alpha regularization parameter, reducing hidden_layer_sizes, or using early stopping.
3. The network is underfitting — the loss plateau means it has not learned enough. Add more hidden layers to increase capacity.
4. The loss plateau is a sign that the learning rate is too high and training is oscillating. Switch to a smaller learning rate.

**Correct Answer:**
2. The network is overfitting — it has memorized training examples. The 100% training accuracy with 88% test accuracy indicates the model has learned noise. Try increasing alpha (L2 regularization), reducing hidden_layer_sizes, or using early stopping.

**Explanation:**
A plateau in training loss combined with 100% training accuracy means the model has converged on a solution that fits the training data perfectly. The 12% gap to test accuracy indicates this solution is not general — the network memorized specific training examples rather than underlying patterns. On a 455-sample dataset, a 100-neuron network (with ~3,300 parameters) is actually overparameterized. The standard remedies: L2 weight decay (alpha), smaller network, or early stopping (halting training when validation performance stops improving).

---

#### **Question 2: Why do neural networks outperform traditional ML methods on image classification, but not necessarily on tabular data?**

1. Neural networks outperform on images because they are faster to train on pixel data than decision trees.
2. Neural networks use convolutional layers that exploit the spatial structure of images — nearby pixels are highly correlated, and edges/textures are meaningful local patterns. This inductive bias is perfectly matched to image data. Tabular data has no spatial structure — each feature is independent and unordered — so the spatial assumptions of CNNs don't apply. Gradient boosting and even logistic regression often match or outperform neural networks on tabular data because they don't need to re-learn that feature ordering is irrelevant.
3. Neural networks outperform on images because they were specifically designed for classification, while decision trees were designed for regression.
4. Neural networks outperform on images because they use more training data — image datasets are always larger than tabular datasets.

**Correct Answer:**
2. Neural networks use convolutional layers that exploit the spatial structure of images — nearby pixels are highly correlated and edges/textures are meaningful local patterns. Tabular data has no such structure, so gradient boosting often matches or beats neural networks on tabular tasks.

**Explanation:**
The architecture of a model encodes assumptions about data structure. CNNs assume local correlations and translation invariance (a cat looks like a cat regardless of where it is in the image). Transformers assume sequential or relational structure between tokens. These assumptions are right for images and text. Tabular data — where column order is arbitrary and each feature is a different measurement type — violates these assumptions. Random forests and gradient boosting, which treat each feature independently, are often better matched to tabular data structure.

---

#### **Question 3: You are asked to build a model that classifies X-ray images as normal or pneumonia. You have 5,000 labeled images. Which approach is most appropriate?**

1. Train an MLPClassifier on flattened pixel values — neural networks handle images natively.
2. Use gradient boosting on handcrafted features extracted from the images (mean brightness, edge density).
3. Use a pre-trained convolutional neural network (e.g., ResNet or EfficientNet from ImageNet) and fine-tune the final layers on your 5,000 images. This transfer learning approach gets the benefit of deep learning without needing millions of labeled medical images.
4. Use logistic regression on the raw pixels — it is the simplest and most interpretable approach.

**Correct Answer:**
3. Use a pre-trained CNN (e.g., ResNet) and fine-tune the final layers on your 5,000 images. Transfer learning enables deep learning with limited labeled data.

**Explanation:**
5,000 images is too few to train a deep CNN from scratch (typical ImageNet models train on 1.2 million images). A pre-trained CNN has already learned general visual features (edges, textures, shapes) from millions of images. Fine-tuning the last few layers adapts these general features to the specific task of detecting pneumonia. This approach typically outperforms both MLPClassifier (which ignores spatial structure) and handcrafted features (which require domain expertise to design). Transfer learning is the standard practice in medical imaging with limited labeled data.
