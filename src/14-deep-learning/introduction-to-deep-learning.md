# Introduction to Deep Learning

## Overview

In module 13, you trained a `MLPClassifier` — a multi-layer perceptron — achieving 97.4% accuracy on the breast cancer dataset. That was scikit-learn's shallow neural network, which works well for small tabular data. Deep learning extends the same idea — neurons, layers, activation functions, backpropagation — to architectures with dozens or hundreds of layers, hundreds of millions of parameters, and the ability to learn directly from raw images, audio, and text. This lesson bridges module 13's conceptual introduction to the practical deep learning tools you'll use throughout this module.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Explain how deep learning architectures differ from shallow neural networks and traditional ML.
- Identify the role of activation functions, loss functions, and backpropagation in training deep networks.

## Key Terms

**Deep learning:** A subfield of machine learning using neural networks with multiple hidden layers (typically 3+). The "depth" enables learning hierarchical representations from raw data.

**Representation learning:** The ability of a network to automatically discover the features needed for a task, rather than relying on hand-engineered features. A CNN learning to detect edges → textures → shapes is an example.

**Forward pass:** The computation of a network's output for a given input: data flows through each layer's weights and activation functions from input to output.

**Loss function:** A function that measures how wrong the network's predictions are. Training minimizes this function. Common choices: cross-entropy (classification), mean squared error (regression).

**Gradient descent:** The optimization algorithm that adjusts weights to minimize the loss. At each step, weights are updated in the direction that reduces the loss: `w ← w − η × ∇L`, where η is the learning rate.

**Backpropagation:** The algorithm for computing gradients of the loss with respect to every weight in the network, by applying the chain rule layer by layer from output back to input. Makes gradient descent tractable for deep networks.

**Mini-batch SGD:** Gradient descent computed on a small random subset of training examples (a "batch") rather than the full dataset. Balances computational efficiency with gradient accuracy.

**Epoch:** One complete pass through the entire training dataset. Training typically runs for 10–100+ epochs.

**Overfitting:** When a model memorizes training examples rather than learning generalizable patterns. Deep networks are especially prone to overfitting due to their large number of parameters.

**TensorFlow / Keras:** TensorFlow is Google's open-source deep learning framework. Keras is its high-level API — the interface you'll write code against. `import tensorflow as tf; from tensorflow import keras` is the standard entry point.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Why Deep Learning?

In module 13, you saw that gradient boosting and SVM achieve ~98% accuracy on the breast cancer dataset — matching simple neural networks with far fewer parameters. So why learn deep learning at all?

The answer is **data type**. Traditional ML (logistic regression, gradient boosting) operates on structured tabular features that humans design. Deep learning operates on **raw unstructured data** — pixels, waveforms, text tokens — and learns its own features automatically.

Consider handwritten digit recognition. A traditional ML approach requires someone to manually design features: stroke length, loop count, angle distributions. A deep network looks at raw pixels and discovers these features itself — edges in early layers, curves and strokes in middle layers, digit shapes in deep layers.

```
Raw pixel data (784 values)
        ↓
Layer 1: learns edge detectors
        ↓
Layer 2: learns stroke patterns
        ↓
Layer 3: learns digit shapes
        ↓
Output: digit class (0–9)
```

This **hierarchical representation learning** is what separates deep learning from shallow models. The depth isn't arbitrary — it mirrors the hierarchical nature of the data itself.

## From MLPClassifier to Keras

In module 13, you used:

```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(X_train, y_train)
```

Keras expresses the same idea but with explicit layer construction:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

The difference is not just syntax:
- **Keras is explicit about layer types** — Dense (fully connected), Conv2D (convolutional), LSTM (recurrent), etc. MLPClassifier only offers dense layers.
- **Keras handles the full training loop** — you specify the optimizer, loss function, and metrics at compile time.
- **Keras supports GPU training** — the same code runs on CPU or GPU without modification.
- **Keras supports arbitrary architectures** — skip connections (ResNet), branching (Inception), attention (Transformer).

## Activation Functions

Module 13 introduced ReLU. Here is the full picture of common activations and when to use them:

```
ReLU:    f(x) = max(0, x)
         ▪ Default for hidden layers
         ▪ Fast, avoids vanishing gradient
         ▪ "Dead neuron" problem if learning rate is too high

Sigmoid: f(x) = 1 / (1 + e⁻ˣ)   output ∈ (0, 1)
         ▪ Output layer for binary classification
         ▪ Saturates at extremes — slows learning in deep networks
         ▪ Rarely used in hidden layers anymore

Tanh:    f(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)   output ∈ (−1, 1)
         ▪ Zero-centered version of sigmoid
         ▪ Used in LSTM/GRU gates

Softmax: f(xᵢ) = eˣⁱ / Σeˣʲ   outputs sum to 1
         ▪ Output layer for multi-class classification
         ▪ Converts raw scores to probability distribution
```

**Rule of thumb:**
- Hidden layers → ReLU
- Binary classification output → Sigmoid + binary cross-entropy loss
- Multi-class output (≥3 classes) → Softmax + sparse categorical cross-entropy loss
- Regression output → No activation (linear)

## Loss Functions

The loss function defines what the network optimizes. The right loss depends on the task:

| Task | Output activation | Loss function | Keras name |
|------|------------------|---------------|------------|
| Binary classification | Sigmoid | Binary cross-entropy | `"binary_crossentropy"` |
| Multi-class (one-hot labels) | Softmax | Categorical cross-entropy | `"categorical_crossentropy"` |
| Multi-class (integer labels) | Softmax | Sparse categorical cross-entropy | `"sparse_categorical_crossentropy"` |
| Regression | None (linear) | Mean squared error | `"mse"` |

Cross-entropy measures how far the predicted probability distribution is from the true distribution. For binary classification: `L = −[y·log(p) + (1−y)·log(1−p)]`. The network learns to make its predicted probabilities close to 1 for the correct class and 0 for all others.

## Backpropagation: The Chain Rule in Practice

Training a neural network requires computing `∂L/∂w` — how much the loss changes when each weight changes. With millions of weights, this seems intractable. Backpropagation makes it efficient using the chain rule:

```
Loss L depends on output layer → output depends on hidden layer → hidden depends on weights

∂L/∂w₁ = (∂L/∂output) × (∂output/∂hidden) × (∂hidden/∂w₁)
```

Each layer computes a local gradient during the forward pass and stores it. During the backward pass, gradients flow from the output layer back to the input, multiplying local gradients along the way. The result: every weight's gradient is computed in one backward pass — the same computational cost as one forward pass.

**You will never implement backpropagation manually.** Keras (and all modern deep learning frameworks) compute gradients automatically using `tf.GradientTape` or the automatic differentiation built into `model.fit()`. The key insight is understanding *why* the computation works, not implementing it.

## MNIST: The "Hello World" of Deep Learning

The MNIST dataset contains 70,000 handwritten digit images (60,000 train, 10,000 test), each 28×28 pixels, with labels 0–9. It is the standard benchmark for introducing deep learning.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST — built into Keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")
print(f"Classes: {np.unique(y_train)}")

# Visualize a few examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("MNIST: Sample Images")
plt.tight_layout()
plt.show()
```

Output:
```
Train: (60000, 28, 28),  Test: (10000, 28, 28)
Pixel range: [0, 255]
Classes: [0 1 2 3 4 5 6 7 8 9]
```

Each image is a 28×28 array of integers 0–255 (grayscale). The preprocessing pipeline is:
1. Normalize pixel values to [0, 1] by dividing by 255
2. Flatten 28×28 images into 784-dimensional vectors (for Dense networks) or keep 2D shape (for CNNs)
3. Labels are already integers 0–9 — use `sparse_categorical_crossentropy`

```python
# Normalize
X_train_n = X_train / 255.0
X_test_n  = X_test  / 255.0

# Flatten for a dense network
X_train_flat = X_train_n.reshape(-1, 784)
X_test_flat  = X_test_n.reshape(-1, 784)

print(f"Normalized range: [{X_train_n.min()}, {X_train_n.max()}]")
print(f"Flattened shape: {X_train_flat.shape}")
```

Output:
```
Normalized range: [0.0, 1.0]
Flattened shape: (60000, 784)
```

## Comparing MLPClassifier to Keras on MNIST

Before diving into Keras's full API in the next lesson, a direct comparison to situate the tooling:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

# MLPClassifier on MNIST
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=30, random_state=42)
mlp.fit(X_train_flat, y_train)
sklearn_time = time.time() - t0
sklearn_acc  = accuracy_score(y_test, mlp.predict(X_test_flat))
print(f"sklearn MLP:  accuracy={sklearn_acc:.3f},  time={sklearn_time:.1f}s")
```

Output:
```
sklearn MLP:  accuracy=0.978,  time=47.3s
```

The Keras equivalent trains faster (GPU-accelerated) and achieves similar or better accuracy — and crucially, Keras supports architectures that scikit-learn cannot: convolutional layers for images, recurrent layers for sequences, attention layers for language.

## What's Next

In the next lesson, you'll learn the Keras API in depth: how to define layers, compile models with the right loss and optimizer, train with `model.fit()`, and interpret training history. You'll build a Dense network for MNIST classification and understand every line of code.

The lessons after that introduce specialized architectures:
- **CNNs** (lesson 3): exploit the spatial structure of images
- **RNNs** (lesson 4): handle sequences and variable-length text
- **Optimization & Regularization** (lesson 5): techniques to train deeper, more accurate models without overfitting

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="2" data-explanation="Logistic regression and gradient boosting need a feature matrix — a table of numbers where someone has already decided that &quot;age,&quot; &quot;income,&quot; and &quot;zip code&quot; are the relevant inputs. A CNN can take raw pixel values and learn that certain pixel patterns correspond to &quot;edges,&quot; then that certain edge patterns correspond to &quot;ears,&quot; then that certain ear-plus-face patterns correspond to &quot;cat&quot; — entirely automatically. This is what &quot;representation learning&quot; means: the network discovers what features are useful rather than being told.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the key difference between deep learning and traditional machine learning methods like logistic regression or gradient boosting?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Deep learning always achieves higher accuracy than traditional ML methods on any dataset.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Deep learning requires more data and compute, so it is only worth using for problems with more than 1 million samples.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>The key difference is representation learning: traditional ML requires humans to engineer features from raw data (e.g., extracting edge counts from pixels), while deep learning learns its own hierarchical representations directly from raw data (pixels, tokens, waveforms). This makes deep learning especially powerful for unstructured data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>Deep learning uses gradient descent while traditional ML does not — deep learning is the only algorithm family that can be optimized iteratively.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="In a 10-layer network using sigmoid activations, a gradient that starts at 0.5 at the output layer might be (0.5)^10 ≈ 0.001 by the time it reaches the first layer — the first layer learns 500× slower than the output layer. ReLU&#039;s gradient for positive inputs is exactly 1, so gradients don&#039;t shrink as they pass through (though dead neurons — neurons stuck at 0 for all inputs — are a different risk). This is why deep learning became practically trainable once ReLU replaced sigmoid in hidden layers around 2010.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why is the ReLU activation function preferred over sigmoid in the hidden layers of deep networks?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>ReLU is preferred because it produces outputs in the range (0, 1), making it compatible with probability interpretation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Sigmoid saturates at both extremes (very large or very small inputs produce gradients near zero), causing the vanishing gradient problem in deep networks — gradients shrink exponentially as they propagate back through many layers, making early layers learn very slowly. ReLU does not saturate for positive inputs (`f(x) = x` for x > 0), so gradients flow through without shrinking, enabling effective training of deep networks.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>ReLU is preferred because it is a probabilistic function, while sigmoid is deterministic.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>ReLU is preferred because it automatically normalizes activations to have zero mean and unit variance.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="This &quot;diverging loss&quot; pattern is the signature of overfitting: the model gets better at training examples but worse at new examples. The validation loss minimum (around epoch 20 in this example) represents the best generalizing model — early stopping saves those weights automatically. Lesson 5 covers all the regularization techniques for avoiding this pattern: dropout (randomly deactivating neurons during training), batch normalization, L2 weight decay, and learning rate scheduling.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A network is trained for 50 epochs. The training loss decreases from 0.8 to 0.02, but the validation loss decreases from 0.8 to 0.15 then rises to 0.45 by epoch 50. What is happening and what would you do?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The network is underfitting — both losses should decrease together. Add more layers to increase model capacity.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>The network is overfitting — after epoch ~20 (where validation loss was lowest at 0.15), the model began memorizing training examples. Approaches: add dropout, reduce network size, use early stopping (save the model weights from the epoch with lowest validation loss), or add L2 regularization.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>The rising validation loss indicates a data leak — some test data was accidentally included in the training set.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>This pattern is normal and expected — training loss is always much lower than validation loss in deep learning.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

