# Module Assessment

## Overview

This assessment brings together the core skills from module 14: building neural networks with the Keras Sequential API, designing convolutional architectures for image classification, applying regularization to reduce overfitting, and interpreting training curves and confusion matrices. You will work with the **Fashion-MNIST dataset** — 70,000 grayscale images of clothing across 10 categories — progressing from a dense baseline to a regularized CNN and analyzing what each architectural choice contributes to final performance.

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Build and train deep learning models using the Keras Sequential API.
- Design a CNN architecture for image classification.
- Apply regularization techniques (Dropout, BatchNormalization, EarlyStopping) to reduce overfitting.
- Interpret training curves and compare model performance across architectures.

## Starter Code

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/14-deep-learning/14-deep-learning-module-assessment_starter.ipynb). [Submit the link to the AI Grader for grading](https://ai-grader-production-07a3.up.railway.app/).

---

## The Dataset

Fashion-MNIST contains 70,000 grayscale 28x28 pixel images in 10 clothing categories: 60,000 for training and 10,000 for testing.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Train: (60000, 28, 28), Test: (10000, 28, 28)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
```

**Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Each image is a 28x28 single-channel (grayscale) array. The label is an integer from 0 to 9.

---

## Part 1: Dense Baseline

### Task 1: Build and Train a Dense Network

Flatten the 28x28 images to a 784-element vector, then build a fully connected network with two hidden layers. Train for 20 epochs with a 20% validation split. Report test accuracy.

```python
# CNN-ready reshape: add channel dimension for later tasks
X_train_cnn = X_train[..., np.newaxis]  # shape (60000, 28, 28, 1)
X_test_cnn  = X_test[...,  np.newaxis]  # shape (10000, 28, 28, 1)

# Dense baseline: flatten first
dense_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="dense_baseline")

dense_model.summary()

dense_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

dense_history = dense_model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=1,
)

dense_test_loss, dense_test_acc = dense_model.evaluate(X_test, y_test, verbose=0)
print(f"\nDense baseline test accuracy: {dense_test_acc:.4f}")
```

Record the test accuracy. You will use it in the comparison table in Task 6.

---

### Task 2: Training Curves

Plot training and validation accuracy and loss over all 20 epochs. Write 2 sentences interpreting whether the model shows signs of overfitting.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(dense_history.history["accuracy"],     label="Train accuracy")
axes[0].plot(dense_history.history["val_accuracy"], label="Val accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Dense Baseline — Accuracy")
axes[0].legend()

# Loss
axes[1].plot(dense_history.history["loss"],     label="Train loss")
axes[1].plot(dense_history.history["val_loss"], label="Val loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Dense Baseline — Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
```

In a markdown cell, answer:

- Does validation loss begin to diverge from training loss? At approximately which epoch?
- What does this pattern tell you about the model's generalization?

---

## Part 2: CNN Architecture

### Task 3: Build a CNN

Design a convolutional network with two Conv-Pool blocks followed by dense layers. Train for 20 epochs and compare test accuracy to the dense baseline.

```python
cnn_model = keras.Sequential([
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same",
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="cnn_baseline")

cnn_model.summary()

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=1,
)

cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nCNN baseline test accuracy: {cnn_test_acc:.4f}")
print(f"Improvement over dense: {(cnn_test_acc - dense_test_acc) * 100:.2f} percentage points")
```

In a markdown cell, note how much the CNN improves over the dense model and suggest why.

---

### Task 4: Apply Regularization

Rebuild the CNN with Dropout after each Dense layer, BatchNormalization after each Conv2D, and EarlyStopping to halt training when validation loss stops improving.

```python
from tensorflow.keras.callbacks import EarlyStopping

cnn_reg_model = keras.Sequential([
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same",
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
], name="cnn_regularized")

cnn_reg_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

cnn_reg_history = cnn_reg_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

cnn_reg_test_loss, cnn_reg_test_acc = cnn_reg_model.evaluate(X_test_cnn, y_test, verbose=0)
epochs_trained = len(cnn_reg_history.history["loss"])
print(f"\nRegularized CNN test accuracy: {cnn_reg_test_acc:.4f}")
print(f"Training stopped at epoch: {epochs_trained}")
```

Report the final test accuracy and the epoch at which EarlyStopping halted training.

---

## Part 3: Analysis

### Task 5: Confusion Matrix

Compute and display the confusion matrix for the regularized CNN. Identify the two class pairs that are most frequently confused.

```python
y_pred_classes = np.argmax(cnn_reg_model.predict(X_test_cnn), axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
ax.set_title("Confusion Matrix — Regularized CNN")
plt.tight_layout()
plt.show()
```

After examining the confusion matrix, write 2 sentences in a markdown cell:

- Which two class pairs have the highest off-diagonal counts (i.e., are confused most often)?
- Why might those specific classes be visually similar in 28x28 grayscale images?

---

### Task 6: Comparison Table

Fill in the table below with your results from Tasks 1, 3, and 4.

| Model | Test Accuracy | Epochs Trained |
|-------|--------------|----------------|
| Dense baseline | | 20 |
| CNN (no regularization) | | 20 |
| CNN + Dropout + BatchNorm + EarlyStopping | | |

*Which model achieves the highest test accuracy? Does stopping early appear to help or hurt performance?*

---

### Task 7: Reflection

Answer each of the following questions in a markdown cell (2–4 sentences per question).

**Question 1:** Why does the CNN outperform the dense network on Fashion-MNIST image data?

**Guidance:** Think about what information is preserved when you apply a convolutional filter versus when you flatten the image first. Consider how edges, textures, and local spatial patterns relate to clothing categories.

**Question 2:** What does Dropout prevent during training?

**Guidance:** Describe what co-adaptation means in the context of neural networks and how randomly dropping units during each training step discourages it.

**Question 3:** When would you prefer EarlyStopping over training for a fixed number of epochs?

**Guidance:** Consider situations where you do not know in advance how many epochs are needed, where training is expensive, or where the validation curve shows instability before plateauing.

---

## Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Dense baseline | 15 | Model built and compiled correctly; trained 20 epochs with validation split; test accuracy reported |
| Task 2: Training curves | 10 | Both accuracy and loss plots generated; markdown interpretation addresses overfitting and identifies approximate epoch |
| Task 3: CNN baseline | 20 | Correct Conv-Pool-Conv-Pool-Flatten-Dense architecture; trained and evaluated; accuracy compared to dense model |
| Task 4: Regularized CNN | 20 | BatchNorm after each Conv2D; Dropout(0.3) after dense layer; EarlyStopping with patience=3 and restore_best_weights; accuracy and stopping epoch reported |
| Task 5: Confusion matrix | 10 | Confusion matrix displayed with class labels; two most confused pairs identified; markdown explanation addresses visual similarity |
| Task 6: Comparison table | 10 | All three rows filled with correct values |
| Task 7: Reflection | 15 | Answers address spatial structure (Q1), co-adaptation (Q2), and adaptive stopping (Q3) with conceptual accuracy |
| **Total** | **100** | |

---

## Knowledge Check

<div class="quiz-container" data-correct="0" data-explanation="A convolutional filter is a small grid (e.g., 3x3) that slides across the image, computing a weighted sum of each local neighborhood. This local connectivity means the network learns to detect edges, corners, and textures at specific spatial positions — features that are meaningful regardless of where in the image they appear (translation equivariance). When you flatten first and feed pixels to Dense layers, each unit receives every pixel simultaneously: there is no concept of spatial adjacency. A pixel in the top-left corner is treated as completely independent of its neighbor, so 2D structure — the very structure that distinguishes a collar from a sleeve — is discarded before any learning happens.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Why does a CNN preserve spatial information that a Flatten + Dense model discards?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>CNNs apply local filters that slide across the image, learning spatial relationships such as edges and textures. Flattening first treats each pixel as independent of its neighbors, destroying the 2D structure before any learning occurs.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>CNNs use more parameters than dense networks, which gives them more capacity to memorize spatial patterns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Dense networks operate on flattened inputs, which compresses the image and removes redundant pixel information that CNNs would otherwise overfit to.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>CNNs use MaxPooling to preserve the original image dimensions throughout the network, while Flatten reduces the image to 1D.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="When training loss continues to decrease while validation loss starts to increase, the model is memorizing the training set rather than learning the underlying pattern. It has seen the training examples enough times that it fits their specific noise — the random variation that does not generalize to new data. This is the definition of overfitting. Regularization techniques (Dropout, L2 weight decay) reduce the model's ability to memorize by introducing noise or penalizing large weights. EarlyStopping addresses it by halting training before the gap between training and validation loss becomes large.">
  <div class="quiz-question">
    <strong>Question 2:</strong> During training, validation loss decreases for 5 epochs then starts increasing while training loss keeps decreasing. What is happening, and which techniques would address it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model is underfitting — it needs more layers or more neurons to capture the complexity of the data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>The model is overfitting — it has memorized the training set and fails to generalize. Dropout, L2 regularization, or EarlyStopping with restore_best_weights would address this.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The learning rate is too high — the optimizer is overshooting the loss minimum on the validation set but not on the training set.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The validation split is too small — with more validation data, the validation loss would continue to decrease alongside training loss.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="BatchNormalization computes the mean and variance of each feature across the current mini-batch, then normalizes the activations to have approximately zero mean and unit variance. It then applies learned scale and shift parameters (gamma and beta) so the layer can represent any distribution if needed. The practical effects are: (1) activations stay on a consistent scale even as earlier weights change during training, which allows larger learning rates and faster convergence; (2) the noise introduced by using mini-batch statistics rather than the full-dataset statistics acts as a mild regularizer, similar to Dropout but weaker. It does not directly reduce the number of parameters or change the loss function.">
  <div class="quiz-question">
    <strong>Question 3:</strong> BatchNormalization is placed after a Conv2D layer in your CNN. What does it do during training?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>It randomly sets some activations to zero, similar to Dropout, which prevents any single filter from becoming too dominant.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>It clips large activation values to a fixed range to prevent gradient explosion during backpropagation.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>It normalizes the activations of the layer for each mini-batch, keeping them on a stable scale. This speeds up training by allowing higher learning rates and acts as a mild regularizer by introducing noise from batch-level statistics.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It reduces the spatial dimensions of the feature maps, similar to MaxPooling, so the network has fewer parameters to learn in subsequent layers.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>
