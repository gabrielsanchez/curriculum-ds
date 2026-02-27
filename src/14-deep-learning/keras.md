# Keras

## Overview

The previous lesson established why deep learning differs from traditional ML and explained the conceptual mechanics of forward passes, backpropagation, and activation functions. This lesson shifts to hands-on practice with the Keras API — TensorFlow's high-level interface for building and training neural networks. By the end of this lesson, you will have built, trained, and evaluated a multi-layer neural network for digit classification and understood every line of code.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Build neural networks using the Keras Sequential API with Dense layers.
- Compile, train, and evaluate a model, and interpret training history plots.

## Key Terms

**Sequential API:** Keras's simplest model construction interface. Layers are stacked in sequence — the output of one layer is the input to the next. Sufficient for most feedforward networks.

**Functional API:** A more flexible Keras interface that supports models with multiple inputs, multiple outputs, and shared layers (e.g., ResNet skip connections). Introduced in the optimization lesson.

**Dense layer:** A fully connected layer where every input neuron connects to every output neuron. The fundamental building block: `output = activation(W × input + b)`.

**Flatten layer:** Reshapes a multi-dimensional input (e.g., a 28×28 image) into a 1D vector (784 values) without any learned parameters. Required before Dense layers when input is not already flat.

**`model.compile()`:** Configures the model for training by specifying the optimizer, loss function, and evaluation metrics.

**`model.fit()`:** Runs the training loop: iterates over the training data in mini-batches, computes loss, runs backpropagation, and updates weights. Returns a `History` object with per-epoch metrics.

**`model.evaluate()`:** Evaluates the trained model on a dataset and returns loss and metrics.

**`model.predict()`:** Returns raw output values (probabilities or regression predictions) for new inputs.

**`validation_split`:** A fraction of training data held out during `model.fit()` for monitoring overfitting. Keras automatically uses the last `fraction × n` samples as validation — **do not shuffle after the initial split** if classes need to be balanced; use `validation_data` instead.

**`History` object:** The return value of `model.fit()`. `history.history` is a dictionary of metric names to lists of per-epoch values (e.g., `{"loss": [...], "val_loss": [...], "accuracy": [...], "val_accuracy": [...]}`).

**Callback:** An object that performs actions at specific points during training. Common callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/14-deep-learning/02_keras_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Loading and Preparing MNIST

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1]
X_train = X_train / 255.0
X_test  = X_test  / 255.0

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print(f"Train labels: {y_train.shape}, unique: {np.unique(y_train)}")
```

Output:
```
Train: (60000, 28, 28),  Test: (10000, 28, 28)
Train labels: (60000,), unique: [0 1 2 3 4 5 6 7 8 9]
```

The data is already split: 60,000 training images and 10,000 test images. Pixel values are normalized to [0, 1] by dividing by 255. Labels are integers 0–9 — compatible with `sparse_categorical_crossentropy`.

## Building a Model with the Sequential API

```python
model = tf.keras.Sequential([
    # Input shape: 28×28 → Flatten to 784
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden layer 1: 256 neurons, ReLU activation
    tf.keras.layers.Dense(256, activation="relu"),

    # Hidden layer 2: 128 neurons, ReLU activation
    tf.keras.layers.Dense(128, activation="relu"),

    # Output layer: 10 neurons (one per digit), Softmax
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()
```

Output:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)              Output Shape         Param #
=================================================================
 flatten (Flatten)         (None, 784)          0
 dense (Dense)             (None, 256)          200,960
 dense_1 (Dense)           (None, 128)          32,896
 dense_2 (Dense)           (None, 10)           1,290
=================================================================
Total params: 235,146
Trainable params: 235,146
Non-trainable params: 0
_________________________________________________________________
```

Reading the summary:
- **Flatten:** 28×28=784 inputs, no parameters (just reshaping).
- **Dense (256):** 784 inputs × 256 neurons + 256 biases = **200,960 parameters**.
- **Dense (128):** 256 inputs × 128 neurons + 128 biases = **32,896 parameters**.
- **Dense (10):** 128 inputs × 10 neurons + 10 biases = **1,290 parameters**.
- **Total: 235,146 parameters** — all learned from 60,000 training examples.

## Compiling the Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Three decisions at compile time:

**Optimizer:** `"adam"` is the standard choice. Adam adapts the learning rate for each parameter individually and converges faster than plain SGD. Default learning rate: 0.001.

**Loss:** `"sparse_categorical_crossentropy"` is appropriate when:
- Task is multi-class classification
- Labels are integers (not one-hot vectors)
If labels were one-hot encoded (e.g., `[0, 0, 1, 0, ...]`), use `"categorical_crossentropy"` instead.

**Metrics:** `["accuracy"]` monitors classification accuracy during training. This is reported alongside the loss but does not affect what the optimizer minimizes.

## Training the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,   # Use 10% of training data for validation
    verbose=1
)
```

Output (last 5 epochs shown):
```
Epoch 16/20: loss: 0.0310 - accuracy: 0.9895 - val_loss: 0.0918 - val_accuracy: 0.9773
Epoch 17/20: loss: 0.0278 - accuracy: 0.9906 - val_loss: 0.0942 - val_accuracy: 0.9775
Epoch 18/20: loss: 0.0264 - accuracy: 0.9912 - val_loss: 0.0986 - val_accuracy: 0.9782
Epoch 19/20: loss: 0.0241 - accuracy: 0.9921 - val_loss: 0.0997 - val_accuracy: 0.9773
Epoch 20/20: loss: 0.0219 - accuracy: 0.9928 - val_loss: 0.1041 - val_accuracy: 0.9768
```

Understanding the output:
- **loss / accuracy:** Computed on the training batches.
- **val_loss / val_accuracy:** Computed on the held-out 10% validation set (6,000 images) after each epoch.
- Notice that by epoch 16, training loss is still decreasing (0.031 → 0.022) but validation loss has plateaued and may be slightly rising (0.092 → 0.104). This is early overfitting. The next lesson addresses this with regularization techniques.

**batch_size=128:** Each gradient update uses 128 randomly selected training examples. With 54,000 training samples (90% of 60,000), each epoch runs `⌈54000/128⌉ = 422` gradient update steps.

## Plotting Training History

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(history.history["loss"],     label="Train loss",      color="steelblue")
axes[0].plot(history.history["val_loss"], label="Validation loss",  color="coral")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (cross-entropy)")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()

# Accuracy curves
axes[1].plot(history.history["accuracy"],     label="Train accuracy",     color="steelblue")
axes[1].plot(history.history["val_accuracy"], label="Validation accuracy", color="coral")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()
```

What to look for in the history plot:
- **Healthy training:** Both curves decline together (loss) or rise together (accuracy) and plateau at similar values.
- **Overfitting:** Validation loss begins rising while training loss keeps falling.
- **Underfitting:** Both curves are still improving at the final epoch — increase `epochs` or `model` capacity.
- **Learning rate too high:** Loss curve is jagged or oscillates wildly.

## Evaluating on the Test Set

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

Output:
```
Test loss:     0.0893
Test accuracy: 0.9772
```

97.7% accuracy on 10,000 held-out test images — the model correctly classifies 9,772 of 10,000 digits.

## Inspecting Predictions

```python
y_proba = model.predict(X_test[:10])    # shape: (10, 10)
y_pred  = np.argmax(y_proba, axis=1)    # predicted class per image

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap="gray")
    correct = y_pred[i] == y_test[i]
    color   = "green" if correct else "red"
    ax.set_title(f"Pred: {y_pred[i]}  True: {y_test[i]}", color=color)
    ax.axis("off")
plt.suptitle("First 10 Test Predictions")
plt.tight_layout()
plt.show()
```

`model.predict()` returns the raw softmax probability vector for each input. `np.argmax(axis=1)` converts this to the predicted class index. Correct predictions are shown in green, incorrect in red.

## Saving and Loading Models

```python
# Save the full model (architecture + weights + optimizer state)
model.save("mnist_dense_model.keras")

# Load it back
loaded_model = tf.keras.models.load_model("mnist_dense_model.keras")
loaded_acc = loaded_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Loaded model accuracy: {loaded_acc:.4f}")
```

Output:
```
Loaded model accuracy: 0.9772
```

The `.keras` format saves everything needed to resume training or make predictions.

## Using Callbacks: Early Stopping

Instead of manually choosing the number of epochs, use `EarlyStopping` to halt training when validation performance stops improving:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",    # Stop when val_loss stops improving
    patience=5,            # Allow 5 epochs of no improvement
    restore_best_weights=True   # Revert to the best epoch's weights
)

model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
])
model2.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

history2 = model2.fit(
    X_train, y_train,
    epochs=100,            # Maximum epochs — early stopping will terminate earlier
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

stopped_epoch = len(history2.history["loss"])
test_acc2 = model2.evaluate(X_test, y_test, verbose=0)[1]
print(f"Training stopped at epoch {stopped_epoch}")
print(f"Test accuracy: {test_acc2:.4f}")
```

Output:
```
Training stopped at epoch 23
Test accuracy: 0.9784
```

Early stopping ran for 23 epochs (instead of 100) and `restore_best_weights=True` reverts to the epoch 18 weights (where validation loss was lowest). This is the standard practice for training deep networks — set `epochs` high and let early stopping find the right stopping point.

## The Full Keras Workflow

Every Keras project follows this pattern:

```
1. Prepare data
   ├── Load / download
   ├── Normalize (scale to [0,1] or standardize)
   └── Split into train / validation / test

2. Build the model
   ├── Choose architecture (Sequential vs. Functional)
   ├── Define layers (Dense, Conv2D, LSTM, ...)
   └── Review model.summary()

3. Compile
   ├── optimizer= (usually "adam")
   ├── loss= (depends on task)
   └── metrics= (usually ["accuracy"])

4. Train
   ├── model.fit(X_train, y_train, epochs=..., validation_split=...)
   └── Plot history to diagnose under/overfitting

5. Evaluate
   └── model.evaluate(X_test, y_test)

6. Iterate
   └── Adjust architecture, regularization, or learning rate
```

## Strengths and Limitations of Dense Networks on Images

The dense network (97.7% on MNIST) is strong, but it treats each pixel independently. Pixel 300 has no special relationship to pixels 299 and 301 in the network's view. Real images have **spatial structure**: neighboring pixels are correlated, and meaningful patterns (edges, textures) are local.

Convolutional Neural Networks (CNNs) exploit this structure explicitly, which is why they dramatically outperform dense networks on challenging image datasets. The next lesson introduces CNNs and demonstrates the accuracy difference on Fashion-MNIST — a more challenging image dataset where dense networks plateau around 88% and CNNs reach 92%+.

## Conclusion

You now have a complete Keras workflow: normalize data, build a Sequential model with Dense layers and ReLU activations, compile with Adam and sparse categorical cross-entropy, train with early stopping, and evaluate on a held-out test set. The 97.7% accuracy on MNIST represents a strong result for a fully connected network. In the next lesson, you'll learn **Convolutional Neural Networks** — the architecture that learns spatial features in images and that powers modern computer vision systems.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/14-deep-learning/02_keras_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Keras offers two equivalent cross-entropy losses for multi-class classification: `&quot;categorical_crossentropy&quot;` (expects one-hot vectors like `[0, 0, 1, 0, ...]`) and `&quot;sparse_categorical_crossentropy&quot;` (expects integer class indices like `2`). Both compute the same math — `sparse` simply skips the one-hot encoding step. Using `&quot;sparse_categorical_crossentropy&quot;` with integer labels is correct and avoids an unnecessary preprocessing step.">
  <div class="quiz-question">
    <strong>Question 1:</strong> You build a Keras model with `Dense(10, activation="softmax")` as the output layer and compile it with `loss="sparse_categorical_crossentropy"`. The training data has labels stored as integers 0–9. Is this correct?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>No — softmax with sparse_categorical_crossentropy is incorrect. You should use sigmoid with binary_crossentropy for 10-class classification.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Yes — this is correct. Softmax converts raw scores into class probabilities that sum to 1, and sparse_categorical_crossentropy expects integer class labels (not one-hot encoded vectors). This is the standard configuration for multi-class classification with integer labels.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>No — you should use `activation="relu"` in the output layer and `loss="mse"` for classification with integer labels.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>No — you need to one-hot encode the labels first and use `loss="categorical_crossentropy"` instead of sparse.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Without a validation set, you can only tell if training loss is decreasing — not whether the model is generalizing. The validation set gives a live view of generalization performance at each epoch. The &quot;last 10%&quot; detail matters: Keras takes the last `validation_split × n` samples, not a random sample. If your data is ordered by class (e.g., all digit 0s then all digit 1s), the last 10% may be unrepresentative. In that case, shuffle before fitting or use `validation_data=(X_val, y_val)` with a pre-shuffled split.">
  <div class="quiz-question">
    <strong>Question 2:</strong> What does `validation_split=0.1` do in `model.fit()`, and why is it important?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>It reserves 10% of the test data for validation — the model is evaluated on 10% of the test set during each epoch.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>It reserves the last 10% of the training data as a validation set, evaluating the model on these examples after each epoch without using them for training. This allows monitoring for overfitting: if training loss keeps decreasing while validation loss plateaues or rises, the model is overfitting.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>It randomly selects 10% of training batches to skip, reducing training time by 10%.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>It applies 10% dropout to the input layer, preventing overfitting.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Consider training for 25 epochs where validation loss bottoms at epoch 18 then rises. Without `restore_best_weights=True`, early stopping halts at epoch 23 (patience=5) and you&#039;re left with epoch 23&#039;s overfit weights. With `restore_best_weights=True`, Keras automatically rolls back to epoch 18&#039;s weights — the most generalizing checkpoint. This is equivalent to running `ModelCheckpoint` to save the best epoch and then loading it, but handled automatically. Always set `restore_best_weights=True` when using `EarlyStopping`.">
  <div class="quiz-question">
    <strong>Question 3:</strong> What does `restore_best_weights=True` do in the `EarlyStopping` callback, and why does it matter?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>It restores the model to its initial random weights if training gets worse than baseline performance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>When `EarlyStopping` halts training because validation loss stopped improving, `restore_best_weights=True` resets the model's weights to the epoch where validation loss was lowest — not the last epoch where training was stopped. Without this, you'd get the weights from the final (overfit) epoch, not the best-generalizing epoch.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>It saves the best weights to disk automatically and reloads them after training completes.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It averages the weights across all epochs to produce a more stable final model.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

