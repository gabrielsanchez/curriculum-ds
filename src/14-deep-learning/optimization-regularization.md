# Model Optimization & Regularization

## Overview

The previous lessons introduced Keras, CNNs, and RNNs — the core architectures of deep learning. A pattern appeared in each lesson: training loss continues to decrease while validation loss plateaus or rises. This **overfitting** is the central challenge of training deep networks. This lesson covers the standard toolkit for addressing it: dropout, batch normalization, learning rate scheduling, data augmentation, and early stopping. These techniques are not optional refinements — they are the difference between a model that generalizes and one that memorizes.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Apply dropout, batch normalization, and learning rate scheduling to prevent overfitting in deep networks.
- Design a regularization strategy based on diagnosing training history.

## Key Terms

**Overfitting:** When a model's training loss is much lower than validation loss, indicating it has memorized training examples rather than learning general patterns. The most common failure mode in deep learning.

**Underfitting:** When a model's training loss is still decreasing at the end of training, indicating the model has insufficient capacity or hasn't trained long enough.

**Dropout:** A regularization technique that randomly sets a fraction of neuron activations to zero during each training step. At test time, all neurons are active and their outputs are scaled. Forces the network to learn redundant representations.

**Dropout rate:** The fraction of neurons zeroed during training. Common values: 0.2–0.5. Higher rates = more regularization = more bias, less variance.

**Batch Normalization (BatchNorm):** Normalizes the activations of each layer to have mean ≈ 0 and variance ≈ 1, using learned scale and shift parameters. Stabilizes training, allows higher learning rates, and acts as mild regularization.

**Learning rate:** The step size used in gradient descent. Too high: loss oscillates or diverges. Too low: training is slow and may get stuck. The single most important hyperparameter in deep learning.

**Learning rate scheduling:** Reducing the learning rate during training according to a schedule. Common schedules: step decay, cosine annealing, `ReduceLROnPlateau`.

**ReduceLROnPlateau:** A Keras callback that reduces the learning rate by a factor when a monitored metric (e.g., validation loss) stops improving for a specified number of epochs.

**L2 regularization (weight decay):** Adds a penalty term proportional to the sum of squared weights to the loss function: `L_total = L_task + λ × Σw²`. Prevents any single weight from becoming too large.

**Data augmentation:** Randomly transforming training images (flipping, rotating, cropping, color jitter) to artificially increase the diversity of the training set. Effective when training data is limited.

**`tf.keras.layers.RandomFlip`:** An augmentation layer that randomly flips images horizontally (and optionally vertically) during training. Applied on-the-fly — each epoch, the same image may be seen in different augmented forms.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Diagnosing the Problem First

Regularization decisions should be driven by the training history. Always plot loss curves before reaching for regularization:

```
Training history patterns and their diagnosis:

Pattern 1: Both curves improving, end together
  loss:     ↘↘↘↘↘↘  (plateau)
  val_loss: ↘↘↘↘↘↘  (plateau)
  → Healthy. Model has converged.

Pattern 2: Training loss low, validation loss high and rising
  loss:     ↘↘↘↘↘↘↘↘
  val_loss: ↘↘↗↗↗↗↗↗
  → Overfitting. Add regularization (dropout, BatchNorm, L2).

Pattern 3: Both curves still declining at final epoch
  loss:     ↘↘↘↘↘↘  (still decreasing)
  val_loss: ↘↘↘↘↘↘  (still decreasing)
  → Underfitting. Train longer (increase epochs) or increase model capacity.

Pattern 4: Loss oscillates wildly
  loss:     ↗↘↗↘↗↘↗
  val_loss: ↗↘↗↘↗↘↗
  → Learning rate too high. Reduce it.
```

## Setup: Fashion-MNIST with a Baseline CNN

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize and add channel dimension
X_train = X_train[..., np.newaxis] / 255.0
X_test  = X_test[..., np.newaxis]  / 255.0

# Reserve 10% of training data for validation
X_val, y_val = X_train[-6000:], y_train[-6000:]
X_tr,  y_tr  = X_train[:-6000], y_train[:-6000]

def build_baseline_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

def compile_and_train(model, X_tr, y_tr, X_val, y_val, epochs=30):
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history

baseline = build_baseline_cnn()
baseline_history = compile_and_train(baseline, X_tr, y_tr, X_val, y_val)
baseline_acc = baseline.evaluate(X_test, y_test, verbose=0)[1]
print(f"Baseline CNN test accuracy: {baseline_acc:.4f}")
```

Output:
```
Baseline CNN test accuracy: 0.9189
```

The baseline achieves 91.9%. Now add regularization techniques one at a time to see their effect.

## Dropout

Dropout randomly zeros a fraction of neurons during each forward pass of training. This prevents neurons from co-adapting — each neuron must learn features that are useful independently of which other neurons happen to be active in that step.

```python
def build_dropout_cnn(dropout_rate=0.3):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropout_rate),          # ← after first block

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropout_rate),          # ← after second block

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),          # ← before output layer
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

dropout_rates = [0.1, 0.2, 0.3, 0.5]
print(f"{'Dropout rate':>14} {'Val acc':>10} {'Test acc':>10}")
print("-" * 36)

for rate in dropout_rates:
    model = build_dropout_cnn(rate)
    h = compile_and_train(model, X_tr, y_tr, X_val, y_val)
    val_acc  = max(h.history["val_accuracy"])
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{rate:>14.1f} {val_acc:>10.4f} {test_acc:>10.4f}")
```

Output:
```
  Dropout rate    Val acc   Test acc
------------------------------------
           0.1     0.9247     0.9214
           0.2     0.9271     0.9238
           0.3     0.9283     0.9258
           0.5     0.9212     0.9175
```

Dropout rate of 0.3 achieves the best test accuracy (92.6%) — a 0.7 percentage point improvement over the baseline. Rate 0.5 over-regularizes: too much information is dropped and the model underfits.

**Important:** Dropout is applied only during training — Keras automatically disables it during `model.evaluate()` and `model.predict()`.

## Batch Normalization

Batch Normalization normalizes the output of each layer across the current mini-batch, then applies a learned scale (γ) and shift (β). This addresses "internal covariate shift" — the changing distribution of layer inputs as earlier layers' weights update.

```python
def build_batchnorm_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),    # normalize, then activate
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

bn_model = build_batchnorm_cnn()
bn_history = compile_and_train(bn_model, X_tr, y_tr, X_val, y_val)
bn_acc = bn_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Batch Normalization CNN test accuracy: {bn_acc:.4f}")
```

Output:
```
Batch Normalization CNN test accuracy: 0.9264
```

BatchNorm improves accuracy to 92.6% and typically allows the model to reach convergence faster (in fewer epochs). Note the order: `Conv2D → BatchNorm → Activation`. BatchNorm is applied before the activation to normalize the raw linear output.

**BatchNorm benefits:**
- Acts as mild regularization (reduces sensitivity to initialization)
- Allows higher learning rates (more stable gradients)
- Reduces the need for careful weight initialization

## Combining Dropout and Batch Normalization

```python
def build_best_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

best_model = build_best_cnn()
best_history = compile_and_train(best_model, X_tr, y_tr, X_val, y_val, epochs=40)
best_acc = best_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Baseline CNN:          {baseline_acc:.4f}")
print(f"Dropout (0.3):         {dropout_rates[2]:.4f}  →  0.9258")
print(f"Batch Normalization:   {bn_acc:.4f}")
print(f"Combined (best model): {best_acc:.4f}")
```

Output:
```
Baseline CNN:          0.9189
Dropout (0.3):         0.3000  →  0.9258
Batch Normalization:   0.9264
Combined (best model): 0.9318
```

Combining both techniques (93.2%) outperforms either alone.

## Learning Rate Scheduling

The default Adam learning rate (0.001) is good at the start of training — large steps make fast progress. Later in training, a smaller rate is needed to fine-tune without overshooting the minimum.

### ReduceLROnPlateau

```python
model_lr = build_best_cnn()
model_lr.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # Multiply learning rate by 0.5 when triggered
        patience=5,        # Wait 5 epochs of no improvement before reducing
        min_lr=1e-6,       # Minimum learning rate floor
        verbose=1
    )
]

lr_history = model_lr.fit(
    X_tr, y_tr,
    epochs=60,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=0
)

lr_acc = model_lr.evaluate(X_test, y_test, verbose=0)[1]
print(f"Combined + LR scheduling: {lr_acc:.4f}")
```

Output:
```
Epoch 18: ReduceLROnPlateau reducing learning rate to 0.0005.
Epoch 24: ReduceLROnPlateau reducing learning rate to 0.00025.
Combined + LR scheduling: 0.9341
```

LR scheduling provides another ~0.2% improvement by allowing the optimizer to converge more precisely after initial fast learning.

## Data Augmentation

When training data is limited, artificially increasing its diversity reduces overfitting. Augmentation is applied only during training — the validation and test sets are evaluated on the original, unaugmented images.

```python
# Keras augmentation layers (applied inside the model, GPU-accelerated)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),      # ±10% rotation
    tf.keras.layers.RandomZoom(0.1),          # ±10% zoom
], name="augmentation")

def build_augmented_cnn():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = augmentation(inputs, training=True)   # Only augment during training
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

aug_model = build_augmented_cnn()
aug_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

aug_model.fit(X_tr, y_tr, epochs=50, batch_size=64,
              validation_data=(X_val, y_val),
              callbacks=callbacks, verbose=0)

aug_acc = aug_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"With data augmentation: {aug_acc:.4f}")
```

Output:
```
With data augmentation: 0.9356
```

Data augmentation pushes to 93.6% — but requires more epochs to converge (the model sees each image in different augmented forms, which takes longer to learn from).

**Note on Fashion-MNIST:** Augmentation gains are modest here because Fashion-MNIST images are small and already somewhat standardized. On real-world photographs with varying lighting, angles, and backgrounds, augmentation gains are much larger.

## L2 Weight Regularization

L2 regularization penalizes large weights, encouraging the model to distribute information across many small weights rather than a few large ones:

```python
from tensorflow.keras import regularizers

model_l2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                           kernel_regularizer=regularizers.l2(0.001),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                           kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu",
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

In practice, L2 regularization is less commonly used in CNNs than dropout + BatchNorm, which are more effective regularizers for deep networks. L2 is more useful for linear models and shallow networks.

## Full Comparison

```python
print(f"\nFashion-MNIST Regularization Comparison")
print(f"{'Model':<35} {'Test Accuracy':>14}")
print("-" * 51)
results = [
    ("Baseline CNN (no regularization)", baseline_acc),
    ("+ Dropout (0.3)",                  0.9258),
    ("+ Batch Normalization",             bn_acc),
    ("+ Dropout + BatchNorm",             best_acc),
    ("+ LR Scheduling",                   lr_acc),
    ("+ Data Augmentation",               aug_acc),
]
for name, acc in results:
    print(f"{name:<35} {acc:>14.4f}")
```

Output:
```
Fashion-MNIST Regularization Comparison
Model                               Test Accuracy
---------------------------------------------------
Baseline CNN (no regularization)           0.9189
+ Dropout (0.3)                            0.9258
+ Batch Normalization                      0.9264
+ Dropout + BatchNorm                      0.9318
+ LR Scheduling                            0.9341
+ Data Augmentation                        0.9356
```

Each technique adds a small but consistent improvement. In production, these are applied together rather than evaluated individually.

## The Regularization Decision Framework

When a model overfits, apply these techniques in order of increasing complexity:

```
1. Early Stopping
   → Always use. Zero cost, prevents wasted compute on overfit models.

2. Dropout (start with rate=0.2–0.3)
   → Add after convolutional blocks and before the output Dense layer.
   → Increase rate if still overfitting; decrease if underfitting.

3. Batch Normalization
   → Add after every Conv2D or Dense layer, before the activation.
   → Especially helpful for deep networks (5+ layers).

4. Learning Rate Scheduling (ReduceLROnPlateau)
   → Add when training loss stops decreasing but hasn't converged.
   → Reduce by factor 0.5 when val_loss plateaus for 5 epochs.

5. Data Augmentation
   → Add when training set is small (< 10,000 images).
   → Choose augmentations that preserve valid examples (horizontal flips for photos, not for text).

6. L2 Regularization
   → Use when the above are insufficient.
   → Start with lambda=0.001.

7. Reduce Model Capacity
   → Last resort: fewer layers or fewer neurons if all else fails.
```

## Conclusion

Overfitting is the central challenge of deep learning, and the techniques in this lesson — dropout, batch normalization, learning rate scheduling, and data augmentation — are the standard tools for addressing it. Applied together, they improved Fashion-MNIST accuracy from 91.9% to 93.6% — without changing the underlying architecture. These techniques compose: in production deep learning systems, all of them are typically applied simultaneously. Module 15 applies this full toolkit to NLP, introducing Transformers and pre-trained models (BERT) for text classification, named entity recognition, and question answering.

## Practice

### Knowledge Check

#### **Question 1: Dropout is set to `rate=0.5` on a Dense layer with 256 neurons during training. How many neurons are active on average in each forward pass, and what happens at test time?**

1. During training, 128 neurons are active on average (256 × 0.5). At test time, dropout is disabled and all 256 neurons are active — Keras automatically scales their outputs by 0.5 to account for the fact that they were only active 50% of the time during training.
2. During training, 256 neurons are active (dropout doesn't change the number of active neurons, only their weights). At test time, 128 neurons are randomly selected to be active.
3. During training, 256 neurons are active, but half their weights are set to zero permanently. At test time, the network has only 128 effective connections.
4. During training, 128 neurons are active. At test time, the same 128 neurons are used (no scaling required).

**Correct Answer:**
1. During training, 128 neurons are active on average (256 × 0.5 = 128). At test time, all 256 neurons are active — Keras automatically scales their outputs by 0.5 (the "inverted dropout" technique) to maintain the same expected activation magnitude.

**Explanation:**
During training with rate=0.5, each neuron is independently zeroed with probability 0.5 at each forward pass — different neurons are dropped at each step. At test time, no dropout occurs: all 256 neurons contribute. Without scaling, the expected activation magnitude would double (128 active neurons → 256 active neurons). Keras uses "inverted dropout": during training, the surviving neurons' outputs are scaled up by `1/(1-rate)` so the expected magnitude is the same at train and test time. This makes inference efficient — no scaling needed at test time.

---

#### **Question 2: You add Batch Normalization to your CNN and notice that it converges in 15 epochs instead of 30. Why does BatchNorm speed up training?**

1. BatchNorm speeds up training by reducing the number of parameters in the model, making each gradient update faster to compute.
2. BatchNorm normalizes each layer's inputs to have mean ≈ 0 and variance ≈ 1 after every mini-batch update. This stabilizes the distribution of inputs to each layer — as earlier layers' weights change, later layers don't need to continuously re-adapt to shifting input distributions. This allows the model to use larger learning rates without instability, and means each layer's weight updates are effective immediately rather than being undermined by changing input statistics.
3. BatchNorm speeds up training by randomly dropping neuron activations, reducing the complexity of the optimization landscape.
4. BatchNorm performs automatic hyperparameter tuning, selecting the optimal learning rate at each epoch.

**Correct Answer:**
2. BatchNorm normalizes each layer's inputs to mean ≈ 0 and variance ≈ 1, stabilizing the distribution of inputs to each layer. This allows larger learning rates and ensures each layer's updates are effective immediately rather than being undermined by shifting input statistics from earlier layers.

**Explanation:**
Without BatchNorm, if the first layer's weights change significantly in one step, the distribution of inputs to the second layer changes — the second layer must then re-adapt to this new distribution, essentially "chasing a moving target." With BatchNorm, every layer receives approximately unit-normalized inputs regardless of what happened in previous layers. This is called reducing "internal covariate shift." The practical effect: training is more stable, allowing 2–3× larger learning rates, which means fewer epochs to convergence. The learned γ (scale) and β (shift) parameters allow the network to undo the normalization if that's optimal for a particular layer.

---

#### **Question 3: For which scenario is data augmentation most effective, and what augmentations would be inappropriate for the given data?**

1. Data augmentation is most effective for large datasets (>100,000 images) because small datasets can't benefit from augmentation. Inappropriate augmentation for a car photo dataset would be vertical flipping.
2. Data augmentation is most effective when training data is scarce (< 10,000 images) relative to model capacity. For a medical X-ray dataset (chest X-rays, upright orientation), horizontal flipping is appropriate (the chest is symmetric), but vertical flipping is inappropriate (flipping upside-down creates anatomically impossible images that would confuse the model, as the heart and lungs have defined positions relative to the image orientation).
3. Data augmentation is only effective for black-and-white images — it is not applicable to color images or text data.
4. Data augmentation is most effective for regression problems — it should not be used for classification because it changes the class labels.

**Correct Answer:**
2. Data augmentation is most effective when training data is scarce. For X-rays, horizontal flipping is appropriate (chest is symmetric) but vertical flipping is not (anatomically impossible).

**Explanation:**
The key principle of data augmentation is that the augmented examples must remain valid and representative of what the model would see at test time. For natural photos (cats, dogs, cars), horizontal flipping, moderate rotation, and brightness changes are all valid — a cat photographed from slightly different angles is still a cat. For medical images in fixed orientation (X-rays, MRIs), vertical flipping would create images that never occur in clinical practice, potentially teaching the model wrong features. For digit recognition, rotating digits by 180° turns a 6 into a 9 — an inappropriate augmentation that would corrupt labels. Always ask: "Would a human expert recognize this augmented image as a valid example of the same class?"
