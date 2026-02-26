# CNNs

## Overview

In the previous lesson, a dense network achieved 97.7% accuracy on MNIST by treating each pixel as an independent feature. This works for MNIST's simple, centered digits, but it ignores the spatial structure of images: nearby pixels are correlated, and meaningful patterns — edges, textures, shapes — are local and appear at different positions. Convolutional Neural Networks (CNNs) are designed specifically for this structure, and they power the computer vision systems behind self-driving cars, medical imaging, and facial recognition.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Explain how convolutional layers, pooling layers, and feature maps extract spatial features from images.
- Build and train a CNN using Keras and compare its performance to a dense baseline.

## Key Terms

**Convolution:** A mathematical operation that slides a small filter (kernel) across an image and computes the dot product at each position. Produces a feature map highlighting where a learned pattern appears in the image.

**Filter (kernel):** A small matrix of learned weights (e.g., 3×3 or 5×5) that detects a specific pattern: horizontal edges, diagonal stripes, color gradients. A `Conv2D` layer learns many filters simultaneously.

**Feature map:** The output of applying a filter to an image — a 2D activation map showing where the filter's pattern appeared and how strongly.

**Padding:** Adding zeros around the image border before convolution. `padding="same"` keeps the output the same spatial size as the input. `padding="valid"` (default) reduces size by `(filter_size − 1)`.

**Stride:** How many pixels the filter moves at each step. Stride=1 (default) moves one pixel at a time. Stride=2 halves the spatial dimensions.

**Pooling (Max Pooling):** A downsampling operation that takes the maximum value in each small region (e.g., 2×2). Reduces spatial size by half, making the representation more compact and translation-invariant.

**Translation invariance:** The property that the same pattern is detected regardless of where it appears in the image. Pooling provides approximate translation invariance.

**Receptive field:** The region of the input image that a particular neuron "sees." Early layers see small local regions; deeper layers see larger regions; the final layers see the entire image.

**Global Average Pooling:** Replaces the Flatten + Dense block by averaging each feature map into a single number. Reduces parameters dramatically and is common in modern architectures.

**Transfer learning:** Using a model pre-trained on a large dataset (e.g., ImageNet) as a starting point for a new task. Fine-tuning the top layers while keeping early layers frozen gives state-of-the-art results with minimal training data.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Why Convolutions?

A dense layer applied to a 32×32 color image (32×32×3 = 3,072 inputs) has ~3,000 weights per neuron. A layer of 128 neurons requires 128 × 3,072 = **393,216 parameters** — just for one layer. More critically, dense layers don't share information across positions: a cat's ear in the top-left corner is treated as completely different from a cat's ear in the bottom-right corner.

A 3×3 convolutional filter solves both problems:
- **Parameter sharing:** The same 9 weights (plus 1 bias) are reused at every position in the image. One filter → one feature map, regardless of image size.
- **Local connectivity:** Each output pixel depends on a 3×3 region of the input — capturing local patterns.
- **Translation invariance:** The same pattern detected anywhere becomes high activation anywhere in the feature map.

```
Input (5×5 image patch):      3×3 Filter:       Convolution output:
  1  2  3  4  5               1  0  1
  6  7  8  9  10              0  1  0        →   ...scalar at each position...
  11 12 13 14 15              1  0  1
  16 17 18 19 20
  21 22 23 24 25

At position (0,0): (1×1 + 2×0 + 3×1 + 6×0 + 7×1 + 8×0 + 11×1 + 12×0 + 13×1) = 43
At position (0,1): (2×1 + 3×0 + 4×1 + 7×0 + 8×1 + 9×0 + 12×1 + 13×0 + 14×1) = 48
...
```

A single filter detects one type of pattern. `Conv2D(32, ...)` learns 32 different filters — 32 different patterns — simultaneously.

## Dataset: Fashion-MNIST

MNIST digits are almost too easy (97%+ accuracy with any reasonable architecture). Fashion-MNIST is a drop-in replacement with 10 clothing categories — more visually complex and a better benchmark for convolutional architectures.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print(f"Classes: {class_names}")

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(class_names[y_train[i]])
    ax.axis("off")
plt.suptitle("Fashion-MNIST: Sample Images")
plt.tight_layout()
plt.show()
```

Output:
```
Train: (60000, 28, 28),  Test: (10000, 28, 28)
Classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

```python
# Normalize and add channel dimension for Conv2D
X_train_cnn = X_train[..., np.newaxis] / 255.0   # shape: (60000, 28, 28, 1)
X_test_cnn  = X_test[..., np.newaxis]  / 255.0   # shape: (10000, 28, 28, 1)

# For the dense baseline, flatten
X_train_flat = X_train.reshape(-1, 784) / 255.0
X_test_flat  = X_test.reshape(-1, 784)  / 255.0

print(f"CNN input shape:   {X_train_cnn.shape}")
print(f"Dense input shape: {X_train_flat.shape}")
```

Output:
```
CNN input shape:   (60000, 28, 28, 1)
Dense input shape: (60000, 784)
```

The `[..., np.newaxis]` adds a channel dimension: `(28, 28)` → `(28, 28, 1)`. Color images would be `(height, width, 3)` for RGB. Keras `Conv2D` always expects `(height, width, channels)`.

## Dense Baseline

```python
dense_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

dense_model.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

dense_history = dense_model.fit(
    X_train_flat, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

dense_test_acc = dense_model.evaluate(X_test_flat, y_test, verbose=0)[1]
print(f"Dense model test accuracy: {dense_test_acc:.4f}")
```

Output:
```
Dense model test accuracy: 0.8857
```

88.6% accuracy — the dense network ceiling on Fashion-MNIST.

## Building a CNN

```python
cnn_model = tf.keras.Sequential([
    # Block 1: Convolution + MaxPooling
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                           padding="same", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 2: Convolution + MaxPooling
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 3: Convolution (no pooling — image is now 7×7)
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

    # Flatten and dense head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

cnn_model.summary()
```

Output:
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)              Output Shape         Param #
=================================================================
 conv2d (Conv2D)           (None, 28, 28, 32)   320
 max_pooling2d (MaxPool)   (None, 14, 14, 32)   0
 conv2d_1 (Conv2D)         (None, 14, 14, 64)   18,496
 max_pooling2d_1 (MaxPool) (None, 7, 7, 64)     0
 conv2d_2 (Conv2D)         (None, 7, 7, 64)     36,928
 flatten_1 (Flatten)       (None, 3136)          0
 dense_3 (Dense)           (None, 128)           401,536
 dense_4 (Dense)           (None, 10)            1,290
=================================================================
Total params: 458,570
Trainable params: 458,570
Non-trainable params: 0
_________________________________________________________________
```

Tracing the spatial dimensions:
- Input: 28×28×1
- After Conv2D(32) + MaxPool(2×2): 14×14×32
- After Conv2D(64) + MaxPool(2×2): 7×7×64
- After Conv2D(64): 7×7×64
- After Flatten: 7×7×64 = 3,136

Notice: the first Conv2D layer has only **320 parameters** (3×3×1×32 filters + 32 biases), while the Dense(128) layer has **401,536**. The convolutional layers learn powerful spatial features with very few parameters; the dense head does the final classification.

## Training and Comparing

```python
cnn_model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)[1]
print(f"Dense model:  {dense_test_acc:.4f}")
print(f"CNN model:    {cnn_test_acc:.4f}")
print(f"Improvement:  {cnn_test_acc - dense_test_acc:+.4f}")
```

Output:
```
Dense model:  0.8857
CNN model:    0.9241
Improvement:  +0.0384
```

The CNN achieves 92.4% accuracy — a 3.8 percentage point improvement over the dense baseline, representing 33% fewer errors (from 1,143 to 759 on 10,000 test images). This gap would be dramatically larger on higher-resolution color images.

## Visualizing What the CNN Learns

A helpful exercise is visualizing the feature maps produced by early convolutional layers:

```python
# Build a model that outputs the first conv layer's activations
feature_model = tf.keras.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[0].output   # First Conv2D output
)

# Get feature maps for one test image
sample = X_test_cnn[0:1]   # shape: (1, 28, 28, 1)
feature_maps = feature_model.predict(sample, verbose=0)  # shape: (1, 28, 28, 32)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
    ax.axis("off")
plt.suptitle(f"First Conv2D Layer: 32 Feature Maps\n(Input: {class_names[y_test[0]]})")
plt.tight_layout()
plt.show()
```

Each of the 32 feature maps shows which regions of the image activate a particular filter. Some filters detect horizontal edges, others detect vertical edges, others respond to specific texture patterns. This is the CNN learning to see — earlier than any human engineer would have specified.

## Transfer Learning: Building on Pre-trained Networks

Training a CNN from scratch works for Fashion-MNIST's 28×28 grayscale images. For full-resolution color photos (224×224×3), training from scratch requires millions of labeled images and days of GPU time. **Transfer learning** solves this by starting from a network already trained on ImageNet (1.2 million images, 1,000 classes).

```python
# Load MobileNetV2 pre-trained on ImageNet, without the top classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,   # Exclude ImageNet's final Dense layer
    weights="imagenet"
)

# Freeze the pre-trained weights — don't update them during training
base_model.trainable = False

# Add a new classification head for your task
model_transfer = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")   # 10 classes for your task
])

model_transfer.summary()
```

Output:
```
...
Total params: 2,521,354
Trainable params: 263,178     ← only the new head
Non-trainable params: 2,258,176  ← frozen MobileNetV2 weights
```

The pre-trained MobileNetV2 has already learned to detect edges, textures, and shapes from 1.2 million natural images. By freezing its weights and training only the 263,178 new head parameters, you get near-state-of-the-art performance with very little labeled data and training time — often converging in 10–20 epochs on a few hundred images.

**When to use transfer learning:**
- Your dataset has < 10,000 images
- Your images are similar to natural photos (not medical scans, satellite images, etc.)
- You want state-of-the-art accuracy without training from scratch

## Strengths and Weaknesses of CNNs

| Strengths | Weaknesses |
|-----------|------------|
| State-of-the-art on image classification, detection, segmentation | Requires large amounts of labeled image data (mitigated by transfer learning) |
| Parameter efficiency through weight sharing | Training from scratch is computationally expensive (GPU required) |
| Translation invariance via pooling | Black box — individual filter responses are hard to interpret |
| Scales to high-resolution images | Architecture choices (depth, filter size, pooling) require tuning |
| Pre-trained models available for almost any image domain | Not designed for non-spatial data (tabular, text, audio sequences) |

## Conclusion

CNNs outperform dense networks on image tasks by exploiting spatial structure — weight sharing, local connectivity, and pooling together learn a hierarchy of visual features that no hand-engineered feature set can match. On Fashion-MNIST, the CNN improves accuracy from 88.6% to 92.4%. On real-world image tasks with hundreds of classes and high-resolution images, CNNs (or Transformer-based vision models) are essentially the only practical approach. In the next lesson, you'll learn **Recurrent Neural Networks** — the architecture designed for sequential data like text, time series, and audio, where the order of elements carries meaning.

## Practice

### Knowledge Check

#### **Question 1: A Conv2D layer has 32 filters of size 3×3, applied to a single-channel (grayscale) input. How many learnable parameters does this layer have?**

1. 32 × 28 × 28 = 25,088 — one parameter per pixel per filter.
2. 32 × (3 × 3 × 1) + 32 = 320 — nine weights per filter times one input channel, plus one bias per filter.
3. 3 × 3 × 32 = 288 — the filter weights without biases.
4. 32 × 28 × 28 × 1 = 25,088 — the number of activations in the feature map.

**Correct Answer:**
2. 32 × (3 × 3 × 1) + 32 = 320 — nine weights per filter times one input channel, plus one bias per filter.

**Explanation:**
A 3×3 filter applied to a 1-channel image has 9 weight values. With 32 filters, that's 32 × 9 = 288 weights, plus one bias per filter = 32 biases = 320 total parameters. Crucially, these 320 parameters are **shared across the entire image** — the same filter is applied at every position. Compare this to a Dense layer that would require 784 × 32 + 32 = 25,120 parameters to connect 784 inputs (flattened 28×28) to 32 neurons. The convolutional layer achieves the same spatial coverage with 98.7% fewer parameters.

---

#### **Question 2: What is the purpose of MaxPooling, and what does it do to the spatial dimensions of a feature map?**

1. MaxPooling applies a learned linear transformation to the feature map, preserving spatial dimensions while increasing the number of channels.
2. MaxPooling is a downsampling operation: it divides the feature map into non-overlapping windows (e.g., 2×2) and takes the maximum value in each window. This halves each spatial dimension (a 14×14 map becomes 7×7), reduces the number of parameters in subsequent layers, and provides approximate translation invariance — small shifts in the input produce the same maximum.
3. MaxPooling randomly drops 50% of neuron activations to prevent overfitting, similar to Dropout.
4. MaxPooling computes the average value in each window, making the representation smoother and more robust to noise.

**Correct Answer:**
2. MaxPooling is a downsampling operation: it takes the maximum value in each window (e.g., 2×2), halving each spatial dimension. This reduces parameters in later layers and provides translation invariance.

**Explanation:**
A 2×2 MaxPool on a 14×14 feature map produces a 7×7 output — the most strongly activated value in each 2×2 region survives. This has two effects: (1) compressing the spatial representation reduces the number of connections in subsequent dense layers, controlling parameter count and computation; (2) the maximum operation is somewhat robust to exact positioning — if a feature (e.g., an edge) appears one pixel off, the same maximum value is likely captured. "Average pooling" takes the mean instead of the maximum — MaxPooling is preferred for feature detection because strong activations (feature present) should dominate over weak ones (feature absent).

---

#### **Question 3: You have 500 labeled images of 5 different plant diseases and want to build a classification model. What is the most appropriate approach?**

1. Train a CNN from scratch with a deep architecture (10+ layers), using data augmentation to expand the 500 images.
2. Use a gradient boosting model on handcrafted image features (mean color, texture statistics) — 500 samples is too few for deep learning.
3. Load a pre-trained CNN (e.g., MobileNetV2 or EfficientNet trained on ImageNet), freeze the convolutional base, and train only a new Dense classification head on your 500 images. Use data augmentation (flipping, rotation, color jitter) to reduce overfitting.
4. Use a dense network on flattened pixel values — CNNs are only useful for datasets with more than 10,000 images.

**Correct Answer:**
3. Load a pre-trained CNN, freeze the convolutional base, and train only the new classification head on your 500 images. Use data augmentation to reduce overfitting.

**Explanation:**
With 500 images across 5 classes (100 per class), training a CNN from scratch would severely overfit — there's not enough data to learn good convolutional filters. A pre-trained model (trained on 1.2M ImageNet images) has already learned rich visual features: edges, textures, colors, object parts. These features transfer well to plant disease detection because both involve visual patterns in natural images. Training only the new head (typically 1–2 Dense layers) requires far less data. Data augmentation — random flips, crops, brightness changes — effectively multiplies your 500 images, further reducing overfitting.
