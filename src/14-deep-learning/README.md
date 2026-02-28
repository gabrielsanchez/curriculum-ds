# Deep Learning

This module transitions from scikit-learn's `MLPClassifier` to TensorFlow/Keras — the standard deep learning framework — and introduces three foundational architectures: Dense networks, Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs). The module concludes with a practical guide to optimization and regularization: the techniques that separate a model that generalizes from one that memorizes.

All lessons use datasets built into Keras (MNIST, Fashion-MNIST, IMDB) to minimize setup friction and maximize focus on architecture and training concepts.

## Lessons

1. **Introduction to Deep Learning** — Why deep learning, representation learning, activation functions (ReLU, sigmoid, softmax), loss functions by task, backpropagation intuition, forward vs. backward pass, and a comparison of sklearn `MLPClassifier` vs. Keras on MNIST.
2. **Keras** — The Sequential API, Dense and Flatten layers, `model.compile()` (optimizer, loss, metrics), `model.fit()` with `validation_split`, history plots (loss and accuracy curves), `model.evaluate()`, model saving/loading, and the `EarlyStopping` callback.
3. **CNNs** — Convolution operation and parameter sharing, filters and feature maps, padding and pooling, building a 3-block CNN on Fashion-MNIST (88.6% dense → 92.4% CNN), feature map visualization, and transfer learning with MobileNetV2.
4. **RNNs** — Vanishing gradient problem, LSTM gating mechanisms, GRU, the text classification pipeline (tokenization → padding → embedding → LSTM → Dense), bidirectional LSTMs, stacked RNNs, and a comparison across architectures on IMDB sentiment (87–88% accuracy). Introduces Transformers as the successor to RNNs for NLP.
5. **Model Optimization & Regularization** — Diagnosing overfitting vs. underfitting from history plots, dropout (rate sweep), batch normalization (placement: Conv → BN → Activation), learning rate scheduling (`ReduceLROnPlateau`), data augmentation, L2 regularization, and a regularization decision framework. Demonstrates cumulative improvement on Fashion-MNIST (91.9% → 93.6%).
