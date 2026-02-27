# RNNs

## Overview

In the previous lesson, CNNs exploited spatial structure in images — the same filter applied at every position to detect local patterns regardless of their location. Text and time series present a different structural challenge: **sequential structure** where the order of elements matters and sequences have variable lengths. A movie review of 50 words and one of 500 words are both valid inputs, but a fixed-size Dense layer cannot handle them without losing information. Recurrent Neural Networks (RNNs) — and their modern variants LSTM and GRU — are designed for exactly this structure.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Explain how RNNs, LSTMs, and GRUs process sequential data and why they were designed as improvements over vanilla RNNs.
- Build a text classification pipeline with tokenization, embedding, and an LSTM or GRU layer in Keras.

## Key Terms

**Recurrent Neural Network (RNN):** A neural network with a hidden state that is updated at each step of a sequence. The hidden state carries information from earlier steps to later steps.

**Hidden state (h_t):** The internal memory of an RNN at time step t. Computed as: `h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)`. Encodes information from all previous inputs.

**Vanishing gradient:** In vanilla RNNs, gradients shrink exponentially as they are backpropagated through many time steps, making it nearly impossible to learn long-range dependencies (e.g., subject-verb agreement across 20 words).

**Long Short-Term Memory (LSTM):** An RNN variant with a dedicated cell state and three gating mechanisms (forget gate, input gate, output gate) that control what information is retained, added, or output at each step. Designed to capture long-range dependencies.

**Gated Recurrent Unit (GRU):** A simplified LSTM with two gates (reset gate, update gate) instead of three. Fewer parameters than LSTM, often similar performance, faster to train.

**Embedding layer:** Converts integer token indices into dense real-valued vectors. A vocabulary of 10,000 words → 128-dimensional embedding learns that "good" and "great" have similar vectors. Trainable as part of the model.

**Tokenization:** Converting text into a sequence of integer indices. Each unique word (or subword) in the vocabulary maps to an integer: "the cat sat" → [4, 72, 198].

**Padding:** Making sequences the same length by appending zeros (or truncating). Required because Keras tensors have fixed dimensions. `tf.keras.preprocessing.sequence.pad_sequences()` handles this.

**Bidirectional RNN:** Runs an RNN forward over the sequence and a second RNN backward, then concatenates the hidden states. The model sees context from both before and after each word.

**Transformer:** An architecture using self-attention (not recurrence) that has largely replaced RNNs for NLP tasks. BERT, GPT, and T5 are Transformers. Introduced here conceptually; covered deeply in module 15.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/14-deep-learning/04_rnns_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Why Sequences Are Different

Consider sentiment classification: "The movie was not at all boring" vs "The movie was boring." A bag-of-words model (counting word occurrences) sees nearly the same words and may misclassify the first sentence as negative. An RNN processes words left to right, building a hidden state that captures the sequence "not at all boring" — understanding that "not" reverses the sentiment of "boring."

```
Sequence: "The  movie  was  not  at  all  boring"

RNN steps:
   h₀ → h₁ → h₂ → h₃ → h₄ → h₅ → h₆
  [The] [movie] [was] [not] [at] [all] [boring]

h₆ encodes the full sequence's meaning → classify as positive/negative
```

The hidden state h₃ (after "not") should encode something like "negation incoming." A well-trained LSTM learns this.

## The Problem with Vanilla RNNs

The basic RNN update rule `h_t = tanh(W_h × h_{t-1} + W_x × x_t)` has a fundamental problem: the gradient of the loss with respect to h_1 (computed via the chain rule) involves multiplying by `W_h` once per time step. Over 100 time steps, this gradient either:
- **Vanishes** (if |W_h| < 1): becomes effectively zero, making early steps irrelevant
- **Explodes** (if |W_h| > 1): becomes numerically unstable

LSTMs and GRUs solve this with gating mechanisms that allow gradients to flow without being multiplied at every step.

## Dataset: IMDB Sentiment Analysis

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative — a binary classification task with real text.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load IMDB — built into Keras
# vocab_size: only the top 10,000 most frequent words
# The rest are mapped to index 2 (OOV = out of vocabulary)
VOCAB_SIZE = 10000
MAX_LEN    = 200   # Truncate / pad to 200 words

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)

print(f"Train samples: {len(X_train)},  Test samples: {len(X_test)}")
print(f"Train label distribution: {np.bincount(y_train)}")
print(f"Sample review (token IDs): {X_train[0][:20]}...")
print(f"Review lengths — min: {min(len(r) for r in X_train)}, "
      f"max: {max(len(r) for r in X_train)}, "
      f"mean: {np.mean([len(r) for r in X_train]):.0f}")
```

Output:
```
Train samples: 25000,  Test samples: 25000
Train label distribution: [12500 12500]
Sample review (token IDs): [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25]...
Review lengths — min: 11, max: 2494, mean: 238
```

The IMDB data is already tokenized: each review is a list of integer token IDs. The vocabulary is the 10,000 most frequent words. Reviews range from 11 to 2,494 words — we need to standardize them.

## Preprocessing: Padding

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Truncate long reviews to MAX_LEN; pad short ones with zeros at the end
X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="post", truncating="post")

print(f"Padded train shape: {X_train_pad.shape}")
print(f"First review (padded): {X_train_pad[0][:30]}...")
```

Output:
```
Padded train shape: (25000, 200)
First review (padded): [1 14 22 16 43 530 973 1622 1385 65 458 4468 66 3941 4 173 36 256 5 25 ...]
```

`padding="post"` appends zeros at the end of short reviews. `truncating="post"` cuts off the end of long reviews. The result: a fixed-size (25000, 200) matrix ready for Keras.

## Dense Baseline: Bag of Words with Embedding

Before building an RNN, establish a dense baseline:

```python
# Global Average Pooling over embedding = bag-of-words with learned embeddings
dense_baseline = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),  # Average over all 200 time steps
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

dense_baseline.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

dense_baseline.fit(
    X_train_pad, y_train,
    epochs=20, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop], verbose=0
)

dense_acc = dense_baseline.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"Dense (GlobalAvgPool) baseline: {dense_acc:.4f}")
```

Output:
```
Dense (GlobalAvgPool) baseline: 0.8721
```

87.2% accuracy — reasonable for a model that ignores word order entirely (it's essentially a learned bag-of-words).

## The Embedding Layer

Before the sequence model processes words, each integer token must be converted to a meaningful vector. The `Embedding` layer maintains a lookup table:

```
Vocabulary: {"the": 4, "movie": 16, "great": 300, "terrible": 450, ...}

Embedding(VOCAB_SIZE=10000, output_dim=64):
  4   →  [0.23, -0.14, 0.87, ..., 0.45]  (64 numbers)
  16  →  [0.11,  0.92, 0.03, ..., -0.22] (64 numbers)
  300 →  [0.68,  0.71, 0.55, ..., 0.38]  (64 numbers)
```

The 64-dimensional vectors are learned during training. Words with similar meanings end up with similar vectors: the model learns that "great," "excellent," and "wonderful" all point in a similar direction in embedding space, while "terrible" points in the opposite direction.

Parameters: `VOCAB_SIZE × embedding_dim = 10,000 × 64 = 640,000`. These are learned from the training data.

## LSTM Model

```python
lstm_model = tf.keras.Sequential([
    # Embedding: integer tokens → 64-dim vectors
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),

    # LSTM: process sequence, return only the final hidden state
    tf.keras.layers.LSTM(64),

    # Dense head for binary classification
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

lstm_model.summary()
```

Output:
```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)              Output Shape         Param #
=================================================================
 embedding (Embedding)     (None, 200, 64)      640,000
 lstm (LSTM)               (None, 64)           33,280
 dense_5 (Dense)           (None, 32)           2,080
 dense_6 (Dense)           (None, 1)            33
=================================================================
Total params: 675,393
_________________________________________________________________
```

The LSTM(64) processes 200 time steps and returns only the **final hidden state** — a 64-dimensional summary of the entire review. The 33,280 LSTM parameters represent four gates (forget, input, output, cell), each with weights connecting the 64-dim input to the 64-dim hidden state.

```python
lstm_model.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    epochs=15, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop], verbose=0
)

lstm_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"Dense (GlobalAvgPool) baseline: {dense_acc:.4f}")
print(f"LSTM model:                     {lstm_acc:.4f}")
```

Output:
```
Dense (GlobalAvgPool) baseline: 0.8721
LSTM model:                     0.8734
```

The LSTM slightly edges out the dense baseline. On IMDB with MAX_LEN=200, the improvement from sequence modeling is modest — IMDB reviews are relatively simple and most sentiment signals are captured by individual word presence. The LSTM would show a larger advantage on tasks requiring longer-range dependencies (e.g., summarization, question answering).

## GRU: A Faster Alternative

```python
gru_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    tf.keras.layers.GRU(64),   # GRU instead of LSTM
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

gru_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
gru_model.fit(X_train_pad, y_train, epochs=15, batch_size=128,
              validation_split=0.1, callbacks=[early_stop], verbose=0)

gru_acc = gru_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"LSTM: {lstm_acc:.4f}")
print(f"GRU:  {gru_acc:.4f}")
```

Output:
```
LSTM: 0.8734
GRU:  0.8768
```

GRU achieves slightly higher accuracy with fewer parameters (25,088 vs 33,280 for the recurrent layer). On this dataset the difference is negligible — GRU is generally preferred when training speed matters, LSTM when long-range dependencies are critical.

## Bidirectional LSTM

A standard LSTM reads left to right. But "not" in "The movie was not good" only reverses meaning for words that come *after* it — the LSTM processes "not" before it sees "good." A Bidirectional LSTM runs both left-to-right and right-to-left simultaneously, giving each word access to its full context:

```python
bidir_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

bidir_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
bidir_model.fit(X_train_pad, y_train, epochs=15, batch_size=128,
                validation_split=0.1, callbacks=[early_stop], verbose=0)

bidir_acc = bidir_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"LSTM:          {lstm_acc:.4f}")
print(f"GRU:           {gru_acc:.4f}")
print(f"Bidirectional: {bidir_acc:.4f}")
```

Output:
```
LSTM:          0.8734
GRU:           0.8768
Bidirectional: 0.8847
```

The Bidirectional LSTM achieves 88.5% — the best of the three RNN variants, at the cost of doubled parameters (two LSTMs running in opposite directions, concatenated).

## Stacked RNNs

For more complex tasks, stacking multiple RNN layers allows the model to learn higher-level patterns:

```python
stacked_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),

    # First LSTM: return_sequences=True passes full sequence to next layer
    tf.keras.layers.LSTM(64, return_sequences=True),

    # Second LSTM: processes the sequence output of the first
    tf.keras.layers.LSTM(32),

    tf.keras.layers.Dense(1, activation="sigmoid"),
])

stacked_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
stacked_model.fit(X_train_pad, y_train, epochs=15, batch_size=128,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)

stacked_acc = stacked_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"Stacked LSTM: {stacked_acc:.4f}")
```

Output:
```
Stacked LSTM: 0.8801
```

`return_sequences=True` makes the first LSTM output a hidden state at **every** time step (shape: `(batch, 200, 64)`) rather than only the final step — necessary for stacking RNN layers.

## Summary Comparison

```python
print(f"\nIMDB Sentiment — Test Accuracy Comparison")
print(f"{'Model':<30} {'Test Accuracy':>14}")
print("-" * 46)
print(f"{'Dense (GlobalAvgPool)':<30} {dense_acc:>14.4f}")
print(f"{'LSTM':<30} {lstm_acc:>14.4f}")
print(f"{'GRU':<30} {gru_acc:>14.4f}")
print(f"{'Bidirectional LSTM':<30} {bidir_acc:>14.4f}")
print(f"{'Stacked LSTM':<30} {stacked_acc:>14.4f}")
```

Output:
```
IMDB Sentiment — Test Accuracy Comparison
Model                          Test Accuracy
----------------------------------------------
Dense (GlobalAvgPool)                 0.8721
LSTM                                  0.8734
GRU                                   0.8768
Bidirectional LSTM                    0.8847
Stacked LSTM                          0.8801
```

The performance spread is narrow (~1.3%) on IMDB because it is a relatively simple task — most words carry strong sentiment signals independently. Modern Transformer-based models (BERT, RoBERTa) achieve 95%+ on IMDB by pre-training on billions of words and capturing much richer contextual representations.

## The Rise of Transformers

RNNs process sequences step by step — word 200 can only "see" word 1 through 199 intermediate steps. The **Transformer** architecture (introduced in 2017) processes all words simultaneously using **self-attention**: each word directly attends to every other word, computing a weighted sum of their representations based on relevance.

```
RNN:          word₁ → word₂ → ... → word₂₀₀  (sequential)
Transformer:  word₁ ↔ every other word        (parallel attention)
```

Transformers learn richer representations and train faster (parallelizable), but have dominated NLP since BERT (2018). Module 15 covers Transformers, tokenization, and using pre-trained models (BERT, GPT) for practical NLP tasks. RNNs remain relevant for:
- Tasks requiring streaming inference (processing word by word)
- Very long sequences where Transformer's O(n²) attention is prohibitive
- Time series and audio, where CNNs + RNNs often outperform Transformers

## Strengths and Weaknesses of RNNs

| Strengths | Weaknesses |
|-----------|------------|
| Handles variable-length sequences natively | Slow to train (sequential computation) |
| Captures order and temporal structure | Struggles with very long sequences (>500 steps) |
| Memory-efficient for streaming inference | Largely superseded by Transformers for NLP |
| Works for text, time series, and audio | Vanishing gradient even in LSTMs for very long sequences |

## Conclusion

RNNs — specifically LSTMs and GRUs — address a fundamental limitation of fixed-size architectures by maintaining a hidden state that carries information across variable-length sequences. On the IMDB sentiment task, a Bidirectional LSTM achieves 88.5%, outperforming the order-agnostic dense baseline (87.2%). The lesson also introduced the broader NLP pipeline: tokenization, padding, embedding, and sequence classification. In the next lesson, you'll learn **optimization and regularization techniques** — dropout, batch normalization, learning rate scheduling, and data augmentation — which are essential for training the deeper, more capable models that push past these accuracy ceilings.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/14-deep-learning/04_rnns_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="In a 200-step sequence, the gradient of the loss with respect to step 1&#039;s hidden state is computed by multiplying through 199 Jacobian matrices (one per time step). In vanilla RNNs, these multiplications cause gradients to shrink to near-zero in early time steps — the first words effectively don&#039;t contribute to learning. The LSTM cell state acts as a &quot;conveyor belt&quot; with additive (not multiplicative) updates, allowing gradients to flow through many steps without shrinking. The forget gate learns when to clear the cell state, and the input gate learns when to add new information.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What problem do LSTM and GRU solve that vanilla RNNs cannot handle effectively?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>LSTMs and GRUs handle variable-length sequences, while vanilla RNNs can only process fixed-length inputs.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>LSTMs and GRUs solve the vanishing gradient problem: in vanilla RNNs, gradients shrink exponentially as they propagate back through many time steps, making it nearly impossible to learn dependencies between words separated by many steps. LSTM's gating mechanisms create a "highway" for gradients to flow without being multiplied at every step.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>LSTMs and GRUs support bidirectional processing, while vanilla RNNs can only process sequences left to right.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>LSTMs and GRUs are faster to train than vanilla RNNs because they have fewer parameters.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Keras compiles models to fixed-shape tensor operations. In a batch of 128 reviews with lengths from 50 to 500 words, all must become the same length before batching. Zeros (the padding token) have their own embedding vector — typically near zero after training, since they carry no information. `padding=&quot;post&quot;` (padding at the end) is preferred for RNNs: the LSTM processes real tokens first and only encounters padding zeros at the end, after it has already built up meaningful hidden states. `padding=&quot;pre&quot;` would cause the LSTM to first process all zeros before reaching the real text.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why do sequences need to be padded before being fed into a Keras model, and what does `padding="post"` mean?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Padding converts text tokens into floating-point numbers that neural networks can process. `padding="post"` means padding is applied before the beginning of each sequence.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Keras tensors require fixed dimensions — all sequences in a batch must have the same length. Padding appends zeros to short sequences (or truncates long ones) to create uniform length. `padding="post"` places zeros at the end of shorter sequences (after the real tokens), while `padding="pre"` places zeros at the beginning.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Padding is only needed for CNNs — RNNs can process variable-length sequences natively without padding.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Padding is required because LSTM layers cannot process zero values and replace them with learned embeddings.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="This is the argument for transfer learning applied to NLP, parallel to the ImageNet transfer learning argument for vision. A BERT model pre-trained on Common Crawl and Books has learned that &quot;bank&quot; has different meanings in &quot;river bank&quot; vs. &quot;bank account&quot; (context-dependent word meaning), that negation affects sentiment, and thousands of other linguistic patterns. An LSTM trained on 200,000 task-specific examples learns only from those examples. The 200,000 examples are sufficient to fine-tune BERT&#039;s top layers for your specific task, giving you both the broad linguistic knowledge and the task-specific adaptation. On most NLP benchmarks, fine-tuned BERT outperforms LSTM models trained on 10× more labeled data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> A colleague suggests that for a new text classification task with 200,000 labeled examples, you should train an LSTM from scratch. You suggest using a pre-trained Transformer model instead. What is your strongest argument?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>LSTMs can't handle text with more than 10,000 unique words, while Transformers have no vocabulary limit.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Pre-trained Transformers (e.g., BERT) have been trained on billions of words and already encode rich linguistic knowledge: grammar, semantic relationships, named entities, sentiment. Fine-tuning a pre-trained model on your 200,000 examples requires only a few epochs to adapt these rich representations to your specific task, typically achieving higher accuracy than an LSTM trained from scratch — even with 200,000 labeled examples. Training an LSTM from scratch learns word embeddings and sequence patterns only from your dataset, which is limited compared to what BERT learned from the entire web.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>LSTMs are slower to train than Transformers on GPUs, so even if the LSTM would achieve higher accuracy, the time cost makes Transformers preferable.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Transformers should always be used instead of LSTMs regardless of dataset size because they are architecturally superior.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

