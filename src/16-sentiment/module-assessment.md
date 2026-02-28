# Module Assessment

## Overview

This assessment wraps up the module 16 sentiment analysis case study. Across the preceding lessons you defined the problem, explored the IMDB dataset, built preprocessing pipelines, and evaluated deep learning models. This assessment asks you to demonstrate the complete end-to-end pipeline: data exploration, a simple TF-IDF baseline, an LSTM, a bidirectional LSTM with regularization, and a structured comparison of all three. You will also perform error analysis and reflect on what each architectural choice contributes to model performance.

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Build a complete end-to-end sentiment analysis pipeline using deep learning.
- Apply text preprocessing and tokenization for sequence models.
- Train and evaluate an LSTM and a bidirectional LSTM on the IMDB dataset.
- Compare a deep learning model against a simpler baseline and articulate the trade-offs.

## Starter Code

Complete the assessment using this [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/16-sentiment/module-assessment_starter.ipynb).

---

## The Dataset

IMDB Movie Reviews — 50,000 reviews labeled as positive (1) or negative (0) sentiment. The Keras built-in version pre-encodes words as integers ranked by frequency, making it straightforward to experiment with vocabulary size and sequence length.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN    = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Positive reviews in train: {y_train.sum()} / {len(y_train)}")
```

Output:
```
Train: (25000, 200), Test: (25000, 200)
Positive reviews in train: 12500 / 25000
```

---

## Coding Assessment

Practice the concepts from this module using the included [*Colaboratory notebook*](#). After completing all tasks, save your notebook to GitHub and submit the link for grading.

### Task 1: Data Exploration

Decode 3 sample reviews from integer sequences back to readable text using `imdb.get_word_index()`. For each review, print the original integer sequence (first 20 integers), the decoded text (first 100 words), and the label (positive or negative). Write one sentence per review describing the sentiment expressed.

```python
word_index = imdb.get_word_index()
reverse_index = {v + 3: k for k, v in word_index.items()}
reverse_index[0] = '<PAD>'
reverse_index[1] = '<START>'
reverse_index[2] = '<UNK>'

def decode_review(sequence):
    return ' '.join([reverse_index.get(i, '?') for i in sequence
                     if i != 0])   # skip padding tokens

# Load raw (unpadded) sequences for cleaner decoding
(X_raw_train, y_raw_train), _ = imdb.load_data(num_words=VOCAB_SIZE)

for i in [0, 1, 2]:
    label = 'positive' if y_raw_train[i] == 1 else 'negative'
    decoded = decode_review(X_raw_train[i])
    first_100 = ' '.join(decoded.split()[:100])

    print(f"Review {i}  |  Label: {label}")
    print(f"Integer sequence (first 20): {X_raw_train[i][:20].tolist()}")
    print(f"Decoded (first 100 words): {first_100}")
    print()
```

### Task 2: Baseline — TF-IDF + Logistic Regression

Before building any deep learning model, establish a bag-of-words baseline. Convert the padded integer sequences back to binary feature vectors (a simple approach is to create a binary matrix where entry `[i, j] = 1` if word index `j` appears in review `i`), then train `LogisticRegression(max_iter=500)` and evaluate on the test set. Report accuracy and AUC.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def sequences_to_bow(sequences, vocab_size):
    """Convert integer sequences to binary bag-of-words matrix."""
    matrix = np.zeros((len(sequences), vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for j in seq:
            if j > 0:   # skip padding
                matrix[i, j] = 1.0
    return matrix

print("Converting sequences to BoW (this may take ~30 seconds)...")
X_bow_train = sequences_to_bow(X_train, VOCAB_SIZE)
X_bow_test  = sequences_to_bow(X_test,  VOCAB_SIZE)

lr_baseline = LogisticRegression(max_iter=500, random_state=42)
lr_baseline.fit(X_bow_train, y_train)

y_pred_lr  = lr_baseline.predict(X_bow_test)
y_proba_lr = lr_baseline.predict_proba(X_bow_test)[:, 1]

acc_lr = (y_pred_lr == y_test).mean()
auc_lr = roc_auc_score(y_test, y_proba_lr)

print(f"Baseline — Accuracy: {acc_lr:.4f}  AUC: {auc_lr:.4f}")
```

### Task 3: Build and Train an LSTM

Build a Keras Sequential model with the architecture below. Compile with `binary_crossentropy` loss, the `adam` optimizer, and `accuracy` as a metric. Train for 5 epochs with a 20% validation split. Report test accuracy and AUC after training.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import roc_auc_score

model_lstm = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    LSTM(64),
    Dense(1, activation='sigmoid'),
], name='lstm_model')

model_lstm.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model_lstm.summary()

history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)

loss_lstm, acc_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)
y_proba_lstm = model_lstm.predict(X_test, verbose=0).squeeze()
auc_lstm = roc_auc_score(y_test, y_proba_lstm)

print(f"LSTM — Test Accuracy: {acc_lstm:.4f}  AUC: {auc_lstm:.4f}")
```

### Task 4: Bidirectional LSTM with Regularization

Extend the model by wrapping the LSTM in a `Bidirectional` layer and adding `Dropout(0.5)` before the output layer. Use `EarlyStopping(patience=2, restore_best_weights=True)` to prevent overfitting. Report test accuracy, AUC, and which epoch training stopped.

```python
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model_bilstm = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
], name='bilstm_model')

model_bilstm.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
)

history_bilstm = model_bilstm.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

stopped_epoch = early_stop.stopped_epoch
loss_bilstm, acc_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)
y_proba_bilstm = model_bilstm.predict(X_test, verbose=0).squeeze()
auc_bilstm = roc_auc_score(y_test, y_proba_bilstm)

print(f"Bidirectional LSTM — Stopped at epoch: {stopped_epoch + 1}")
print(f"Bidirectional LSTM — Test Accuracy: {acc_bilstm:.4f}  AUC: {auc_bilstm:.4f}")
```

### Task 5: Comparison Table

Fill in the table below with the values you obtained in Tasks 2, 3, and 4.

| Model | Test Accuracy | AUC |
|-------|--------------|-----|
| BoW + Logistic Regression (baseline) | | |
| LSTM | | |
| Bidirectional LSTM + Dropout | | |

Write 1–2 sentences summarizing the pattern you observe. Does the added complexity of each model yield a corresponding improvement in performance?

### Task 6: Error Analysis

Using the best-performing model from Task 3 or 4, identify 3 misclassified reviews. Decode each review and print the first 100 words, the true label, and the predicted label (include the predicted probability). Write one sentence per review identifying a linguistic feature that may have caused the error — for example, sarcasm, negation, domain-specific vocabulary, or a mixed-sentiment structure.

```python
y_pred_labels = (y_proba_bilstm >= 0.5).astype(int)
misclassified  = np.where(y_pred_labels != y_test)[0]

# Load raw test sequences for decoding
_, (X_raw_test, y_raw_test) = imdb.load_data(num_words=VOCAB_SIZE)

for idx in misclassified[:3]:
    true_label = 'positive' if y_test[idx] == 1 else 'negative'
    pred_label = 'positive' if y_pred_labels[idx] == 1 else 'negative'
    prob       = y_proba_bilstm[idx]
    decoded    = decode_review(X_raw_test[idx])
    first_100  = ' '.join(decoded.split()[:100])

    print(f"Index: {idx}")
    print(f"  True:      {true_label}")
    print(f"  Predicted: {pred_label}  (probability: {prob:.3f})")
    print(f"  Text:      {first_100}")
    print()
```

### Task 7: Reflection

Answer each question in 2–4 sentences in your notebook:

1. Why might an LSTM outperform a TF-IDF + Logistic Regression model on sentiment analysis, even with the same training data?
2. What does the `Bidirectional` wrapper add to the LSTM, and why might it improve sentiment classification specifically?
3. For a production deployment of this classifier — for example, automatically routing negative reviews to a customer support queue — what three additional steps would you take before going live?

---

### Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Data exploration | 10 | 3 reviews decoded correctly; integer sequences printed; sentiment described in one sentence each |
| Task 2: Baseline | 15 | BoW conversion implemented correctly; accuracy and AUC reported |
| Task 3: LSTM | 20 | Model architecture matches specification; trained for 5 epochs; test accuracy and AUC reported |
| Task 4: Bidirectional LSTM | 20 | Bidirectional wrapper and Dropout added; EarlyStopping used; stopped epoch and metrics reported |
| Task 5: Comparison table | 10 | All three models compared; written summary is specific and accurate |
| Task 6: Error analysis | 15 | 3 decoded reviews shown with true label, predicted label, and probability; linguistic features identified specifically |
| Task 7: Reflection | 10 | Answers demonstrate conceptual understanding of sequential modeling, bidirectionality, and production readiness |
| **Total** | **100** | |

---

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="TF-IDF creates a bag-of-words representation: each document is a vector of term weights, discarding all information about word order. &quot;The movie was not bad&quot; and &quot;The movie was bad, not good&quot; produce similar TF-IDF vectors because both contain the words &quot;movie,&quot; &quot;bad,&quot; &quot;not,&quot; and &quot;good&quot; at roughly the same frequencies. An LSTM reads the sequence token by token, maintaining a hidden state that carries forward contextual information — it can learn that &quot;not&quot; followed by &quot;bad&quot; expresses a different sentiment than &quot;bad&quot; alone. This sequential processing is especially valuable for negation, irony, and multi-clause sentences where the meaning depends on word order and long-range dependencies. On short, simple text, TF-IDF often performs comparably because most sentiment is expressed through single high-signal words like &quot;terrible&quot; or &quot;excellent.&quot; On longer, more complex writing, the sequential context captured by LSTMs becomes more decisive.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Why might an LSTM outperform TF-IDF + Logistic Regression on sentiment analysis?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>LSTMs are always more accurate than TF-IDF models regardless of dataset size or task type, because they have more parameters.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>LSTMs process words sequentially and can capture word order and long-range dependencies — for example, understanding &quot;not bad at all&quot; as expressing a positive sentiment — whereas TF-IDF treats each word independently and loses all positional context.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>TF-IDF cannot handle more than 10,000 words in its vocabulary, which limits its coverage of the IMDB review vocabulary and causes it to miss important sentiment words.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>LSTMs automatically remove stopwords and perform stemming during the forward pass, which reduces vocabulary noise before computing sentiment scores.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="This is a domain shift problem. The model was trained on movie review vocabulary, sentence structures, and sentiment patterns. Words like &quot;director,&quot; &quot;plot,&quot; &quot;cinematography,&quot; &quot;screenplay,&quot; and actor names are common in movie reviews but rare in product reviews. The LSTM&#039;s embedding layer maps words to learned vectors based on their co-occurrence patterns in the training corpus — those vectors are well-calibrated for movie sentiment but poorly calibrated for product sentiment signals like &quot;battery life,&quot; &quot;shipping,&quot; or &quot;fits true to size.&quot; Additionally, the writing style of product reviews (shorter, more telegraphic, often listing features) differs from movie reviews (longer, narrative, evaluative). Domain adaptation — fine-tuning on a sample of labeled product reviews, or training from scratch on product data — is the standard remedy. A transformer model pre-trained on diverse text (such as DistilBERT) would be more robust to domain shift because its representations are not tied to one domain&#039;s vocabulary.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A sentiment model trained on movie reviews achieves 89% accuracy. When deployed to classify product reviews, accuracy drops to 72%. What is the most likely cause?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The model needs to be retrained with more movie reviews — a larger training set will allow it to generalize to product reviews automatically.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Product reviews have longer texts that exceed the model&#039;s MAX_LEN=200 sequence length, causing truncation of critical sentiment signals.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The model learned vocabulary and sentiment patterns specific to movie reviews that do not transfer to product reviews, which use different terminology, writing style, and sentiment cues — a problem called domain shift.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The embedding layer cannot represent out-of-vocabulary words from product reviews, so it maps them all to the unknown token, losing too much information.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="3" data-explanation="Recurrent models process sequences one token at a time and require inputs of a consistent length (or use masking to skip padding). Padding with zeros fills short sequences to MAX_LEN — the model can learn to ignore padding tokens, particularly when a Masking layer is added before the LSTM. Truncation is the more consequential operation: a 500-word review truncated to 200 words loses 60% of its content. If the most informative sentiment words appear in the discarded portion (for example, a reviewer who spends the first 300 words summarizing the plot before delivering their verdict), the model sees only the plot summary and must predict sentiment without the evaluative conclusion. In practice, you would plot the distribution of review lengths, choose MAX_LEN to cover the majority of reviews (for example, the 90th or 95th percentile), and accept moderate padding overhead for the shorter reviews rather than discarding content from the longer ones.">
  <div class="quiz-question">
    <strong>Question 3:</strong> Padding sequences to MAX_LEN=200 and truncating longer ones — what trade-off does this introduce?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Padding increases model accuracy by standardizing input length, which allows the LSTM to learn more consistent hidden state representations.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Truncation removes random words from long reviews, introducing noise that degrades the quality of the embedding layer&#039;s representations.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Short reviews are padded with zero tokens that cause the LSTM&#039;s hidden state to decay toward zero, erasing the signal from the actual review words.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Padding adds uninformative tokens the model must learn to ignore, increasing sequence length and computation; truncation discards content from long reviews and may remove the most informative sentiment words. The choice of MAX_LEN balances computational cost against information loss.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>
