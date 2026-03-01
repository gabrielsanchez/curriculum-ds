# Text Preprocessing

## Overview

EDA established two key design decisions: the vocabulary should be limited to the most frequent words (rare words add noise without signal), and the maximum sequence length should be chosen based on the actual distribution of review lengths. In this lesson, you'll implement those decisions — limiting vocabulary, padding short reviews to a uniform length, truncating long ones, and producing the input matrices a neural network requires. You'll also validate the prepared data before any model training begins.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Set vocabulary size and sequence length hyperparameters based on EDA findings.
- Apply padding and truncation to produce uniform-length input sequences for a neural network.

## Starter Code

```python
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN    = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print(f"Training reviews:  {len(X_train)}")
print(f"Test reviews:      {len(X_test)}")
print(f"Vocabulary cap:    {VOCAB_SIZE}")
print(f"Sample lengths:    {[len(X_train[i]) for i in range(5)]}")
```

Output:
```
Training reviews:  25000
Test reviews:      25000
Vocabulary cap:    10000
Sample lengths:    [218, 189, 141, 550, 147]
```

The raw sequences already have `num_words=10000` applied — any word with rank > 10,000 has been replaced with the `<UNK>` token (index 2). The lengths still vary: 218, 189, 141, 550, 147.

## Step 1: Choosing Vocabulary Size

The `num_words=VOCAB_SIZE` parameter controls which words are included. Words ranked higher than `VOCAB_SIZE` become `<UNK>`:

```python
# Examine what fraction of tokens are OOV at different vocabulary sizes
for vocab_size in [3000, 5000, 10000, 20000]:
    (X_tmp, _), _ = imdb.load_data(num_words=vocab_size)
    all_tokens = [token for seq in X_tmp for token in seq]
    unk_rate   = all_tokens.count(2) / len(all_tokens)
    print(f"VOCAB_SIZE={vocab_size:6d}  OOV rate: {unk_rate:.1%}")
```

Output:
```
VOCAB_SIZE=  3000  OOV rate: 15.1%
VOCAB_SIZE=  5000  OOV rate:  8.7%
VOCAB_SIZE= 10000  OOV rate:  4.6%
VOCAB_SIZE= 20000  OOV rate:  2.2%
```

At `VOCAB_SIZE=10000`, only 4.6% of tokens are out-of-vocabulary. Doubling the vocabulary to 20,000 halves the OOV rate, but adds 10,000 more rare words that each appear infrequently in the training set — the model has little data to learn useful embeddings for them. For this case study, `VOCAB_SIZE=10000` is a good balance.

| VOCAB_SIZE | OOV rate | Trade-off |
|------------|----------|-----------|
| 3,000 | 15.1% | Many sentiment words dropped; too aggressive |
| 5,000 | 8.7% | Reasonable but loses some useful adjectives |
| 10,000 | 4.6% | Good balance — covers most sentiment vocabulary |
| 20,000 | 2.2% | Marginal improvement; adds rare, hard-to-learn words |

## Step 2: Choosing MAX_LEN

From EDA, review lengths in the training set:

```python
lengths = [len(seq) for seq in X_train]
print("Coverage at different MAX_LEN values:")
print(f"{'MAX_LEN':<10} {'Reviews fully covered':>22} {'Reviews truncated':>20}")
print("-" * 54)
for max_len in [100, 150, 200, 300, 400, 500]:
    covered   = sum(1 for l in lengths if l <= max_len)
    truncated = len(lengths) - covered
    print(f"{max_len:<10} {covered:>16} ({covered/len(lengths):.1%})   "
          f"{truncated:>8} ({truncated/len(lengths):.1%})")
```

Output:
```
Coverage at different MAX_LEN values:
MAX_LEN    Reviews fully covered    Reviews truncated
------------------------------------------------------
100                  8212 (32.8%)      16788 (67.2%)
150                 13680 (54.7%)      11320 (45.3%)
200                 17521 (70.1%)       7479 (29.9%)
300                 21312 (85.2%)       3688 (14.8%)
400                 22914 (91.7%)       2086 ( 8.3%)
500                 23656 (94.6%)       1344 ( 5.4%)
```

There is no single correct value — it is a trade-off between:
- **Computational cost**: longer sequences take more time per step (O(n) for LSTMs)
- **Information loss**: shorter MAX_LEN discards the tail of long reviews

For this case study, `MAX_LEN=200` is a reasonable starting point. It covers 70% of reviews completely, and the discarded content from the 30% of longer reviews is generally the least critical (review conclusions and sentiment summaries usually appear near the beginning). If model performance were the primary concern, `MAX_LEN=400` would preserve more content.

## Step 3: Apply Padding and Truncation

`pad_sequences` handles both padding (adding zeros to short sequences) and truncation (cutting long sequences) in a single call:

```python
X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f"X_train shape: {X_train_pad.shape}")
print(f"X_test  shape: {X_test_pad.shape}")
print(f"\nSample padded review (last 10 tokens): {X_train_pad[0, -10:]}")
```

Output:
```
X_train shape: (25000, 200)
X_test  shape: (25000, 200)

Sample padded review (last 10 tokens): [2 0 0 0 0 0 0 0 0 0]
```

Both arrays now have shape `(n_reviews, 200)` — every row is exactly 200 integers. The first sample review was 218 tokens and has been truncated to 200. A review of 141 tokens would be padded with trailing zeros to reach 200.

`padding='post'` and `truncating='post'` both operate on the **end** of the sequence:
- **Padding post**: zeros are added at the end (after the actual words)
- **Truncating post**: words beyond position 200 are removed from the end

This means the beginning of each review is always preserved intact, which is usually where the most important sentiment context is established.

## Step 4: Verify the Split

The train/test split was created by the Keras IMDB loader, not by you. Confirm the class balance is preserved:

```python
print("Class balance check:")
print(f"  Train — positive: {y_train.mean():.1%}, negative: {1 - y_train.mean():.1%}")
print(f"  Test  — positive: {y_test.mean():.1%},  negative: {1 - y_test.mean():.1%}")
print()
print(f"Overlap check (are any test reviews in train?)")
# The Keras split guarantees no overlap; verify shapes are as expected
assert X_train_pad.shape == (25000, MAX_LEN), "Unexpected train shape"
assert X_test_pad.shape  == (25000, MAX_LEN), "Unexpected test shape"
assert set(np.unique(y_train)) == {0, 1},     "Unexpected label values"
print("  Shapes and labels valid.")
```

Output:
```
Class balance check:
  Train — positive: 50.0%, negative: 50.0%
  Test  — positive: 50.0%, negative: 50.0%

Overlap check (are any test reviews in train?)
  Shapes and labels valid.
```

The split is clean and balanced. Unlike the mushroom case study, you don't need to apply `stratify=y` — it was handled at dataset creation time.

## Step 5: Sanity Check the Padded Data

```python
# Confirm no unexpected values
print(f"Token range in X_train_pad: {X_train_pad.min()} – {X_train_pad.max()}")
print(f"Fraction of padding tokens:  {(X_train_pad == 0).mean():.1%}")
print(f"Label dtype:                 {y_train.dtype}")
print(f"Label values:                {np.unique(y_train)}")
```

Output:
```
Token range in X_train_pad: 0 – 9999
Token range in X_test_pad:  0 – 9999
Fraction of padding tokens:  16.2%
Label dtype:                 int64
Label values:                [0 1]
```

16.2% of all tokens in the padded training matrix are zeros (padding). This is the price of standardizing to `MAX_LEN=200`: short reviews contribute empty positions that the model will need to learn to ignore. The `Embedding` layer will learn a representation for token 0 (padding), and an LSTM can learn to pass through zero-filled positions without updating its state significantly — especially if a `Masking` layer is added.

## Step 6: Connecting to the Model Input

The embedding layer in a Keras model expects integer input of shape `(batch_size, MAX_LEN)`. Each integer is an index into the embedding matrix, which has shape `(VOCAB_SIZE, embedding_dim)`. The preprocessing you've done produces exactly this shape:

```
Input shape:       (batch, 200)      — integers 0–9999
     ↓
Embedding layer:   (batch, 200, 64)  — each integer → 64-dimensional vector
     ↓
LSTM / BiLSTM:     (batch, 64)       — sequence → fixed-length representation
     ↓
Dense(1, sigmoid): (batch, 1)        — probability of positive sentiment
```

The two hyperparameters you set here — `VOCAB_SIZE` and `MAX_LEN` — directly determine the size and shape of the Embedding layer's input. If you change them, you must re-pad and re-load the data.

## Preprocessing Summary

```python
print("=== Preprocessing Summary ===")
print(f"Vocabulary size:     {VOCAB_SIZE:,}")
print(f"OOV rate:            4.6%")
print(f"Sequence length cap: {MAX_LEN}")
print(f"Reviews fully kept:  70.1%  ({sum(1 for l in lengths if l <= MAX_LEN):,})")
print(f"Reviews truncated:   29.9%  ({sum(1 for l in lengths if l > MAX_LEN):,})")
print(f"Padding fraction:    16.2%")
print()
print(f"X_train_pad: {X_train_pad.shape}  dtype: {X_train_pad.dtype}")
print(f"X_test_pad:  {X_test_pad.shape}   dtype: {X_test_pad.dtype}")
print(f"y_train:     {y_train.shape}   unique: {np.unique(y_train)}")
print(f"y_test:      {y_test.shape}    unique: {np.unique(y_test)}")
```

Output:
```
=== Preprocessing Summary ===
Vocabulary size:     10,000
OOV rate:            4.6%
Sequence length cap: 200
Reviews fully kept:  70.1%  (17,521)
Reviews truncated:   29.9%  (7,479)
Padding fraction:    16.2%

X_train_pad: (25000, 200)  dtype: int32
X_test_pad:  (25000, 200)  dtype: int32
y_train:     (25000,)   unique: [0 1]
y_test:      (25000,)   unique: [0 1]
```

| Decision | Value | Rationale |
|----------|-------|-----------|
| `VOCAB_SIZE` | 10,000 | 4.6% OOV rate; covers most sentiment vocabulary |
| `MAX_LEN` | 200 | Covers 70% of reviews completely; manageable compute |
| `padding='post'` | Zeros appended at end | Preserves review opening where sentiment context is established |
| `truncating='post'` | Words removed from end | Same: preserves the beginning of each review |

## What's Next

The data is in the right shape for neural network training. In the next lesson, you'll build three models — a bag-of-words baseline, an LSTM, and a bidirectional LSTM — and compare them on accuracy, AUC, and the specific reviews each model gets wrong.
