# Introduction to the Case Study

## Overview

In the previous modules, you learned how to build neural networks with Keras, how to represent text as token sequences, and how transformers have reshaped NLP. Now you'll apply those tools to a real deployment-grade problem: sentiment classification of movie reviews. This case study walks you through the complete pipeline — from loading a pre-indexed text dataset to a trained and evaluated deep learning model — and forces you to think about what deep learning actually adds over a simpler baseline.

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/16-sentiment/01_sentiment-case-study_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Learning Objectives

By the end of this case study, you will have learned how to:

- Frame a binary text classification problem and define success criteria appropriate to the domain.
- Apply the full NLP-to-deep-learning pipeline end-to-end on a real text dataset.

## What Is This Case Study?

Module 10 applied the classification pipeline to structured, tabular data (the mushroom dataset). This module applies the same discipline — EDA, preprocessing, feature engineering, and model selection — to **unstructured text**, and goes one step further: it compares a classical NLP baseline directly against deep learning models to answer a question practitioners face constantly: *does the added complexity actually help?*

The structure parallels module 10:

1. **Exploratory Data Analysis** — Understand the dataset's class balance, review length distributions, vocabulary composition, and early signals of what separates positive from negative reviews.
2. **Text Preprocessing** — Transform raw integer sequences into padded inputs suitable for a neural network, with principled decisions about vocabulary size and maximum sequence length.
3. **Model Building** — Train a bag-of-words baseline, an LSTM, and a bidirectional LSTM; compare them on accuracy and AUC; and analyze the errors each model makes.
4. **Summary** — Synthesize findings, reflect on when deep learning outperforms classical NLP, and consider what a production sentiment classifier would need beyond a good test-set score.

## The Dataset

The **IMDB Movie Reviews** dataset contains 50,000 movie reviews from the Internet Movie Database, split evenly into 25,000 training reviews and 25,000 test reviews. Each review is labeled as positive (1) or negative (0) based on the star rating the reviewer assigned: reviews with ≥ 7 stars are labeled positive, reviews with ≤ 4 stars are labeled negative. Reviews with intermediate scores were excluded.

The Keras built-in version of IMDB pre-processes the raw text into integer sequences, where each integer is the rank of a word by frequency — word 1 is the most common, word 10,000 is the 10,000th most common. This makes it easy to control vocabulary size without managing raw text files.

**How to load it:**

```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
```

### What Makes This Dataset Useful for Learning

- **Balanced classes** — 50% positive, 50% negative. Accuracy is a meaningful metric.
- **Variable-length sequences** — Reviews range from a dozen words to over 2,000. Choosing how to handle this variation is a real design decision.
- **Bag-of-words is competitive** — Unlike many text tasks, IMDB sentiment is largely driven by individual high-signal words ("terrible", "brilliant", "boring", "masterpiece"). A classical bag-of-words model performs surprisingly well, making this an ideal setting for comparing classical and deep learning approaches.
- **Pre-indexed vocabulary** — Because the text is already converted to integers, you can focus on model architecture rather than tokenization mechanics.

### The Business Question

> **Can we automatically classify a movie review as positive or negative with high enough accuracy to be trusted in a production recommendation or moderation system?**

Sentiment classifiers are deployed widely in industry: star-rating prediction, customer support ticket routing, social media monitoring, and product review aggregation. Unlike the mushroom dataset (where one type of error was catastrophically worse), sentiment classification generally treats both error types symmetrically — a misclassified positive review and a misclassified negative review have similar costs. Accuracy and AUC are appropriate primary metrics here.

## Starter Code

Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/16-sentiment/01_sentiment-case-study_starter.ipynb) to follow along with the case study. The notebook contains all code blocks from all four lessons.

## Loading the Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

VOCAB_SIZE = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print(f"Training set:  {len(X_train)} reviews")
print(f"Test set:      {len(X_test)} reviews")
print(f"Sample review (first 10 integers): {X_train[0][:10]}")
print(f"Sample label:  {y_train[0]}")
```

Output:
```
Training set:  25000 reviews
Test set:      25000 reviews
Sample review (first 10 integers): [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65]
Sample label:  1
```

Each review is a Python list of integers. The integers are word indices — you'll decode a few of these into readable text in the EDA lesson.

## Initial Impressions

### Class Balance

```python
import numpy as np

pos_train = y_train.sum()
neg_train = len(y_train) - pos_train
pos_test  = y_test.sum()
neg_test  = len(y_test) - pos_test

print("Training set:")
print(f"  Positive: {pos_train}  ({pos_train / len(y_train):.1%})")
print(f"  Negative: {neg_train}  ({neg_train / len(y_train):.1%})")
print()
print("Test set:")
print(f"  Positive: {pos_test}  ({pos_test / len(y_test):.1%})")
print(f"  Negative: {neg_test}  ({neg_test / len(y_test):.1%})")
```

Output:
```
Training set:
  Positive: 12500  (50.0%)
  Negative: 12500  (50.0%)

Test set:
  Positive: 12500  (50.0%)
  Negative: 12500  (50.0%)
```

Perfectly balanced — 50% positive, 50% negative in both splits. A trivial classifier that always predicts positive would achieve 50% accuracy. Any model worth deploying needs to do substantially better than that.

### Review Length Distribution

```python
lengths = [len(seq) for seq in X_train]

print(f"Min length:    {min(lengths)}")
print(f"Max length:    {max(lengths)}")
print(f"Mean length:   {np.mean(lengths):.0f}")
print(f"Median length: {np.median(lengths):.0f}")
print(f"90th pctile:   {np.percentile(lengths, 90):.0f}")
print(f"95th pctile:   {np.percentile(lengths, 95):.0f}")
```

Output:
```
Min length:    11
Max length:    2494
Mean length:   238
Median length: 178
90th pctile:   449
95th pctile:   595
```

Reviews vary enormously — from 11 words to 2,494. The median is 178 words, but the distribution is right-skewed: a small number of very long reviews pull the mean to 238. This variability has direct implications for how to set the sequence length cap in preprocessing.

### Decoding a Sample Review

The integers are usable as-is for modeling, but it's instructive to look at the actual words:

```python
word_index = imdb.get_word_index()
reverse_index = {v + 3: k for k, v in word_index.items()}
reverse_index[0] = '<PAD>'
reverse_index[1] = '<START>'
reverse_index[2] = '<UNK>'

def decode_review(sequence):
    return ' '.join([reverse_index.get(i, '?') for i in sequence])

sample = decode_review(X_train[0])
print(f"Label: {'positive' if y_train[0] == 1 else 'negative'}")
print(f"Review (first 60 words):")
print(' '.join(sample.split()[:60]))
```

Output:
```
Label: positive
Review (first 60 words):
<START> this film was just brilliant casting location scenery story
direction everyone's really suited the part they played and you could
just imagine being there robert <UNK> is an amazing actor and now
the same being director <UNK> father came from the same scottish
island as myself so i loved the fact there was a real connection
with this film the witty remarks throughout the film
```

Even truncated to the first 60 words, the positive sentiment is unmistakable — "brilliant casting", "amazing actor", "loved the fact". This illustrates why bag-of-words models work well here: the sentiment is expressed through individual words that appear early and prominently.

## What's Next

The initial scan surfaces the key design decisions for this case study:

1. **Vocabulary size** — which words to include (frequent words carry the most signal; rare words are noisy)
2. **Sequence length cap** — how to handle the wide range of review lengths without discarding too much content
3. **Baseline vs. deep learning** — whether a simple bag-of-words model already captures the sentiment signal

In the next lesson, you'll dig into the data before committing to any of these choices — examining review length by class, the most common words overall, and how the vocabulary differs between positive and negative reviews.
