# Exploratory Data Analysis

## Overview

In the previous lesson, you loaded the IMDB dataset and confirmed that it is perfectly balanced (50% positive, 50% negative), that review lengths range from 11 to 2,494 words, and that sentiment is often expressed through a handful of high-signal words. In this lesson, you'll dig into the data before making any preprocessing decisions — examining how review length relates to sentiment, which words appear most frequently overall, and how the vocabulary differs between positive and negative reviews.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Analyze the distribution of sequence lengths and identify how they relate to the target class.
- Identify the most informative vocabulary signals for sentiment classification.

## Step 1: Review Length Distribution

How long are the reviews? Plot a histogram of word counts across the full training set:

```python
lengths = [len(seq) for seq in X_train]

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(lengths, bins=60, color="steelblue", edgecolor="white")
ax.axvline(np.median(lengths), color="orange", linestyle="--", label=f"Median ({int(np.median(lengths))})")
ax.axvline(np.percentile(lengths, 90), color="red", linestyle="--", label=f"90th pctile ({int(np.percentile(lengths, 90))})")
ax.set_title("Distribution of Review Lengths (Training Set)")
ax.set_xlabel("Number of tokens")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.show()
```

```python
print(f"{'Statistic':<20} {'Value':>8}")
print("-" * 30)
for label, val in [
    ("Min",               min(lengths)),
    ("5th percentile",    np.percentile(lengths, 5)),
    ("25th percentile",   np.percentile(lengths, 25)),
    ("Median",            np.median(lengths)),
    ("75th percentile",   np.percentile(lengths, 75)),
    ("90th percentile",   np.percentile(lengths, 90)),
    ("95th percentile",   np.percentile(lengths, 95)),
    ("Max",               max(lengths)),
]:
    print(f"{label:<20} {int(val):>8}")
```

Output:
```
Statistic               Value
------------------------------
Min                        11
5th percentile             59
25th percentile           119
Median                    178
75th percentile           289
90th percentile           449
95th percentile           595
Max                      2494
```

The distribution is right-skewed. Half of all reviews are under 178 tokens, but the top 5% exceed 595 tokens and the longest review has 2,494 tokens. If you cap at 200 tokens, you cover the full content of about 57% of training reviews; the remaining 43% will be truncated.

## Step 2: Review Length by Sentiment

Do positive and negative reviews differ in length?

```python
pos_lengths = [len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 1]
neg_lengths = [len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograms overlaid
axes[0].hist(pos_lengths, bins=50, alpha=0.6, color="#2ecc71", label="Positive")
axes[0].hist(neg_lengths, bins=50, alpha=0.6, color="#e74c3c", label="Negative")
axes[0].set_title("Length Distribution by Sentiment")
axes[0].set_xlabel("Tokens")
axes[0].set_ylabel("Count")
axes[0].legend()

# Box plots
axes[1].boxplot([pos_lengths, neg_lengths], labels=["Positive", "Negative"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
axes[1].set_title("Review Length by Sentiment (Box Plot)")
axes[1].set_ylabel("Tokens")

plt.tight_layout()
plt.show()

print(f"Positive — median: {int(np.median(pos_lengths))}, mean: {np.mean(pos_lengths):.0f}")
print(f"Negative — median: {int(np.median(neg_lengths))}, mean: {np.mean(neg_lengths):.0f}")
```

Output:
```
Positive — median: 174, mean: 231
Negative — median: 182, mean: 244
```

Negative reviews are very slightly longer on average, but the difference is small (median difference of 8 tokens). **Review length is not a reliable signal for sentiment.** A long review is not more likely to be negative, and a short review is not more likely to be positive. Any model that implicitly relies on length will not find signal there.

## Step 3: Word Frequency Analysis

What are the most common words in the training set? Word frequency in the Keras IMDB dataset is already encoded — word index 4 is the 4th most frequent word (indices 1, 2, 3 are reserved tokens). To find actual word strings, decode through `reverse_index`:

```python
from collections import Counter

# Count every token occurrence across all training reviews
token_counts = Counter()
for seq in X_train:
    token_counts.update(seq)

# Convert token indices to words, excluding special tokens (0, 1, 2, 3)
word_counts = Counter()
for token, count in token_counts.items():
    if token > 3:
        word = reverse_index.get(token, '<UNK>')
        word_counts[word] = count

print("Top 20 most frequent words (all reviews):")
for word, count in word_counts.most_common(20):
    print(f"  {word:<20} {count:>8}")
```

Output:
```
Top 20 most frequent words (all reviews):
  the                  336953
  and                  164107
  a                    163174
  of                   145765
  to                   135720
  is                   107285
  br                    97049
  in                    93063
  it                    91031
  i                     87119
  this                  76246
  that                  73624
  was                   70060
  as                    59384
  for                   54367
  with                  51854
  movie                 46536
  but                   43963
  film                  43437
  not                   40871
```

The most frequent words are stopwords — articles, prepositions, and conjunctions. They appear in almost every review and carry no sentiment signal. Note that "not" appears frequently; negation words like this are exactly what bag-of-words models struggle with ("not bad at all" and "bad at all" look almost identical to a bag-of-words model).

## Step 4: Top Sentiment Words per Class

To find words that actually carry sentiment signal, compare word frequencies between positive and negative reviews. A word that appears frequently in positive reviews but rarely in negative ones is a positive signal, and vice versa:

```python
from collections import defaultdict

pos_counts = Counter()
neg_counts = Counter()

for seq, label in zip(X_train, y_train):
    if label == 1:
        pos_counts.update(seq)
    else:
        neg_counts.update(seq)

total_pos_tokens = sum(pos_counts.values())
total_neg_tokens = sum(neg_counts.values())

# Compute rate difference: (freq in positive) - (freq in negative)
# Normalize by total tokens in each class first
vocab = set(pos_counts.keys()) | set(neg_counts.keys())
signal = {}
for token in vocab:
    if token <= 3:
        continue
    pos_rate = pos_counts.get(token, 0) / total_pos_tokens
    neg_rate = neg_counts.get(token, 0) / total_neg_tokens
    signal[token] = pos_rate - neg_rate

# Sort by signal
sorted_signal = sorted(signal.items(), key=lambda x: x[1], reverse=True)

print("Top 15 words more frequent in POSITIVE reviews:")
for token, diff in sorted_signal[:15]:
    print(f"  {reverse_index.get(token, '?'):<20}  +{diff*1000:.2f} per 1k tokens")

print()
print("Top 15 words more frequent in NEGATIVE reviews:")
for token, diff in sorted_signal[-15:][::-1]:
    print(f"  {reverse_index.get(token, '?'):<20}  {diff*1000:.2f} per 1k tokens")
```

Output:
```
Top 15 words more frequent in POSITIVE reviews:
  wonderful             +0.47 per 1k tokens
  brilliant             +0.43 per 1k tokens
  perfect               +0.38 per 1k tokens
  best                  +0.36 per 1k tokens
  superb                +0.34 per 1k tokens
  beautifully           +0.29 per 1k tokens
  loved                 +0.28 per 1k tokens
  great                 +0.26 per 1k tokens
  favorite              +0.25 per 1k tokens
  enjoys                +0.22 per 1k tokens
  exceptional           +0.21 per 1k tokens
  stunning              +0.20 per 1k tokens
  fantastic             +0.20 per 1k tokens
  heartwarming          +0.19 per 1k tokens
  magnificent           +0.18 per 1k tokens

Top 15 words more frequent in NEGATIVE reviews:
  waste                 -0.52 per 1k tokens
  awful                 -0.50 per 1k tokens
  terrible              -0.48 per 1k tokens
  boring                -0.44 per 1k tokens
  worst                 -0.44 per 1k tokens
  ridiculous            -0.37 per 1k tokens
  poorly                -0.32 per 1k tokens
  horrible              -0.31 per 1k tokens
  dull                  -0.30 per 1k tokens
  supposed              -0.28 per 1k tokens
  unfortunately         -0.26 per 1k tokens
  lame                  -0.24 per 1k tokens
  cheap                 -0.22 per 1k tokens
  pathetic              -0.21 per 1k tokens
  garbage               -0.21 per 1k tokens
```

These words are strong, clear sentiment signals — and most are adjectives or adverbs that appear independently of context. A model that learns that "waste", "awful", and "terrible" strongly predict negative sentiment, and "wonderful", "brilliant", and "perfect" strongly predict positive sentiment, will already perform well. This observation explains why bag-of-words is so competitive on IMDB.

## Step 5: Visualizing the Sentiment Vocabulary

A word count comparison bar chart makes the class-specific signal visually clear:

```python
top_pos = [reverse_index.get(t, '?') for t, _ in sorted_signal[:12]]
top_neg = [reverse_index.get(t, '?') for t, _ in sorted_signal[-12:][::-1]]
top_pos_vals = [diff * 1000 for _, diff in sorted_signal[:12]]
top_neg_vals = [-diff * 1000 for _, diff in sorted_signal[-12:][::-1]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(top_pos[::-1], top_pos_vals[::-1], color="#2ecc71")
axes[0].set_title("Words Most Predictive of Positive Sentiment")
axes[0].set_xlabel("Rate difference (per 1k tokens)")

axes[1].barh(top_neg[::-1], top_neg_vals[::-1], color="#e74c3c")
axes[1].set_title("Words Most Predictive of Negative Sentiment")
axes[1].set_xlabel("Rate difference (per 1k tokens)")

plt.suptitle("Top Sentiment-Discriminating Words", fontsize=13)
plt.tight_layout()
plt.show()
```

The visualization confirms that sentiment in this dataset is largely conveyed by strong evaluative adjectives, making it a favorable setting for vocabulary-based approaches.

## Step 6: Are There Reviews Without Strong Sentiment Words?

```python
# Define a minimal sentiment lexicon from EDA findings
pos_words = {"wonderful", "brilliant", "perfect", "superb", "loved",
             "great", "favorite", "fantastic", "magnificent", "best"}
neg_words = {"waste", "awful", "terrible", "boring", "worst",
             "ridiculous", "horrible", "dull", "lame", "garbage"}

pos_word_tokens = {token for token, word in reverse_index.items()
                   if word in pos_words}
neg_word_tokens = {token for token, word in reverse_index.items()
                   if word in neg_words}

has_pos = [any(t in pos_word_tokens for t in seq) for seq in X_train]
has_neg = [any(t in neg_word_tokens for t in seq) for seq in X_train]
has_neither = [not p and not n for p, n in zip(has_pos, has_neg)]

print(f"Reviews containing a positive signal word:   {sum(has_pos) / len(X_train):.1%}")
print(f"Reviews containing a negative signal word:   {sum(has_neg) / len(X_train):.1%}")
print(f"Reviews containing neither (ambiguous):      {sum(has_neither) / len(X_train):.1%}")
```

Output:
```
Reviews containing a positive signal word:   57.3%
Reviews containing a negative signal word:   52.8%
Reviews containing neither (ambiguous):      22.4%
```

More than three-quarters of reviews contain at least one strong signal word from a very small lexicon. The 22% of reviews with neither are the harder cases — they require understanding context, negation, or more subtle evaluative language. These are the reviews where sequential models like LSTMs might provide an advantage over bag-of-words.

## EDA Summary

| Finding | Implication |
|---------|-------------|
| Dataset is perfectly balanced (50/50) | Accuracy is a meaningful metric; no need for class weighting |
| Review lengths range from 11 to 2,494 tokens (median 178) | MAX_LEN must be chosen carefully; 200 covers ~57% of reviews fully |
| Positive and negative reviews have similar length distributions | Length alone carries minimal signal |
| Sentiment is largely conveyed by strong evaluative adjectives | Bag-of-words baseline may be very competitive |
| ~22% of reviews contain no strong signal words from a simple lexicon | Sequential context will matter most for this minority |
| Top 20 most frequent words are stopwords | Vocabulary filtering (limiting to top-N words) is the primary preprocessing step |

## What's Next

EDA establishes two critical preprocessing decisions: choosing `VOCAB_SIZE` (which words to include) and `MAX_LEN` (how to handle variable-length sequences). In the next lesson, you'll convert these EDA findings into a concrete preprocessing pipeline and produce the padded input matrices that deep learning models require.
