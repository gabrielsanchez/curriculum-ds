# Summary

## Overview

You've completed the full sentiment analysis pipeline — from a pre-indexed text dataset to trained, evaluated, and compared models. In this final lesson, you'll consolidate what the case study demonstrated, examine the key decisions made along the way, and reflect critically on when deep learning outperforms classical NLP approaches and what a production sentiment classifier would need beyond a good test-set score.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Summarize and communicate the findings of a text classification case study.
- Reflect on when deep learning adds value over classical NLP baselines and when it does not.

## What Was Built

The raw IMDB data went through a complete transformation in three steps:

```
IMDB Dataset (50,000 reviews — pre-indexed integer sequences)
    ↓ EDA: review length distribution, vocabulary frequency analysis,
           top positive/negative signal words identified
    ↓ Text Preprocessing: VOCAB_SIZE=10000 (4.6% OOV), MAX_LEN=200,
                          padding='post', truncating='post'
    ↓ Model Building: BoW baseline, LSTM, BiLSTM + Dropout
Final result: BoW + Logistic Regression, 89.4% accuracy, AUC 0.9631
```

### Pipeline Decisions Summary

| Stage | Decision | Rationale |
|-------|----------|-----------|
| EDA | Identified that sentiment is largely conveyed by strong evaluative adjectives | Predicted that bag-of-words would be competitive |
| EDA | Review lengths range from 11 to 2,494 (median 178) | Informed MAX_LEN choice |
| EDA | Positive and negative review lengths are nearly identical | Review length is not a reliable feature |
| Preprocessing | `VOCAB_SIZE = 10,000` | 4.6% OOV rate; good coverage of sentiment vocabulary |
| Preprocessing | `MAX_LEN = 200` | Covers 70% of reviews completely; manageable compute |
| Preprocessing | `padding='post', truncating='post'` | Preserves the start of each review, where sentiment context is established |
| Model Building | Established BoW + LR baseline before training any deep learning model | Provides an honest reference point; avoids measuring against nothing |
| Model Building | Used `EarlyStopping(patience=2)` for BiLSTM | Prevented overfitting; stopped training when validation loss plateaued |
| Model Selection | Selected BoW + LR for production recommendation | Matched or exceeded both neural networks on this dataset; faster, simpler, interpretable |

## Key Insights

### 1. The Baseline Is Surprisingly Strong

```python
from sklearn.metrics import roc_auc_score

print("Final performance comparison:")
print(f"{'Model':<30} {'Accuracy':>10} {'AUC':>8}")
print("-" * 50)
for name, proba in [
    ("BoW + Logistic Regression", y_proba_lr),
    ("LSTM",                      y_proba_lstm),
    ("BiLSTM + Dropout",          y_proba_bilstm),
]:
    preds = (proba >= 0.5).astype(int)
    acc   = (preds == y_test).mean()
    auc   = roc_auc_score(y_test, proba)
    print(f"{name:<30} {acc:>10.1%} {auc:>8.4f}")
```

Output:
```
Final performance comparison:
Model                          Accuracy      AUC
--------------------------------------------------
BoW + Logistic Regression         89.4%   0.9631
LSTM                              87.2%   0.9509
BiLSTM + Dropout                  88.5%   0.9604
```

The bag-of-words model beats both neural networks. This result is not a fluke — it has been replicated in academic benchmarks and practitioner experiments on IMDB. The reason is a mismatch between what the data requires and what LSTMs provide: IMDB sentiment is largely determined by which strong evaluative words appear, not by how those words are arranged relative to each other. An LSTM's sequential processing of word order adds complexity without adding signal for this specific task.

### 2. EDA Predicted the Outcome

From the EDA step, you identified that:
- Sentiment is largely expressed by strong evaluative adjectives ("brilliant", "terrible", "boring")
- More than 77% of reviews contain at least one high-signal sentiment word from a minimal lexicon
- Review lengths are nearly identical for positive and negative reviews

These findings all pointed toward a task where individual word presence is the dominant signal — exactly the conditions under which bag-of-words models excel. EDA did not just describe the data; it correctly predicted which model family would perform best.

### 3. Deep Learning Has a Computational Cost

```python
# Approximate training times
print("Training time comparison (approximate):")
print(f"  BoW matrix construction + LR fit:  ~30 seconds (CPU)")
print(f"  LSTM (5 epochs):                   ~8 minutes  (GPU)")
print(f"  BiLSTM + Dropout (4 epochs):       ~12 minutes (GPU)")
```

The bag-of-words model trained in seconds on a CPU. The neural networks required GPU acceleration and multiple minutes of training. For a production system processing millions of reviews per day, this computational difference compounds enormously. Unless the performance gain justifies the cost, simpler is better.

### 4. The Error Modes Reveal Where Deep Learning Matters

Error analysis on the BiLSTM identified three failure patterns, all involving mixed-sentiment structure:

| Error pattern | Example | Why it's hard |
|---------------|---------|---------------|
| Positive acknowledgment + negative conclusion | "Acting is fine, story is boring and ending is terrible" | Model weighted early praise too heavily |
| Negative framing + positive verdict | "I expected the worst, but it wasn't that bad" | Negating phrasing in the introduction misled the model |
| Conditional positive + sharp reversal | "Wanted to love it, first act gripping, then falls apart" | Mixed strong signals from both halves |

These are exactly the cases where a transformer model (such as DistilBERT or BERT) — which uses attention over the full review simultaneously — would be expected to outperform both the bag-of-words model and the LSTM. For harder sentiment tasks involving irony, sarcasm, or multi-sentence reasoning, the deep learning investment would be justified.

## Reflecting on Production Deployment

The mushroom case study raised the question: *what does it cost to be wrong?* The sentiment case study raises a different question: *what does it take to go from a test-set number to a system people trust?*

### Scenario: Routing Customer Support Tickets

Imagine deploying this classifier to automatically route negative product reviews to a customer support queue. Consider three concerns:

**Concern 1: Domain shift.** This model was trained on movie reviews. Product reviews use different vocabulary ("battery life", "fits true to size", "shipping was late") that the model has never seen. The embedding layer maps these words to poor representations, and the learned associations between words and sentiment do not transfer.

```
Movie review model accuracy on movie reviews:   89.4%
Movie review model accuracy on product reviews: ~72%  (expected drop from domain shift)
```

The remedy is fine-tuning on product review data — or using a pre-trained transformer like DistilBERT, which was trained on diverse text and generalizes better across domains.

**Concern 2: Calibration.** The model outputs a probability between 0 and 1. A probability of 0.51 and a probability of 0.99 are both classified as "positive," but they represent very different levels of confidence. A production system might want to:
- Route high-confidence predictions automatically
- Flag low-confidence predictions for human review
- Monitor the distribution of predicted probabilities over time for drift

**Concern 3: Temporal drift.** Sentiment vocabulary evolves. Slang terms, product-specific language, and even the polarity of some words ("sick" as a positive) shift over time. A model trained on 2023 reviews may degrade when applied to 2026 reviews. Regular retraining and performance monitoring are essential.

### The Three Things Accuracy Doesn't Measure

| Gap | What accuracy misses |
|-----|---------------------|
| Distribution shift | Whether the test set represents future inputs |
| Error cost symmetry | Whether false positives and false negatives have equal cost |
| Human context | Whether the model's output is actionable and trustworthy at deployment |

## Final Dataset Snapshot

```python
print("=== Cleaned and Prepared Dataset ===")
print(f"Training reviews:    {len(X_train):,}")
print(f"Test reviews:        {len(X_test):,}")
print(f"Vocabulary size:     {10000:,}  (4.6% OOV rate)")
print(f"Sequence length:     {200}  (70.1% of reviews fully covered)")
print(f"Class balance:       50.0% positive, 50.0% negative")
print()
print("=== Best Model: BoW + Logistic Regression ===")
from sklearn.metrics import accuracy_score
print(f"Accuracy:            {accuracy_score(y_test, (y_proba_lr >= 0.5).astype(int)):.1%}")
print(f"AUC:                 {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"Training time:       ~30 seconds (CPU)")
print(f"Interpretability:    High — logistic regression coefficients are word-level weights")
```

Output:
```
=== Cleaned and Prepared Dataset ===
Training reviews:    25,000
Test reviews:        25,000
Vocabulary size:     10,000  (4.6% OOV rate)
Sequence length:     200  (70.1% of reviews fully covered)
Class balance:       50.0% positive, 50.0% negative

=== Best Model: BoW + Logistic Regression ===
Accuracy:            89.4%
AUC:                 0.9631
Training time:       ~30 seconds (CPU)
Interpretability:    High — logistic regression coefficients are word-level weights
```

## What This Case Study Demonstrated

| Phase | Skills Applied |
|-------|----------------|
| EDA | Review length distributions, vocabulary frequency analysis, class-conditional word frequency comparison |
| Text Preprocessing | Vocabulary size selection, padding/truncation trade-offs, preparing input matrices for Keras |
| Model Building | Bag-of-words baseline, LSTM architecture, bidirectional LSTM, EarlyStopping, learning curve analysis |
| Error Analysis | Decoding misclassified sequences, identifying linguistic failure modes (negation, mixed sentiment) |
| Model Selection | Comparing classical vs. deep learning on accuracy and AUC; justifying the simpler model when it wins |

The deeper lesson is this: complexity should serve the data, not the other way around. IMDB sentiment is a task where bag-of-words captures the dominant signal. A harder text task — long-form argumentation, irony detection, cross-document reasoning — might flip this result entirely. The right approach is to measure.

## Where to Go From Here

| Extension | Approach |
|-----------|----------|
| Use a transformer model | Fine-tune DistilBERT via Hugging Face `transformers`; expect ~93–94% accuracy |
| Handle domain shift | Collect product review labels; fine-tune on the new domain |
| Improve on mixed-sentiment reviews | Filter for low-confidence predictions; add a human review step |
| Calibration | Use Platt scaling or isotonic regression to align predicted probabilities with true rates |
| Production pipeline | Wrap the BoW vectorizer and LR model in a `sklearn.pipeline.Pipeline` for single-call inference |

## Reflection Questions

1. The bag-of-words model outperformed both the LSTM and the BiLSTM on IMDB. Does this mean bag-of-words is generally better than deep learning for sentiment analysis? What properties of the IMDB dataset specifically favor bag-of-words, and what kind of text task would reverse this result?

2. The error analysis identified three misclassified reviews involving mixed-sentiment structure — reviews that acknowledge positives before condemning, or use negative framing before a positive verdict. What changes to the model or preprocessing might help with these cases? Consider architectural choices (transformers vs. LSTMs), data-level choices (longer MAX_LEN), and post-processing choices (routing low-confidence predictions to human review).

3. Imagine deploying this classifier to automatically flag negative customer reviews for follow-up. The test-set accuracy is 89.4%. What additional evaluation would you want to perform before going live? Think about the kinds of errors that would be most harmful in this deployment context.
