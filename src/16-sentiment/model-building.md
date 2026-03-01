# Model Building

## Overview

You now have padded input matrices of shape `(25000, 200)` — 25,000 reviews, each represented as 200 integers. In this lesson, you'll train three models in increasing order of architectural complexity: a bag-of-words logistic regression baseline, an LSTM, and a bidirectional LSTM with dropout regularization. You'll compare them on accuracy and AUC, plot their training curves, and examine specific reviews each model gets wrong. The results contain a genuine surprise.

## Learning Objectives

By the end of this lesson, you will have learned how to:

- Establish a classical NLP baseline before building deep learning models.
- Train and evaluate an LSTM and bidirectional LSTM for text classification.
- Interpret learning curves and error analysis to understand where models succeed and fail.

## The Evaluation Priority

Both sentiment classes are treated symmetrically in this domain. A misclassified positive review and a misclassified negative review have similar costs. The primary metrics are:

1. **Accuracy** — fraction of reviews correctly classified
2. **AUC** — area under the ROC curve; threshold-independent, measures overall separability
3. **F1-score** — balance of precision and recall for each class

Unlike the mushroom case study, there is no asymmetric error cost requiring threshold adjustment. The default threshold of 0.5 is appropriate.

## Model 1: Bag-of-Words + Logistic Regression (Baseline)

Before building any neural network, establish what a simple classical model can achieve. A bag-of-words model represents each review as a binary vector: position `j` is 1 if word `j` appears in the review, regardless of how many times or where:

```python
from sklearn.linear_model import LogisticRegression

def sequences_to_bow(sequences, vocab_size):
    """Convert integer sequences to binary bag-of-words matrix."""
    matrix = np.zeros((len(sequences), vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for token in seq:
            if token > 2:   # skip PAD, START, UNK
                matrix[i, token] = 1.0
    return matrix

print("Converting to bag-of-words (may take ~30 seconds)...")
X_bow_train = sequences_to_bow(X_train, VOCAB_SIZE)
X_bow_test  = sequences_to_bow(X_test,  VOCAB_SIZE)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_bow_train, y_train)

y_pred_lr  = lr.predict(X_bow_test)
y_proba_lr = lr.predict_proba(X_bow_test)[:, 1]

acc_lr = (y_pred_lr == y_test).mean()
auc_lr = roc_auc_score(y_test, y_proba_lr)

print(f"\n=== Baseline: BoW + Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))
print(f"AUC: {auc_lr:.4f}")
```

Output:
```
Converting to bag-of-words (may take ~30 seconds)...

=== Baseline: BoW + Logistic Regression ===
              precision    recall  f1-score   support

    Negative       0.90      0.89      0.90     12500
    Positive       0.89      0.90      0.90     12500

    accuracy                           0.89     25000
   macro avg       0.90      0.90      0.89     25000
weighted avg       0.90      0.90      0.89     25000

AUC: 0.9631
```

**89.4% accuracy, AUC of 0.9631.** That is a strong result for a model that ignores word order entirely — it knows "terrible" and "brilliant" appear but doesn't know if they were negated, modified, or placed in a qualifying clause. This is the benchmark the deep learning models must beat.

## Model 2: LSTM

An LSTM reads the review one token at a time, maintaining a hidden state that can carry information about earlier words forward:

```python
model_lstm = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    LSTM(64),
    Dense(1, activation='sigmoid'),
], name='lstm')

model_lstm.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model_lstm.summary()
```

Output:
```
Model: "lstm"
_________________________________________________________________
 Layer (type)            Output Shape          Param #
=================================================================
 embedding (Embedding)   (None, 200, 64)       640,000
 lstm (LSTM)             (None, 64)            33,024
 dense (Dense)           (None, 1)                 65
=================================================================
Total params: 673,089
Trainable params: 673,089
```

```python
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

print(f"\n=== LSTM ===")
y_pred_lstm = (y_proba_lstm >= 0.5).astype(int)
print(classification_report(y_test, y_pred_lstm, target_names=["Negative", "Positive"]))
print(f"AUC: {auc_lstm:.4f}")
```

Output:
```
Epoch 1/5: loss: 0.5841 - accuracy: 0.6797 - val_loss: 0.4128 - val_accuracy: 0.8234
Epoch 2/5: loss: 0.3521 - accuracy: 0.8539 - val_loss: 0.3617 - val_accuracy: 0.8448
Epoch 3/5: loss: 0.2908 - accuracy: 0.8810 - val_loss: 0.3529 - val_accuracy: 0.8516
Epoch 4/5: loss: 0.2622 - accuracy: 0.8946 - val_loss: 0.3478 - val_accuracy: 0.8548
Epoch 5/5: loss: 0.2441 - accuracy: 0.9024 - val_loss: 0.3512 - val_accuracy: 0.8564

=== LSTM ===
              precision    recall  f1-score   support

    Negative       0.88      0.86      0.87     12500
    Positive       0.86      0.88      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

AUC: 0.9509
```

**87.2% accuracy, AUC of 0.9509.** The LSTM is worse than the bag-of-words baseline on both metrics. The training accuracy is climbing (90.2% by epoch 5) while the validation loss has plateaued and ticked up — a signal of slight overfitting. Five epochs is enough to expose this problem but not enough to fully converge.

## Model 3: Bidirectional LSTM with Dropout

The bidirectional LSTM processes the review in both directions — forward from position 1 to 200, and backward from position 200 to 1 — and concatenates both hidden states. This lets each position attend to future context as well as past context. Dropout regularization reduces overfitting:

```python
model_bilstm = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
], name='bilstm')

model_bilstm.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model_bilstm.summary()
```

Output:
```
Model: "bilstm"
_________________________________________________________________
 Layer (type)            Output Shape          Param #
=================================================================
 embedding_1 (Embedding) (None, 200, 64)       640,000
 bidirectional (Bidirec) (None, 128)           131,584
 dropout (Dropout)       (None, 128)                 0
 dense_1 (Dense)         (None, 1)                129
=================================================================
Total params: 771,713
Trainable params: 771,713
```

The bidirectional layer doubles the LSTM's output size: each direction produces a 64-dimensional hidden state, which are concatenated to produce a 128-dimensional representation.

```python
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

print(f"\nTraining stopped at epoch: {early_stop.stopped_epoch + 1}")

loss_bilstm, acc_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)
y_proba_bilstm = model_bilstm.predict(X_test, verbose=0).squeeze()
auc_bilstm = roc_auc_score(y_test, y_proba_bilstm)

print(f"\n=== Bidirectional LSTM + Dropout ===")
y_pred_bilstm = (y_proba_bilstm >= 0.5).astype(int)
print(classification_report(y_test, y_pred_bilstm, target_names=["Negative", "Positive"]))
print(f"AUC: {auc_bilstm:.4f}")
```

Output:
```
Epoch 1/10: loss: 0.5212 - accuracy: 0.7198 - val_loss: 0.3791 - val_accuracy: 0.8378
Epoch 2/10: loss: 0.3194 - accuracy: 0.8698 - val_loss: 0.3451 - val_accuracy: 0.8542
Epoch 3/10: loss: 0.2671 - accuracy: 0.8928 - val_loss: 0.3290 - val_accuracy: 0.8640
Epoch 4/10: loss: 0.2313 - accuracy: 0.9097 - val_loss: 0.3342 - val_accuracy: 0.8634

Training stopped at epoch: 4

=== Bidirectional LSTM + Dropout ===
              precision    recall  f1-score   support

    Negative       0.89      0.88      0.89     12500
    Positive       0.88      0.89      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.88      0.89     25000
weighted avg       0.89      0.88      0.89     25000

AUC: 0.9604
```

**88.5% accuracy, AUC of 0.9604.** EarlyStopping fired after epoch 4 (val_loss increased from 0.3290 to 0.3342). The BiLSTM improves on the plain LSTM by 1.3 percentage points in accuracy and 0.0095 in AUC. It is now very close to the bag-of-words baseline.

## Model Comparison

```python
results = [
    {
        "Model":     "BoW + Logistic Regression",
        "Accuracy":  f"{acc_lr:.1%}",
        "AUC":       f"{auc_lr:.4f}",
        "Parameters":"~0 trainable (sklearn)",
        "Notes":     "No word order; fast to train",
    },
    {
        "Model":     "LSTM",
        "Accuracy":  f"{acc_lstm:.1%}",
        "AUC":       f"{auc_lstm:.4f}",
        "Parameters":"673,089",
        "Notes":     "Slight overfitting at 5 epochs",
    },
    {
        "Model":     "BiLSTM + Dropout",
        "Accuracy":  f"{acc_bilstm:.1%}",
        "AUC":       f"{auc_bilstm:.4f}",
        "Parameters":"771,713",
        "Notes":     "Best deep learning result; stopped early",
    },
]

import pandas as pd
print(pd.DataFrame(results).to_string(index=False))
```

Output:
```
                    Model  Accuracy     AUC  Parameters                        Notes
BoW + Logistic Regression     89.4%  0.9631  ~0 trainable (sklearn)  No word order; fast to train
                     LSTM     87.2%  0.9509         673,089           Slight overfitting at 5 epochs
         BiLSTM + Dropout     88.5%  0.9604         771,713  Best deep learning result; stopped early
```

**The bag-of-words baseline matches or beats both neural networks.** This is the key finding of the case study.

## Learning Curves

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, key, title in [
    (axes[0], 'accuracy', 'Accuracy'),
    (axes[1], 'loss',     'Loss'),
]:
    ax.plot(history_lstm.history[key],         label='LSTM train',       linestyle='-',  color='steelblue')
    ax.plot(history_lstm.history[f'val_{key}'],label='LSTM val',         linestyle='--', color='steelblue')
    ax.plot(history_bilstm.history[key],       label='BiLSTM train',     linestyle='-',  color='orange')
    ax.plot(history_bilstm.history[f'val_{key}'], label='BiLSTM val',    linestyle='--', color='orange')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

The learning curves reveal two patterns:

1. **LSTM**: training loss keeps decreasing while validation loss plateaus around epoch 3–4. The gap between training accuracy (~90%) and validation accuracy (~85%) is a signature of overfitting. The model is memorizing training examples rather than generalizing.

2. **BiLSTM + Dropout**: the dropout layer closes the train/val gap. Validation accuracy improves to ~86.4% before EarlyStopping fires. Dropout is doing its job — the model is more regularized even though it has more parameters than the plain LSTM.

## Error Analysis

Understanding what the best model gets wrong is as important as knowing its accuracy. Examine three misclassified reviews from the BiLSTM:

```python
y_pred_bilstm_labels = (y_proba_bilstm >= 0.5).astype(int)
misclassified = np.where(y_pred_bilstm_labels != y_test)[0]

print(f"Total misclassified by BiLSTM: {len(misclassified)} / {len(y_test)}")
print()

for idx in misclassified[:3]:
    true_label = 'positive' if y_test[idx] == 1 else 'negative'
    pred_label = 'positive' if y_pred_bilstm_labels[idx] == 1 else 'negative'
    prob       = y_proba_bilstm[idx]
    decoded    = decode_review(X_test_raw[idx])
    first_80   = ' '.join(decoded.split()[:80])

    print(f"Index {idx}  |  True: {true_label}  |  Predicted: {pred_label}  (p={prob:.3f})")
    print(f"  {first_80}")
    print()
```

Output:
```
Total misclassified by BiLSTM: 2875 / 25000

Index 7   |  True: negative  |  Predicted: positive  (p=0.807)
  this film is a decent enough thriller the acting is fine and the
  cinematography is actually quite good but the story is just so
  boring and predictable you can see every plot twist coming a mile
  away and the ending is just terrible

Index 23  |  True: positive  |  Predicted: negative  (p=0.189)
  well this certainly was not what i expected having heard so many
  negative things about it i was prepared for the worst but honestly
  it wasn't that bad the <UNK> was actually quite touching in places
  and the leads gave solid performances

Index 41  |  True: negative  |  Predicted: positive  (p=0.734)
  i wanted to love this movie i really did the premise is fascinating
  and the first act is genuinely gripping but then it just falls apart
  completely what started as a smart thriller turns into a mess of
  clichés by the third act absolutely unwatchable
```

These three errors illustrate common failure modes:

- **Index 7**: The model was misled by "decent", "fine", "quite good" — all positive words appearing early. The overall structure is *acknowledge-positives then condemn*, but the model weighted the early praise too heavily.

- **Index 23**: The review opens with "not what I expected" and "negative things about it" — a low-confidence negative framing — before delivering a positive verdict. The BiLSTM was swamped by the negative-framing language early in the review.

- **Index 41**: Conditional positive ("wanted to love", "genuinely gripping") followed by sharp reversal ("falls apart completely", "absolutely unwatchable"). The model likely treated the positive words in the first half independently of the negating structure.

All three involve **mixed-sentiment structure**: the review contains both strong positive and strong negative language, with the overall sentiment determined by which wins. This is exactly where bag-of-words models fail too — they can't determine which sentiment was the conclusion.

## Selecting the Final Model

```python
print("=== Final Model Recommendation ===")
print()
print("For deployment: BoW + Logistic Regression (baseline)")
print()
print("Rationale:")
print("  - Matches or exceeds both neural networks on this dataset")
print("  - Trains in seconds on a CPU (no GPU required)")
print("  - Decisions are fully interpretable (coefficient weights per word)")
print("  - Scales easily to millions of documents")
print("  - BiLSTM is worth using if interpretability is not required")
print("    and when the domain involves more complex sentence structure")
```

For this specific task on this specific dataset, the bag-of-words model is the right choice for production:

| Model | Accuracy | AUC | Interpretability | Training time |
|-------|----------|-----|-----------------|---------------|
| BoW + LR | 89.4% | 0.9631 | High (word weights) | ~30 seconds |
| LSTM | 87.2% | 0.9509 | Low | ~8 minutes (GPU) |
| BiLSTM + Dropout | 88.5% | 0.9604 | Low | ~12 minutes (GPU) |

The choice might be different on a harder dataset — longer documents, more subtle language, cross-lingual text — where sequential context matters more than high-signal vocabulary. The lesson is not "LSTMs are bad" but rather "always measure whether your added complexity is justified by your specific data."

## What's Next

In the final lesson, you'll synthesize everything this case study demonstrated — the EDA findings, preprocessing decisions, model choices, and error analysis — and reflect on what a production-ready sentiment classifier would need beyond a good test-set accuracy score.
