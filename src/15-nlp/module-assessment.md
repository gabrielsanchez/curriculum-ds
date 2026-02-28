# Module Assessment

## Overview

This assessment brings together the four lessons in module 15: text preprocessing and vectorization (tokenization, stopwords, stemming, lemmatization, CountVectorizer, TF-IDF), word embeddings and transformers (GloVe, self-attention, Hugging Face DistilBERT), and NLP workflows (text classification, named entity recognition, summarization, zero-shot classification). You will build a complete text classification pipeline from raw text to a tuned model, compare bag-of-words and neural approaches, and practice the error analysis skills needed to diagnose real-world NLP failures.

## Learning Objectives

By the end of this assessment, you will have demonstrated the ability to:

- Build a complete text classification pipeline from raw text to model evaluation.
- Compare bag-of-words (TF-IDF) and transformer-based approaches.
- Preprocess text and analyze the effect of preprocessing choices on model performance.
- Interpret classification reports and analyze common misclassifications.

## Starter Code

Use the included [*Colaboratory notebook*](#) to complete this assessment. [Submit the link to the AI Grader for grading](https://ai-grader-pql9.onrender.com/)

---

## The Dataset

20 Newsgroups — a collection of approximately 18,000 newsgroup posts across 20 categories. For this assessment you will use a four-category subset that represents two science topics and two non-science topics, making the multi-class boundaries meaningful and non-trivial.

```python
from sklearn.datasets import fetch_20newsgroups

# Use 4 categories for a manageable multi-class problem
categories = ['sci.med', 'sci.space', 'rec.sport.hockey', 'talk.politics.guns']

train = fetch_20newsgroups(subset='train', categories=categories,
                           remove=('headers', 'footers', 'quotes'))
test  = fetch_20newsgroups(subset='test',  categories=categories,
                           remove=('headers', 'footers', 'quotes'))

print(f"Train: {len(train.data)} documents")
print(f"Test:  {len(test.data)} documents")
print(f"Categories: {train.target_names}")
```

Output:
```
Train: 2170 documents
Test:  1447 documents
Categories: ['rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.guns']
```

---

## Coding Assessment

Practice the concepts from this module using the included [*Colaboratory notebook*](#). After completing all tasks, save your notebook to GitHub and submit the link for grading.

### Task 1: Exploratory Text Analysis

Load the dataset. For each category, compute and print: the number of documents, the average document length in characters, and the 3 most common non-stopword tokens (use NLTK stopwords or a simple frequency count after lowercasing and splitting on whitespace). Write 2 sentences describing any differences you observe between categories — for example, whether one category uses longer documents or whether the vocabulary differs noticeably.

```python
import nltk
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

for label_id, category in enumerate(train.target_names):
    docs = [train.data[i] for i in range(len(train.data))
            if train.target[i] == label_id]

    avg_len = sum(len(d) for d in docs) / len(docs)

    all_tokens = []
    for doc in docs:
        tokens = [t.lower() for t in doc.split() if t.lower() not in stop
                  and t.isalpha()]
        all_tokens.extend(tokens)

    top3 = Counter(all_tokens).most_common(3)

    print(f"\n{category}")
    print(f"  Documents: {len(docs)}")
    print(f"  Avg length (chars): {avg_len:.0f}")
    print(f"  Top tokens: {top3}")
```

### Task 2: TF-IDF Baseline Pipeline

Build a `Pipeline` with `TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))` followed by `LogisticRegression(max_iter=1000)`. Train on the training split and evaluate on the test split. Print the full classification report.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pipe_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000,
                              stop_words='english',
                              ngram_range=(1, 2))),
    ('clf',   LogisticRegression(max_iter=1000, random_state=42)),
])

pipe_lr.fit(train.data, train.target)
y_pred = pipe_lr.predict(test.data)

print(classification_report(test.target, y_pred,
                             target_names=train.target_names))
```

### Task 3: Preprocessing Ablation

Compare three TF-IDF configurations paired with the same `LogisticRegression(max_iter=1000)`. Report accuracy for each configuration and identify which performs best on the test set.

| Configuration | Description |
|---------------|-------------|
| A | Default `TfidfVectorizer()` — no stopword removal, unigrams only |
| B | `TfidfVectorizer(stop_words='english')` |
| C | `TfidfVectorizer(stop_words='english', min_df=5)` |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

configs = {
    'A — no preprocessing':           TfidfVectorizer(),
    'B — stop_words':                 TfidfVectorizer(stop_words='english'),
    'C — stop_words + min_df=5':      TfidfVectorizer(stop_words='english',
                                                       min_df=5),
}

for name, vec in configs.items():
    pipe = Pipeline([('tfidf', vec),
                     ('clf',   LogisticRegression(max_iter=1000,
                                                  random_state=42))])
    pipe.fit(train.data, train.target)
    acc = pipe.score(test.data, test.target)
    print(f"{name}: {acc:.4f}")
```

Write 1–2 sentences explaining which configuration performed best and why you think that is.

### Task 4: Naive Bayes Comparison

Train a `Pipeline` with `TfidfVectorizer(stop_words='english')` and `MultinomialNB()`. Compare accuracy and per-class F1 with the Logistic Regression from Task 2. Fill in the comparison table with values from your classification reports.

```python
from sklearn.naive_bayes import MultinomialNB

pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf',   MultinomialNB()),
])

pipe_nb.fit(train.data, train.target)
y_pred_nb = pipe_nb.predict(test.data)

print(classification_report(test.target, y_pred_nb,
                             target_names=train.target_names))
```

Fill in the table with the values reported by `classification_report`:

| Model | Accuracy | rec.sport.hockey F1 | sci.med F1 | sci.space F1 | talk.politics.guns F1 |
|-------|----------|---------------------|-----------|-------------|----------------------|
| Logistic Regression (Task 2) | | | | | |
| Naive Bayes | | | | | |

### Task 5: Error Analysis

Using the best-performing model from Tasks 2–4, identify 3 misclassified documents from the test set. For each document, print the first 200 characters of text, the true label, and the predicted label. Write 2 sentences explaining why each document was difficult to classify correctly.

```python
best_pipe = pipe_lr   # or whichever model performed best

y_pred_best = best_pipe.predict(test.data)
misclassified_idx = [i for i in range(len(test.data))
                     if y_pred_best[i] != test.target[i]]

for idx in misclassified_idx[:3]:
    true_label = train.target_names[test.target[idx]]
    pred_label = train.target_names[y_pred_best[idx]]
    text_snippet = test.data[idx][:200].replace('\n', ' ')

    print(f"Index: {idx}")
    print(f"  True:      {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Text:      {text_snippet}")
    print()
```

### Task 6: Reflection

Answer each question in 2–4 sentences in your notebook:

1. What information does TF-IDF capture that raw term frequency (TF) alone does not? Why does this matter for distinguishing newsgroup categories?
2. In what scenario would you choose a transformer (for example, DistilBERT) over TF-IDF + Logistic Regression for a text classification task?
3. The `headers`, `footers`, and `quotes` were removed from the dataset using `remove=('headers', 'footers', 'quotes')`. Why is this important for a fair evaluation? What would happen to accuracy if you left them in?

---

### Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Exploratory text analysis | 10 | Per-category stats computed correctly; written description is meaningful and specific |
| Task 2: TF-IDF + LR pipeline | 20 | Pipeline built correctly with specified parameters; full classification report printed |
| Task 3: Preprocessing ablation | 15 | All three configurations tested; results compared; written explanation demonstrates understanding |
| Task 4: Naive Bayes comparison | 20 | Pipeline built correctly; comparison table filled in with values from classification reports |
| Task 5: Error analysis | 20 | 3 misclassified documents shown with text snippet, true label, and predicted label; explanations are specific to the text content |
| Task 6: Reflection | 15 | Answers demonstrate conceptual understanding of TF-IDF, transformer trade-offs, and data leakage |
| **Total** | **100** | |

---

## Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="TF (term frequency) measures how often a term appears in a document. IDF (inverse document frequency) penalizes terms that appear in many documents — common words like &quot;the&quot; appear in almost every document and carry little discriminative signal regardless of how often they appear in one document. The product TF &times; IDF gives high weight to terms that are frequent in a specific document but rare across the corpus. These are exactly the terms that distinguish one document from another. Raw TF alone cannot make this distinction: a word that appears 10 times in a document looks important whether it is a domain-specific term like &quot;lymphocyte&quot; or a ubiquitous word like &quot;said.&quot;">
  <div class="quiz-question">
    <strong>Question 1:</strong> What does TF-IDF do that raw term frequency alone cannot?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>TF-IDF removes all punctuation and stopwords automatically before computing any weights.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>TF-IDF downweights terms that appear frequently across many documents (like &quot;the&quot; or &quot;said&quot;), so that rare but distinctive terms receive higher weight, making them more useful for distinguishing one document from another.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>TF-IDF uses neural network embeddings to represent words as dense vectors that encode semantic similarity.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>TF-IDF limits the vocabulary to the most frequent N words, reducing noise from rare misspellings and typos.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="Stopword removal is a useful heuristic for topic classification but is not universally beneficial. For sentiment analysis, words like &quot;not,&quot; &quot;never,&quot; and &quot;barely&quot; appear on standard stopword lists but carry crucial negation signals — &quot;not bad&quot; expresses a positive sentiment that becomes &quot;bad&quot; once &quot;not&quot; is removed. The ablation study result (accuracy drops from 83% to 79%) is the empirical signal that stopword removal is harming this specific task, not helping it. For 20 Newsgroups topic classification, stopword removal typically improves performance because topical vocabulary dominates; for sentiment or negation-heavy tasks, it can hurt. This is why ablation experiments — systematically testing with and without each preprocessing step — are standard practice before committing to a preprocessing pipeline.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A student trains a text classifier, removes stopwords in preprocessing, and finds that test accuracy drops from 83% to 79%. What is the most likely explanation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Removing stopwords always improves accuracy on classification tasks; the drop must be caused by a bug in the preprocessing code.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Stopwords contain important semantic meaning for all NLP tasks, so they should never be removed regardless of task type.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Some stopwords carry signal in this specific domain or task — for example, function words like &quot;not&quot; or &quot;never&quot; can reverse sentiment, and removing them discards meaningful negation cues that the model relied on.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>TF-IDF breaks when stopwords are removed because it requires a full vocabulary including function words to compute IDF correctly.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="3" data-explanation="Model selection is never purely about maximizing accuracy. A simpler model that meets the performance requirement is often the better engineering choice: it is faster to train, easier to debug, cheaper to serve at inference time, and easier to update when data distribution shifts. The 4% accuracy gap (87% to 91%) translates to roughly 520 additional correct predictions per 13,000 test examples. Whether this gap justifies 90&times; longer training, GPU infrastructure costs, larger memory footprint, and more complex deployment depends on the business context. If each misclassification carries high cost (for example, routing a critical support ticket to the wrong team), 91% may be worth the complexity. For a moderate-stakes internal tool where 87% already meets the service-level requirement, TF-IDF + Logistic Regression is the pragmatic choice. Good engineering practice is to start with the simpler model and escalate only when the performance gap is demonstrably consequential.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You have 10,000 labeled customer support emails and want to build a topic classifier. TF-IDF + Logistic Regression achieves 87% accuracy in 30 seconds of training. Fine-tuning DistilBERT achieves 91% accuracy but requires 45 minutes of GPU training. Which should you choose?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Always choose the highest accuracy model regardless of training cost or infrastructure requirements, because accuracy is the only metric that matters in production.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Always choose the faster model to minimize infrastructure costs, because training time is the most important factor in model selection.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Retrain both models weekly on new data to avoid model drift, and serve whichever achieves higher accuracy that week.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>It depends on the use case — if 87% meets the business requirement and training cost or inference latency matters, TF-IDF + Logistic Regression is the better choice. If the 4% accuracy improvement significantly impacts business outcomes and GPU resources are available, DistilBERT may justify the added complexity.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>
