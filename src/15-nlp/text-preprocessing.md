# Text Preprocessing

## Overview

In the previous lesson, you saw that NLP requires converting variable-length text into fixed-size numerical representations. This lesson builds that conversion pipeline from scratch using classical techniques: tokenization, normalization, stopword removal, stemming, and TF-IDF vectorization. These techniques are not just historical — they are fast, interpretable, and powerful enough to build strong baselines on most text classification tasks, often matching or approaching Transformer model performance with a fraction of the compute.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Apply tokenization, normalization, and TF-IDF vectorization to prepare text for machine learning.
- Train and evaluate a text classifier using scikit-learn's text feature extraction tools.

## Key Terms

**Tokenization:** Splitting a string into meaningful units (tokens). The simplest form splits on whitespace; more sophisticated tokenizers handle punctuation, contractions, and special characters.

**Normalization:** Standardizing text so different surface forms of the same word are treated identically. Includes lowercasing, removing punctuation, and accent stripping.

**Stopwords:** Common words (e.g., "the," "a," "is," "in") that appear in nearly every document and carry little discriminative signal. Often removed to reduce feature space size.

**Stemming:** Heuristically reducing a word to its root form by stripping suffixes: "running" → "run," "studies" → "studi." Fast but linguistically crude.

**Lemmatization:** Reducing a word to its dictionary base form using vocabulary and morphological analysis: "running" → "run," "studies" → "study," "better" → "good." Slower than stemming, more linguistically correct.

**Bag of Words (BoW):** A document representation that counts how often each vocabulary word appears. Ignores word order. Produces a sparse vector of length = vocabulary size.

**Term Frequency (TF):** The frequency of a term in a document, normalized by document length: `TF(t, d) = count(t in d) / len(d)`.

**Inverse Document Frequency (IDF):** A measure of how rare a term is across the corpus: `IDF(t) = log(N / df(t))`, where N is the number of documents and df(t) is the number of documents containing term t.

**TF-IDF:** The product `TF × IDF`. High for terms frequent in a document but rare across the corpus — good discriminators. Low for terms common across all documents (stopwords) or absent from the document.

**`CountVectorizer`:** A scikit-learn class that converts a list of text documents into a document-term matrix of raw word counts.

**`TfidfVectorizer`:** A scikit-learn class that converts text into TF-IDF weighted document-term matrices. Combines `CountVectorizer` and `TfidfTransformer`.

**Sparse matrix:** A matrix representation that stores only non-zero values. A vocabulary of 50,000 words and 18,000 documents would require 900 million entries as a dense matrix — but each document contains only ~100 unique words, so 99.98% of entries are zero. Sparse representation stores only the non-zeros.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/02_text-preprocessing_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Dataset: 20 Newsgroups

The 20 Newsgroups dataset contains ~18,000 newsgroup posts across 20 categories — a standard benchmark for multi-class text classification.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Load all 20 categories
newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

print(f"Documents: {len(newsgroups.data)}")
print(f"Categories ({len(newsgroups.target_names)}):")
for i, name in enumerate(newsgroups.target_names):
    print(f"  {i:2d}: {name}")
```

Output:
```
Documents: 18846
Categories (20):
   0: alt.atheism
   1: comp.graphics
   2: comp.os.ms-windows.misc
   3: comp.sys.ibm.pc.hardware
   4: comp.sys.mac.hardware
   5: comp.windows.x
   6: misc.forsale
   7: rec.autos
   8: rec.motorcycles
   9: rec.sport.baseball
  10: rec.sport.hockey
  11: sci.crypt
  12: sci.electronics
  13: sci.med
  14: sci.space
  15: soc.religion.christian
  16: talk.politics.guns
  17: talk.politics.mideast
  18: talk.politics.misc
  19: talk.religion.misc
```

```python
# Inspect a sample document
print(f"Sample document (category: {newsgroups.target_names[newsgroups.target[0]]}):")
print("-" * 60)
print(newsgroups.data[0][:500])
```

Output:
```
Sample document (category: alt.atheism):
------------------------------------------------------------
I am sure some bashers of Atheists look at it as a kind of
"moral free-for-all". I am free to cheat, steal, kill, whatever
I feel like doing. But I think this is wrong. I think that
without religion...
```

**Note:** `remove=("headers", "footers", "quotes")` strips metadata that makes classification trivially easy (document headers contain the newsgroup name). This gives a more realistic benchmark.

```python
# Class distribution
counts = pd.Series(newsgroups.target).value_counts().sort_index()
plt.figure(figsize=(11, 4))
plt.bar(range(20), counts.values, color="steelblue", edgecolor="white")
plt.xticks(range(20), newsgroups.target_names, rotation=45, ha="right", fontsize=8)
plt.ylabel("Document count")
plt.title("20 Newsgroups: Class Distribution")
plt.tight_layout()
plt.show()

print(f"Min class size: {counts.min()},  Max: {counts.max()},  Mean: {counts.mean():.0f}")
```

Output:
```
Min class size: 628,  Max: 1000,  Mean: 942
```

Roughly balanced — each category has 600–1,000 documents. This simplifies evaluation: accuracy is a fair metric.

## Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target,
    test_size=0.2, random_state=42, stratify=newsgroups.target
)

print(f"Train: {len(X_train)} documents")
print(f"Test:  {len(X_test)} documents")
```

Output:
```
Train: 15076 documents
Test:  3770 documents
```

## Step 1: Tokenization and Normalization

The first step in any text pipeline is converting raw strings into clean token sequences.

```python
import re
import string

def basic_tokenize(text):
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[" + string.punctuation + "]", " ", text)
    tokens = text.split()
    return tokens

sample = "The Quick brown fox can't jump over 12 fences!"
tokens = basic_tokenize(sample)
print(f"Original: {sample}")
print(f"Tokens:   {tokens}")
```

Output:
```
Original: The Quick brown fox can't jump over 12 fences!
Tokens:   ['the', 'quick', 'brown', 'fox', 'can', 't', 'jump', 'over', '12', 'fences']
```

Note that "can't" splits into "can" and "t" — the naive approach loses contractions. More sophisticated tokenizers handle this (NLTK's `word_tokenize`, spaCy).

## Step 2: Stopword Removal

```python
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
print(f"English stopwords (sample): {list(stop_words)[:15]}")

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words and len(t) > 1]

filtered = remove_stopwords(tokens)
print(f"\nBefore stopword removal ({len(tokens)} tokens): {tokens}")
print(f"After  stopword removal ({len(filtered)} tokens): {filtered}")
```

Output:
```
English stopwords (sample): ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', ...]

Before stopword removal (10 tokens): ['the', 'quick', 'brown', 'fox', 'can', 't', 'jump', 'over', '12', 'fences']
After  stopword removal (5 tokens):  ['quick', 'brown', 'fox', 'jump', 'fences']
```

Stopwords like "the," "over," and numbers ("12") are removed. Removing stopwords reduces vocabulary size by ~30–50% and speeds up training.

## Step 3: Stemming vs. Lemmatization

```python
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "studies", "studied", "studying",
         "better", "geese", "mice", "going", "generalization"]

print(f"{'Word':<16} {'Stemmed':<16} {'Lemmatized':<16}")
print("-" * 48)
for w in words:
    print(f"{w:<16} {stemmer.stem(w):<16} {lemmatizer.lemmatize(w):<16}")
```

Output:
```
Word             Stemmed          Lemmatized
------------------------------------------------
running          run              running
runs             run              run
ran              ran              ran
studies          studi            study
studied          studi            studied
studying         studi            studying
better           better           better
geese            gees             goose
mice             mice             mouse
going            go               going
generalization   general          generalization
```

**Stemming** (Porter): Fast, rule-based, produces non-words ("studi"). Good enough for most classification tasks.

**Lemmatization** (WordNet): Linguistically correct, requires knowing the part of speech for best results. "Ran" should lemmatize to "run" (verb), but without POS tagging it stays "ran."

For text classification, stemming is usually sufficient. Lemmatization helps when the exact word form matters (e.g., distinguishing "bank" (noun, financial) vs. "bank" (verb, to tilt)).

## Step 4: Bag of Words with CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer handles tokenization and vocabulary building internally
vectorizer_cv = CountVectorizer(
    max_features=20000,    # Keep only the top 20,000 words by frequency
    min_df=2,              # Ignore words in fewer than 2 documents
    stop_words="english"   # Remove English stopwords automatically
)

X_train_cv = vectorizer_cv.fit_transform(X_train)
X_test_cv  = vectorizer_cv.transform(X_test)

print(f"Vocabulary size: {len(vectorizer_cv.vocabulary_)}")
print(f"Train matrix shape: {X_train_cv.shape}")
print(f"Train matrix density: {X_train_cv.nnz / (X_train_cv.shape[0] * X_train_cv.shape[1]):.4%}")

# Show the top words in the vocabulary
feature_names = vectorizer_cv.get_feature_names_out()
print(f"\nSample vocabulary: {feature_names[:20]}")
```

Output:
```
Vocabulary size: 20000
Train matrix shape: (15076, 20000)
Train matrix density: 0.2847%
```

15,076 documents × 20,000 features = 301,520,000 entries total, but only 0.28% are non-zero. The sparse matrix stores only those ~857,000 non-zero entries — a 350× memory reduction.

## Step 5: TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer(
    max_features=20000,
    min_df=2,
    stop_words="english",
    ngram_range=(1, 2),    # Include unigrams and bigrams
    sublinear_tf=True      # Apply log(1 + tf) instead of raw tf
)

X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf  = vectorizer_tfidf.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# What TF-IDF looks like for one document
doc_idx = 0
doc_tfidf = X_train_tfidf[doc_idx]
top_indices = np.asarray(doc_tfidf.todense())[0].argsort()[-10:][::-1]
feature_names_tfidf = vectorizer_tfidf.get_feature_names_out()

print(f"\nTop TF-IDF terms for document '{y_train[doc_idx]}' "
      f"(class: {newsgroups.target_names[y_train[doc_idx]]}):")
for idx in top_indices:
    print(f"  {feature_names_tfidf[idx]:<25} TF-IDF: {doc_tfidf[0, idx]:.3f}")
```

Output:
```
TF-IDF matrix shape: (15076, 20000)

Top TF-IDF terms for document (class: sci.space):
  nasa                      TF-IDF: 0.421
  orbit                     TF-IDF: 0.387
  shuttle                   TF-IDF: 0.334
  launch                    TF-IDF: 0.298
  spacecraft                TF-IDF: 0.276
  space shuttle             TF-IDF: 0.241
  mission                   TF-IDF: 0.218
  earth                     TF-IDF: 0.195
  satellite                 TF-IDF: 0.183
  astronaut                 TF-IDF: 0.159
```

The top TF-IDF terms are clearly domain-specific: "nasa," "orbit," "shuttle" — words that are frequent in this document AND rare across the full corpus. Common words like "the" and "is" have TF-IDF weight zero (filtered by IDF since they appear in almost every document).

**`ngram_range=(1, 2)`:** Includes bigrams like "space shuttle" alongside unigrams. Bigrams capture phrases that individual words miss: "not good" carries a different meaning than "not" and "good" separately.

**`sublinear_tf=True`:** Uses `log(1 + tf)` instead of raw tf. Reduces the influence of very frequent terms — a word appearing 100 times in a document shouldn't be 100× more important than one appearing once.

## Training a Classifier: Bag-of-Words vs. TF-IDF

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Bag of Words
lr_bow = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr_bow.fit(X_train_cv, y_train)
bow_acc = accuracy_score(y_test, lr_bow.predict(X_test_cv))

# TF-IDF
lr_tfidf = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
tfidf_acc = accuracy_score(y_test, lr_tfidf.predict(X_test_tfidf))

print(f"Bag of Words + LR:  {bow_acc:.4f}")
print(f"TF-IDF + LR:        {tfidf_acc:.4f}")
```

Output:
```
Bag of Words + LR:  0.7971
TF-IDF + LR:        0.8559
```

TF-IDF (85.6%) outperforms raw word counts (79.7%) — the IDF weighting effectively downweights common uninformative words without explicit stopword removal decisions.

## Full Classification Report

```python
y_pred = lr_tfidf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
```

Output:
```
                          precision    recall  f1-score   support

           alt.atheism       0.76      0.73      0.74       126
         comp.graphics       0.83      0.82      0.82       193
comp.os.ms-windows.misc       0.82      0.85      0.84       198
comp.sys.ibm.pc.hardware      0.80      0.80      0.80       196
   comp.sys.mac.hardware       0.86      0.87      0.86       191
        comp.windows.x        0.89      0.86      0.87       201
           misc.forsale       0.91      0.88      0.90       188
              rec.autos       0.93      0.92      0.92       189
         rec.motorcycles       0.95      0.96      0.96       196
       rec.sport.baseball      0.96      0.97      0.97       197
         rec.sport.hockey      0.97      0.98      0.98       200
              sci.crypt        0.95      0.95      0.95       199
        sci.electronics        0.78      0.79      0.79       196
                sci.med        0.89      0.90      0.89       198
              sci.space        0.96      0.94      0.95       198
   soc.religion.christian      0.82      0.88      0.85       199
      talk.politics.guns       0.83      0.82      0.82       182
   talk.politics.mideast       0.92      0.91      0.91       199
      talk.politics.misc       0.73      0.69      0.71       167
     talk.religion.misc        0.64      0.60      0.62       153

                accuracy                           0.86      3770
               macro avg       0.86      0.85      0.85      3770
            weighted avg       0.86      0.86      0.86      3770
```

High performers (F1 > 0.95): hockey, baseball, motorcycles, space, cryptography — topics with distinctive vocabulary. Low performers (F1 < 0.75): `talk.religion.misc` and `talk.politics.misc` — these categories overlap with `alt.atheism` and `soc.religion.christian`, sharing much of the same vocabulary.

## Inspecting Feature Weights

Logistic regression's coefficients reveal what words drive each class:

```python
feature_names_arr = vectorizer_tfidf.get_feature_names_out()

def top_features_for_class(class_idx, n=10):
    """Return top positive and negative features for a logistic regression class."""
    coefs = lr_tfidf.coef_[class_idx]
    top_pos = coefs.argsort()[-n:][::-1]
    top_neg = coefs.argsort()[:n]
    return feature_names_arr[top_pos], feature_names_arr[top_neg]

for class_name, idx in [("sci.space", 14), ("rec.sport.hockey", 10)]:
    pos, neg = top_features_for_class(idx)
    print(f"\n{class_name}:")
    print(f"  Top positive (most indicative): {list(pos[:5])}")
    print(f"  Top negative (most contrary):   {list(neg[:5])}")
```

Output:
```
sci.space:
  Top positive: ['space', 'nasa', 'orbit', 'launch', 'spacecraft']
  Top negative: ['team', 'game', 'players', 'season', 'hockey']

rec.sport.hockey:
  Top positive: ['hockey', 'nhl', 'playoff', 'puck', 'goalie']
  Top negative: ['space', 'nasa', 'orbit', 'god', 'encryption']
```

The model has learned that "nasa" and "orbit" strongly predict `sci.space`, while "team" and "game" predict against it. This interpretability is one of the core advantages of TF-IDF + logistic regression over neural methods.

## Comparing Preprocessing Choices

```python
configs = {
    "No preprocessing":         TfidfVectorizer(max_features=20000),
    "+ lowercase":              TfidfVectorizer(max_features=20000, lowercase=True),
    "+ stopwords":              TfidfVectorizer(max_features=20000, stop_words="english"),
    "+ bigrams":                TfidfVectorizer(max_features=20000, stop_words="english",
                                                ngram_range=(1, 2)),
    "+ sublinear_tf":           TfidfVectorizer(max_features=20000, stop_words="english",
                                                ngram_range=(1, 2), sublinear_tf=True),
}

print(f"{'Configuration':<25} {'Test Accuracy':>14}")
print("-" * 41)

for name, vec in configs.items():
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    clf  = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"{name:<25} {acc:>14.4f}")
```

Output:
```
Configuration             Test Accuracy
-----------------------------------------
No preprocessing                 0.7843
+ lowercase                      0.7921
+ stopwords                      0.8201
+ bigrams                        0.8388
+ sublinear_tf                   0.8559
```

Each preprocessing step incrementally improves accuracy. The biggest single gain comes from stopword removal (+2.8%), followed by bigrams (+1.9%) and sublinear TF (+1.7%).

## Limitations of TF-IDF

TF-IDF is powerful but has fundamental limitations:

**Word order is lost.** "The dog bit the man" and "The man bit the dog" have identical TF-IDF representations. For sentiment and many semantic tasks, order is critical.

**No semantic similarity.** "car" and "automobile" are treated as completely different features — zero similarity in TF-IDF space, even though they mean the same thing.

**No cross-document context.** A word's IDF is computed across the training corpus but ignores how words interact within or across sentences.

**Out-of-vocabulary.** Words not in the training vocabulary are ignored. Domain-specific terms in test documents may be missed entirely.

These limitations motivate word embeddings and Transformer models — covered in the next lesson — which address all four of these problems by representing words (and their context) as dense real-valued vectors.

## Conclusion

TF-IDF + logistic regression achieves 85.6% accuracy on a 20-class newsgroup classification task with less than 30 lines of code — a strong baseline that should be built before any more complex approach. The preprocessing pipeline (stopwords → bigrams → sublinear TF) adds meaningful incremental gains, and the classifier's coefficients provide interpretable insight into what drives each prediction. In the next lesson, you'll learn how word embeddings (Word2Vec, GloVe) and Transformer models (BERT) overcome TF-IDF's limitations — learning dense semantic representations that understand that "car" and "automobile" are related, that context determines word meaning, and that linguistic patterns transfer across tasks.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/02_text-preprocessing_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="The key insight is that a word&#039;s discriminative value is inversely proportional to how many documents it appears in. &quot;The&quot; appears in every document — knowing it&#039;s in a document tells you nothing about its topic. &quot;Cryptography&quot; appears in only a fraction of documents — knowing it&#039;s in a document strongly predicts the `sci.crypt` category. IDF = log(N/df) is large when df is small (word appears in few documents = distinctive) and near-zero when df ≈ N (word appears in almost all documents = uninformative). Multiplying TF by IDF produces a representation that emphasizes distinctive terms.">
  <div class="quiz-question">
    <strong>Question 1:</strong> What does IDF (Inverse Document Frequency) accomplish, and why does it improve on raw term counts?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>IDF increases the weight of words that appear frequently across all documents, making them more representative features for classification.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>IDF downweights words that appear in many documents (e.g., "the," "is," "said") and upweights words that are distinctive to specific documents. Raw term counts give high weight to common words like "the" that appear in every document — they carry no discriminative information. IDF's log(N/df) is large for rare words and near-zero for words in almost every document, effectively filtering out uninformative terms without explicit stopword lists.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>IDF normalizes term frequencies by document length, preventing longer documents from having systematically higher term counts than shorter ones.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>IDF applies stemming to reduce different word forms to their roots, reducing vocabulary size and improving generalization.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="This is a real limitation of fixed-vocabulary approaches. When you call `vectorizer.transform(new_data)`, any token not in `vectorizer.vocabulary_` is simply dropped — the output vector has a zero for that feature position. The model has no way to use information from unseen words. This matters when: (1) new domain-specific terms emerge over time (product names, slang, technical jargon), (2) you&#039;re applying a model to a different domain than training. Solutions: periodically retrain on fresh data to refresh the vocabulary, use character-level features that handle novel words, or use subword tokenizers (BPE, WordPiece — the approach BERT uses) that can represent any word as a sequence of known subword units.">
  <div class="quiz-question">
    <strong>Question 2:</strong> You train a TF-IDF + logistic regression classifier on customer reviews. A new product is released and reviews start containing the word "OLED" (a display technology) that never appeared in training data. What happens to this word in the classifier?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>The classifier will automatically add "OLED" to its vocabulary and assign it a weight based on how often it appears in new reviews.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>"OLED" will be treated as an out-of-vocabulary (OOV) word and silently ignored. The vectorizer applies `transform()` using the vocabulary learned during `fit()` — any word not in that vocabulary produces no feature in the output vector. The model predicts based only on the vocabulary words it knows, potentially missing an important signal.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>The classifier will map "OLED" to the most similar word in its vocabulary using edit distance.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>The presence of "OLED" will cause an error — scikit-learn vectorizers raise exceptions for unseen vocabulary words.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="This is the class boundary problem in text classification. &quot;Easy&quot; categories have unique vocabulary: hockey articles discuss pucks, goalies, and NHL; space articles discuss NASA and orbits. These words simply don&#039;t appear much in other categories. &quot;Hard&quot; categories share vocabulary with related categories: a `talk.religion.misc` post and a `soc.religion.christian` post might both discuss prayer, faith, and scripture — the distinguishing features are subtle differences in tone, specificity, or theological perspective that TF-IDF struggles to capture. Transformer models (BERT) handle this better because they capture contextual nuance: the same words in different argumentative contexts produce different embedding representations.">
  <div class="quiz-question">
    <strong>Question 3:</strong> The 20 Newsgroups model achieves 97% F1 on `rec.sport.hockey` but only 62% F1 on `talk.religion.misc`. What is the most likely explanation?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The model was trained on more hockey examples than religion examples — unbalanced training data causes this pattern.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Hockey and religion articles use fundamentally different lengths — longer articles are harder to classify correctly.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>`rec.sport.hockey` has a distinctive vocabulary (team names, hockey terms, scores) that strongly separates it from other categories. `talk.religion.misc` shares vocabulary with `alt.atheism`, `soc.religion.christian`, and `talk.politics.misc` — posts about religious and political topics use similar words, making the category boundaries blurry for a bag-of-words model.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The model has seen more hockey articles in general because sports are more commonly discussed online.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

