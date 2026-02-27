# Word Embeddings & Transformer Models

## Overview

In the previous lesson, TF-IDF + logistic regression achieved 85.6% accuracy on 20 Newsgroups by treating words as independent frequency counts. Two fundamental limitations remain: "car" and "automobile" are different features despite being synonyms, and "bank" gets the same representation in "river bank" and "investment bank." This lesson covers the two generations of solutions — static word embeddings (Word2Vec, GloVe) that give each word a dense vector capturing semantic similarity, and Transformer models (BERT) that give each word a contextual embedding that changes based on surrounding text.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Explain the difference between static word embeddings and contextual Transformer embeddings.
- Use the Hugging Face `transformers` library to fine-tune a pre-trained model for text classification.

## Key Terms

**Word embedding:** A dense vector representation of a word, typically 50–300 dimensions. Words with similar meanings or usage patterns have similar vectors.

**Word2Vec:** A neural technique (Google, 2013) that learns word embeddings by training a model to predict a word from its context (CBOW) or predict context words from a word (Skip-gram). Produces static embeddings — one fixed vector per word regardless of context.

**GloVe (Global Vectors):** An embedding technique (Stanford, 2014) trained on word co-occurrence statistics across a large corpus. Similar to Word2Vec in quality; available as pre-trained vectors.

**Word analogy:** A demonstration of embedding geometry: `king − man + woman ≈ queen`. The arithmetic works because the embedding space encodes semantic relationships as directions.

**Contextual embedding:** A word representation that changes based on surrounding words. "Bank" in "river bank" and "investment bank" produces different vectors. Produced by Transformer models.

**Self-attention:** The core mechanism of Transformers. Each word attends to every other word in the sequence, computing a weighted average of their representations where weights reflect relevance. Allows direct modeling of long-range dependencies.

**BERT (Bidirectional Encoder Representations from Transformers):** A pre-trained Transformer (Google, 2018) that reads text bidirectionally. Pre-trained on masked language modeling (predict masked words) and next sentence prediction on 3.3 billion words.

**DistilBERT:** A smaller, faster version of BERT (66% of the parameters, 97% of the performance). Preferred when speed matters.

**Fine-tuning:** Continuing to train a pre-trained model on a specific downstream task with task-specific data. The pre-trained weights provide a warm start; only a few epochs of fine-tuning are needed.

**Hugging Face `transformers`:** An open-source Python library providing 100,000+ pre-trained models, tokenizers, and training utilities. The standard ecosystem for applied Transformer NLP.

**`pipeline()`:** A high-level Hugging Face API that wraps model loading, tokenization, inference, and output decoding into a single callable.

**Subword tokenization:** Tokenizing text into subword units rather than whole words. "unbelievable" → ["un", "##believ", "##able"]. BERT uses WordPiece tokenization, allowing representation of any word (including OOV) as a sequence of known subword pieces.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/04_embeddings-transformers_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Part 1: Static Word Embeddings

### Word2Vec Geometry

Word2Vec represents each word as a 300-dimensional vector. The vectors encode semantic and syntactic relationships as geometric directions:

```python
import gensim.downloader as api
import numpy as np

# Download pre-trained Word2Vec (Google News, 300d, 3M words)
# This is ~1.6 GB — use GloVe for a smaller alternative
# For demo: use the smaller GloVe Twitter vectors
wv = api.load("glove-wiki-gigaword-50")   # 50-dim GloVe, ~66MB

print(f"Vocabulary size: {len(wv)}")
print(f"Vector shape:    {wv['king'].shape}")

# Semantic similarity
pairs = [
    ("king", "queen"), ("man", "woman"),
    ("paris", "london"), ("dog", "cat"),
    ("king", "bicycle"), ("car", "automobile"),
]
print(f"\n{'Word pair':<25} {'Cosine similarity':>18}")
print("-" * 45)
for w1, w2 in pairs:
    sim = wv.similarity(w1, w2)
    print(f"  {w1} ↔ {w2:<20} {sim:>18.4f}")
```

Output:
```
Vocabulary size: 400000
Vector shape:    (50,)

Word pair                  Cosine similarity
---------------------------------------------
  king ↔ queen                        0.7839
  man ↔ woman                         0.8490
  paris ↔ london                      0.8371
  dog ↔ cat                           0.9218
  king ↔ bicycle                      0.1423
  car ↔ automobile                    0.7892
```

"car" and "automobile" have cosine similarity 0.789 — they are nearby in embedding space. "king" and "bicycle" are nearly orthogonal (unrelated). This semantic geometry is learned purely from word co-occurrence patterns in text.

### The Analogy Property

```python
# king - man + woman ≈ ?
result = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
print("king - man + woman ≈")
for word, score in result:
    print(f"  {word:<15} {score:.4f}")

print()

# paris - france + germany ≈ ?
result = wv.most_similar(positive=["paris", "germany"], negative=["france"], topn=5)
print("paris - france + germany ≈")
for word, score in result:
    print(f"  {word:<15} {score:.4f}")
```

Output:
```
king - man + woman ≈
  queen           0.8609
  princess        0.7552
  monarch         0.7029
  throne          0.6899
  elizabeth       0.6624

paris - france + germany ≈
  berlin          0.9412
  munich          0.8734
  frankfurt       0.8211
  hamburg         0.8098
  cologne         0.7861
```

The embedding space has learned that the direction from "man" to "woman" is the same direction as from "king" to "queen" — the gender axis. Similarly, the direction from a country to its capital is consistent across countries.

### Using GloVe Embeddings for Classification

For text classification, a simple approach averages the word vectors of all words in a document:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target,
    test_size=0.2, random_state=42, stratify=newsgroups.target
)

def doc_to_vector(text, wv, dim=50):
    """Average GloVe vectors for all words in the text."""
    tokens = text.lower().split()
    vectors = [wv[w] for w in tokens if w in wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

X_train_glove = np.array([doc_to_vector(doc, wv) for doc in X_train])
X_test_glove  = np.array([doc_to_vector(doc, wv) for doc in X_test])

lr_glove = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr_glove.fit(X_train_glove, y_train)
glove_acc = accuracy_score(y_test, lr_glove.predict(X_test_glove))
print(f"GloVe averaged embeddings + LR:  {glove_acc:.4f}")
print(f"TF-IDF + LR (from lesson 2):     0.8559")
```

Output:
```
GloVe averaged embeddings + LR:  0.7934
TF-IDF + LR (from lesson 2):     0.8559
```

Averaged GloVe embeddings (79.3%) underperform TF-IDF (85.6%) on this task. Averaging loses all word order and term frequency information — a "bag of vectors" suffers from the same word-order problem as BoW, while also losing TF-IDF's discriminative weighting. The advantage of embeddings emerges in tasks requiring semantic similarity and in neural architectures (LSTM, Transformer) that process the sequence rather than averaging it.

## Part 2: Transformer Models

### Why Static Embeddings Fall Short

```python
# "bank" has one static embedding regardless of context
bank_vec = wv["bank"]

sentences = [
    "I deposited money at the bank",
    "We sat by the river bank",
    "The plane began to bank sharply",
]

# Static embeddings: bank has ONE vector in all three contexts
print("Static embedding for 'bank' — same in all three sentences:")
print(f"  First 5 dimensions: {bank_vec[:5].round(3)}")
print("\nContextual embeddings (BERT) would produce DIFFERENT vectors for each.")
```

Static embeddings cannot distinguish word senses. This limitation is fundamental: a lookup table maps each word to one vector, period. Transformer models compute a new representation for each word in each context, attending to surrounding words.

### The Self-Attention Mechanism (Intuition)

```
Input: "The bank announced record profits"

Self-attention for the word "bank":
  "bank" attends to all other words simultaneously:
    "The"      → attention weight: 0.05  (function word, low relevance)
    "bank"     → attention weight: 0.30  (self)
    "announced"→ attention weight: 0.15
    "record"   → attention weight: 0.20
    "profits"  → attention weight: 0.30  (HIGH — financial context)

  Contextual embedding of "bank" = weighted average of all word embeddings
  where weights are the attention scores.

  Result: "bank" embedding is pulled toward "profits" and "announced" →
  the model learns this is a FINANCIAL bank, not a river bank.
```

By computing attention between every pair of words in parallel (not sequentially like RNNs), Transformers capture long-range dependencies without the vanishing gradient problem and train much faster on GPUs.

### The Hugging Face Ecosystem

```python
# Install: pip install transformers datasets
from transformers import pipeline

# The pipeline() function handles: loading the model, tokenization,
# forward pass, and decoding the output
```

Hugging Face's `pipeline()` is a three-line way to run pre-trained models:

```python
# Sentiment analysis with a pre-trained model
sentiment_pipe = pipeline("text-classification",
                          model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "This movie was absolutely fantastic! One of the best I've seen.",
    "Terrible experience. The food was cold and the service was rude.",
    "The product is okay. Nothing special but it works.",
    "I can't believe how bad this was. Total waste of money.",
]

results = sentiment_pipe(texts)
print(f"{'Text':<55} {'Label':<10} {'Score':>6}")
print("-" * 73)
for text, result in zip(texts, results):
    print(f"{text[:54]:<55} {result['label']:<10} {result['score']:>6.4f}")
```

Output:
```
Text                                                    Label      Score
-------------------------------------------------------------------------
This movie was absolutely fantastic! One of the best   POSITIVE   0.9997
Terrible experience. The food was cold and the servi   NEGATIVE   0.9995
The product is okay. Nothing special but it works.     POSITIVE   0.6201
I can't believe how bad this was. Total waste of mo    NEGATIVE   0.9992
```

Three lines of code; 99.9% confidence on the clear cases. "The product is okay" is correctly identified as mildly positive (62% confidence) — the model understands hedged sentiment.

### Subword Tokenization

BERT uses **WordPiece tokenization** — any word in any language can be represented as a sequence of known subword pieces, eliminating the OOV problem:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

examples = [
    "The cat sat on the mat.",
    "Supercalifragilisticexpialidocious",
    "COVID-19 vaccination rates are increasing.",
    "I can't believe it's already 2024!",
]

print(f"{'Input text':<45} {'Tokens'}")
print("-" * 90)
for text in examples:
    tokens = tokenizer.tokenize(text)
    print(f"{text:<45} {tokens}")
```

Output:
```
Input text                                    Tokens
------------------------------------------------------------------------------------------
The cat sat on the mat.                       ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']
Supercalifragilisticexpialidocious            ['super', '##cal', '##if', '##rag', '##il', '##istic', '##exp', '##ial', '##idoc', '##ious']
COVID-19 vaccination rates are increasing.    ['co', '##vid', '-', '19', 'vaccination', 'rates', 'are', 'increasing', '.']
I can't believe it's already 2024!            ['i', 'can', "'", 't', 'believe', 'it', "'", 's', 'already', '2024', '!']
```

"Supercalifragilisticexpialidocious" — not in the vocabulary — is split into known subword pieces (marked with `##`). "COVID-19" is split into known components. The model can always produce a representation, even for novel words.

### Fine-Tuning DistilBERT on 20 Newsgroups (Selected Classes)

Fine-tuning the full 20-class dataset requires significant compute. This example fine-tunes on 4 clearly separable categories for a compact demonstration:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np

# Use 4 categories for a faster demo
categories = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.misc"]
newsgroups_4 = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

X_tr4, X_te4, y_tr4, y_te4 = train_test_split(
    newsgroups_4.data, newsgroups_4.target,
    test_size=0.2, random_state=42, stratify=newsgroups_4.target
)

print(f"Train: {len(X_tr4)},  Test: {len(X_te4)}")
print(f"Categories: {newsgroups_4.target_names}")

# Tokenize
MODEL_NAME = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = Dataset.from_dict({"text": X_tr4, "label": y_tr4.tolist()})
test_dataset  = Dataset.from_dict({"text": X_te4, "label": y_te4.tolist()})

train_dataset = train_dataset.map(tokenize_batch, batched=True)
test_dataset  = test_dataset.map(tokenize_batch, batched=True)
```

```python
# Load pre-trained model with a classification head for 4 classes
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4
)

training_args = TrainingArguments(
    output_dir       = "./distilbert-newsgroups",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model  = "accuracy",
    logging_steps = 50,
    seed = 42,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics,
)

trainer.train()
```

Output (epoch 3):
```
{'eval_loss': 0.1143, 'eval_accuracy': 0.9418, 'epoch': 3.0}
```

```python
# Compare results
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec4 = TfidfVectorizer(max_features=20000, stop_words="english",
                       ngram_range=(1, 2), sublinear_tf=True)
X_tr4_tfidf = vec4.fit_transform(X_tr4)
X_te4_tfidf = vec4.transform(X_te4)
lr4 = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr4.fit(X_tr4_tfidf, y_tr4)
tfidf_acc_4 = accuracy_score(y_te4, lr4.predict(X_te4_tfidf))

print(f"4-class classification:")
print(f"  TF-IDF + LR:       {tfidf_acc_4:.4f}")
print(f"  DistilBERT (3 ep): 0.9418")
```

Output:
```
4-class classification:
  TF-IDF + LR:       0.9089
  DistilBERT (3 ep): 0.9418
```

DistilBERT (94.2%) outperforms TF-IDF (90.9%) with only 3 epochs of fine-tuning. On the full 20-class dataset, the gap is larger — BERT's contextual representations handle ambiguous category boundaries that TF-IDF's word-count features cannot.

## When to Use Each Approach

| Approach | Accuracy | Speed | Interpretability | When to use |
|----------|----------|-------|-----------------|-------------|
| TF-IDF + LR | Good | Very fast | High (coefficients) | Baseline, interpretability required, < 1M docs |
| GloVe averaged + LR | Lower than TF-IDF | Fast | Moderate | When semantic similarity matters for classification |
| LSTM + Embedding | Good | Moderate | Low | Sequential patterns matter; modest resources |
| Fine-tuned BERT | Excellent | Slow | Low | Maximum accuracy; enough compute |
| `pipeline()` (no fine-tuning) | Very good | Moderate | Low | Generic tasks (sentiment, NER) without labeled data |

## Zero-Shot Classification

Hugging Face also offers zero-shot classification — classify text into categories you define at inference time, without any task-specific fine-tuning:

```python
zero_shot = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

text = "NASA announced that the James Webb Space Telescope has captured \
        the deepest infrared image of the universe ever taken."

candidate_labels = ["science", "politics", "sports", "entertainment", "technology"]

result = zero_shot(text, candidate_labels=candidate_labels)

print(f"Text: {text[:80]}...")
print(f"\nClassification scores:")
for label, score in zip(result["labels"], result["scores"]):
    bar = "█" * int(score * 30)
    print(f"  {label:<15} {score:.4f}  {bar}")
```

Output:
```
Text: NASA announced that the James Webb Space Telescope has captured the deepest in...

Classification scores:
  science         0.6234  ██████████████████
  technology      0.2891  ████████
  politics        0.0521  █
  entertainment   0.0271
  sports          0.0083
```

No training data required. The model uses its pre-trained knowledge of language to estimate which categories the text matches. This is powerful for quick prototyping or tasks where labeled data is unavailable.

## Conclusion

Word embeddings (Word2Vec, GloVe) solved the synonym problem of TF-IDF by giving semantically similar words similar vector representations. Transformers (BERT, DistilBERT) solved the context problem by making word representations depend on surrounding text. Fine-tuned DistilBERT achieves 94.2% on the 4-class task vs. 90.9% for TF-IDF — a meaningful improvement that becomes even larger on tasks with subtler category boundaries. The Hugging Face ecosystem makes both pre-trained inference (via `pipeline()`) and fine-tuning accessible without deep framework expertise. In the next lesson, you'll put these tools together into complete NLP workflows for text classification, named entity recognition, and summarization — the pipeline structure that will carry directly into module 16's sentiment analysis case study.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/04_embeddings-transformers_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="This emergent structure is the most striking property of word embeddings. The model trained only to predict context words from target words, yet its internal representations captured gender, nationality, tense, and many other semantic axes as geometric directions. This happens because words used in similar contexts get similar vectors (the distributional hypothesis: &quot;you shall know a word by the company it keeps&quot;). &quot;King&quot; and &quot;queen&quot; appear in similar contexts except for gender-related contexts where they differ — exactly the gender direction. This geometric structure is why embeddings transfer well: the gender direction learned from English royalty generalizes to president/female-president, actor/actress, etc.">
  <div class="quiz-question">
    <strong>Question 1:</strong> Word2Vec represents "king − man + woman ≈ queen." What does this arithmetic demonstrate about the embedding space?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Word2Vec performs grammatical substitution: it replaces "king" with "queen" when "man" is changed to "woman."</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>The arithmetic demonstrates that semantic relationships are encoded as consistent directions (vectors) in the embedding space. The direction from "man" to "woman" (the gender direction) is approximately the same as the direction from "king" to "queen." By adding the gender direction to "king," we arrive near "queen." This geometric structure emerges from training on co-occurrence patterns — the model was never explicitly told that gender is a meaningful axis.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Word2Vec uses a mathematical formula where antonyms cancel and synonyms reinforce — the formula king − man + woman is a built-in rule for gendered nouns.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The arithmetic shows that Word2Vec stores all words as fractions of other words — "queen" is literally stored as "king − man + woman" in the vocabulary.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="WordPiece (BERT&#039;s tokenizer) learns a vocabulary of ~30,000 subword units from a large corpus. Frequent words appear as single tokens (&quot;the,&quot; &quot;running,&quot; &quot;president&quot;). Rare or novel words are decomposed: &quot;COVID-19&quot; → [&quot;co&quot;, &quot;##vid&quot;, &quot;-&quot;, &quot;19&quot;]; &quot;transformer&quot; → [&quot;transform&quot;, &quot;##er&quot;]. The `##` prefix marks continuation pieces. Because any text can be decomposed into these subword units, BERT never encounters a truly unknown word — it might not have seen &quot;COVID&quot; as a whole word in training, but it can still produce a representation from its subword components. This is why fine-tuning BERT on a medical corpus can handle medical terminology it wasn&#039;t explicitly trained on.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A pre-trained BERT tokenizer splits "unbelievable" into ["un", "##believ", "##able"]. Why does BERT use subword tokenization instead of whole-word tokenization?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Subword tokenization is used because BERT was trained on multiple languages, and subwords are language-neutral.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Subword tokenization allows BERT to represent any word — including novel technical terms, brand names, and foreign words — as a sequence of known subword pieces, eliminating the out-of-vocabulary problem. Whole-word tokenization with a vocabulary of 30,000 words would silently drop many domain-specific or newly coined words. Subword tokenization with 30,000 pieces can represent essentially any word in any language by composing known pieces.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Subword tokenization reduces memory usage because subword pieces are shorter than full words.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Subword tokenization allows BERT to understand prefixes and suffixes — "un-" always means negation, so the model can derive the meaning of unseen words from their components.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Zero-shot classification uses a natural language inference model (e.g., BART-MNLI) that was pre-trained to determine if a premise entails a hypothesis. At inference time, the &quot;premise&quot; is the review text and the &quot;hypothesis&quot; is each candidate label framed as a sentence (&quot;This review expresses very negative sentiment&quot;). The model scores which hypothesis is most entailed by the text. This requires no task-specific training data. The accuracy is lower than a fine-tuned model would achieve, but for quick prototyping or when labeling data is not yet possible, zero-shot classification is a practical starting point. If accuracy is insufficient, the next step would be labeling ~100 examples and fine-tuning a BERT model, which typically achieves strong results with that small amount of data.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You need to classify 1,000 product reviews into 5 sentiment categories (very negative, negative, neutral, positive, very positive) but have no labeled training data. What is the most appropriate approach?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Use TF-IDF + logistic regression — it does not require labeled training data because TF-IDF is unsupervised.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Use a zero-shot classification pipeline with a pre-trained model. Define the 5 categories as text labels ("very negative," "slightly negative," etc.) and let the model classify each review using its pre-trained understanding of sentiment language, without any task-specific training data.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Use a pre-trained BERT model fine-tuned on IMDB — since IMDB is also sentiment analysis, it transfers directly to product reviews.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Without labeled data, no machine learning approach is possible. You must manually label at least 200 reviews before building any classifier.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

