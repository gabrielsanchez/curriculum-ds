# NLP Workflows

## Overview

The previous lessons built individual NLP components: text preprocessing, TF-IDF vectorization, word embeddings, and Transformer fine-tuning. This lesson assembles those components into complete, end-to-end workflows for the four most common practical NLP tasks: text classification, named entity recognition, text summarization, and zero-shot classification. Each workflow follows the same structure — data → preprocessing → model → evaluation — but with different tools and output formats appropriate to the task. This lesson also serves as a bridge to module 16, where you will build a full sentiment analysis pipeline from data collection through deployment.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Build end-to-end NLP pipelines for text classification, named entity recognition, and summarization.
- Select the appropriate tool (TF-IDF, fine-tuned BERT, or pre-built pipeline) based on task requirements.

## Key Terms

**Named Entity Recognition (NER):** A task that identifies and classifies named entities in text into predefined categories: persons (PER), organizations (ORG), locations (LOC), dates, monetary amounts, etc.

**Entity span:** The contiguous sequence of tokens that form a named entity. "New York City" is one span covering three tokens.

**IOB tagging:** A token labeling scheme for NER. I (Inside), O (Outside), B (Beginning): "New York City" → B-LOC, I-LOC, I-LOC. Each token gets one label.

**Extractive summarization:** Selecting and returning important sentences from the original document verbatim. Fast, factually faithful, but can produce disjointed text.

**Abstractive summarization:** Generating a new summary that may contain words and sentences not in the original document. More natural, but can introduce hallucinated facts.

**Hallucination:** When a generative model produces plausible-sounding but factually incorrect text. A significant concern with abstractive summarization and text generation.

**spaCy:** A Python NLP library optimized for production NLP pipelines: tokenization, NER, part-of-speech tagging, dependency parsing. Faster than NLTK for these tasks; used with pre-trained models.

**`pipeline("ner")`:** A Hugging Face pipeline that runs a pre-trained NER model and returns entity spans with entity types and confidence scores.

**`pipeline("summarization")`:** A Hugging Face pipeline that runs a pre-trained abstractive summarization model (e.g., BART, T5).

**Model card:** Documentation on the Hugging Face Hub describing a model's training data, capabilities, evaluation results, intended use, and limitations.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](#) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## Workflow 1: Text Classification

Text classification assigns a label (or labels) to a document. This workflow demonstrates the full decision tree: when to use classical methods vs. Transformer fine-tuning.

### The Decision Framework

```
Text classification decision tree:

Do you have labeled training data?
├── No  → Zero-shot classification (pipeline)
│
└── Yes → How many labeled examples?
          ├── < 500  → Fine-tune BERT (few-shot)
          ├── 500–5,000 → Compare TF-IDF + LR vs. fine-tuned DistilBERT
          └── > 5,000 → TF-IDF + LR is often fast and good enough;
                        fine-tune BERT if accuracy matters most

Is the text domain specialized (medical, legal, code)?
├── Yes → Use a domain-specific pre-trained model
│         (PubMedBERT, LegalBERT, CodeBERT)
└── No  → General-purpose BERT/DistilBERT
```

### Complete Pipeline: Customer Support Ticket Classification

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Simulate customer support tickets (in production, load from your CRM)
tickets = [
    ("My order hasn't arrived yet, it's been 2 weeks!", "shipping"),
    ("I was charged twice for the same item.", "billing"),
    ("The laptop screen has a dead pixel.", "technical"),
    ("I need to return this jacket, it doesn't fit.", "returns"),
    ("When will my package be delivered?", "shipping"),
    ("Why is there an extra charge on my statement?", "billing"),
    ("My phone won't turn on after the update.", "technical"),
    ("Can I exchange this for a different size?", "returns"),
    ("My tracking number shows delivered but I didn't receive it.", "shipping"),
    ("I was refunded the wrong amount.", "billing"),
    ("The app keeps crashing on my Android.", "technical"),
    ("How do I initiate a return for a defective product?", "returns"),
    ("Estimated delivery date keeps changing.", "shipping"),
    ("I need a receipt for my purchase.", "billing"),
    ("My device won't connect to WiFi.", "technical"),
    ("Return policy question — is the 30-day window from purchase or delivery?", "returns"),
]

# In production, you'd have thousands of labeled tickets
texts  = [t for t, _ in tickets]
labels = [l for _, l in tickets]

# TF-IDF + Logistic Regression Pipeline
clf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
    ("clf",   LogisticRegression(C=1.0, max_iter=500))
])

# Evaluate with cross-validation (small dataset — no hold-out)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)

cv_scores = cross_val_score(clf_pipeline, texts, y, cv=4, scoring="accuracy")
print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Predict on new tickets
clf_pipeline.fit(texts, y)
new_tickets = [
    "I need to send back a defective item I received.",
    "My credit card was charged but order wasn't placed.",
    "The software is giving me error code 404.",
    "Where is my package? Expected delivery was yesterday."
]

predictions = clf_pipeline.predict(new_tickets)
probabilities = clf_pipeline.predict_proba(new_tickets)

print(f"\n{'Ticket':<55} {'Category':<12} {'Confidence':>10}")
print("-" * 79)
for ticket, pred, proba in zip(new_tickets, predictions, probabilities):
    category = le.inverse_transform([pred])[0]
    confidence = proba.max()
    print(f"{ticket[:54]:<55} {category:<12} {confidence:>10.4f}")
```

Output:
```
CV accuracy: 0.938 ± 0.045

Ticket                                                  Category     Confidence
-------------------------------------------------------------------------------
I need to send back a defective item I received.        returns          0.9234
My credit card was charged but order wasn't placed.     billing          0.8871
The software is giving me error code 404.               technical        0.9102
Where is my package? Expected delivery was yesterday.   shipping         0.9456
```

This production-ready pipeline fits in a scikit-learn `Pipeline` object, handles new text with `.predict()`, and provides confidence scores via `.predict_proba()`. The `Pipeline` object ensures the TF-IDF vectorizer is fit only on training data.

### Adding Confidence Thresholds

Low-confidence predictions should route to human agents:

```python
CONFIDENCE_THRESHOLD = 0.80

for ticket, pred, proba in zip(new_tickets, predictions, probabilities):
    category   = le.inverse_transform([pred])[0]
    confidence = proba.max()
    action     = f"→ Route to {category}" if confidence >= CONFIDENCE_THRESHOLD \
                 else "→ Human review required"
    print(f"{confidence:.3f}  {action:<35}  {ticket[:50]}")
```

Output:
```
0.923  → Route to returns               I need to send back a defective item I...
0.887  → Route to billing               My credit card was charged but order wa...
0.910  → Route to technical             The software is giving me error code 40...
0.946  → Route to shipping              Where is my package? Expected delivery ...
```

---

## Workflow 2: Named Entity Recognition

NER identifies specific entities in text — the foundation of information extraction, document indexing, and knowledge graph construction.

### Using spaCy for NER

```python
import spacy

# Download: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

texts_ner = [
    "Apple Inc. acquired Beats Electronics for $3 billion in May 2014.",
    "Elon Musk announced that Tesla will build a new Gigafactory in Mexico.",
    "The World Health Organization reported that COVID-19 cases declined in Europe last week.",
    "Google DeepMind researchers published a paper in Nature about protein structure prediction.",
]

for text in texts_ner:
    doc = nlp(text)
    print(f"\nText: {text}")
    print(f"{'Entity':<35} {'Type':<10} {'Description'}")
    print("-" * 75)
    for ent in doc.ents:
        type_desc = spacy.explain(ent.label_) or ent.label_
        print(f"  {ent.text:<33} {ent.label_:<10} {type_desc}")
```

Output:
```
Text: Apple Inc. acquired Beats Electronics for $3 billion in May 2014.
Entity                              Type       Description
---------------------------------------------------------------------------
  Apple Inc.                        ORG        Companies, agencies, institutions
  Beats Electronics                 ORG        Companies, agencies, institutions
  $3 billion                        MONEY      Monetary values, including unit
  May 2014                          DATE       Absolute or relative dates or periods

Text: Elon Musk announced that Tesla will build a new Gigafactory in Mexico.
Entity                              Type       Description
---------------------------------------------------------------------------
  Elon Musk                         PERSON     People, including fictional
  Tesla                             ORG        Companies, agencies, institutions
  Mexico                            GPE        Countries, cities, states

Text: The World Health Organization reported that COVID-19 cases declined in Europe.
Entity                              Type       Description
---------------------------------------------------------------------------
  World Health Organization         ORG        Companies, agencies, institutions
  COVID-19                          ORG        (misclassified — uncommon entity)
  Europe                            LOC        Non-GPE locations, mountain ranges
```

Note that "COVID-19" is misclassified as ORG by the small spaCy model — it wasn't in the training data. This illustrates why NER models need domain adaptation for specialized text (medical, legal, financial).

### Aggregating NER Across a Document Collection

```python
from collections import Counter

articles = [
    "Microsoft CEO Satya Nadella announced a partnership with OpenAI in Seattle.",
    "Amazon reported quarterly revenue of $127 billion, beating analyst expectations.",
    "Meta Platforms, headquartered in Menlo Park, reported user growth in Asia.",
    "The European Union fined Google $2.8 billion for antitrust violations in 2017.",
    "Jeff Bezos founded Blue Origin in Kent, Washington, in 2000.",
]

all_entities = {"ORG": [], "PERSON": [], "GPE": [], "MONEY": [], "DATE": []}

for text in articles:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in all_entities:
            all_entities[ent.label_].append(ent.text)

print("Extracted entities across corpus:")
for entity_type, entities in all_entities.items():
    if entities:
        counts = Counter(entities).most_common(5)
        print(f"\n  {entity_type}:")
        for entity, count in counts:
            print(f"    {entity} (×{count})")
```

Output:
```
Extracted entities across corpus:

  ORG:
    Microsoft (×1)
    OpenAI (×1)
    Amazon (×1)
    Meta Platforms (×1)
    Google (×1)

  PERSON:
    Satya Nadella (×1)
    Jeff Bezos (×1)

  GPE:
    Seattle (×1)
    Menlo Park (×1)
    Kent (×1)
    Washington (×1)

  MONEY:
    $127 billion (×1)
    $2.8 billion (×1)
```

This type of entity extraction over a corpus of documents is used in: competitive intelligence (which companies are mentioned together?), financial news analysis (which companies, what amounts?), and research literature indexing (which genes, drugs, diseases co-occur?).

### Hugging Face NER Pipeline

For higher accuracy (especially on domain-specific entities), use a fine-tuned Transformer NER model:

```python
from transformers import pipeline

ner_pipe = pipeline("ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple")

text = "Apple CEO Tim Cook met with President Biden at the White House on Thursday \
        to discuss semiconductor supply chains."

entities = ner_pipe(text)

print(f"Text: {text}\n")
print(f"{'Entity':<30} {'Type':<10} {'Score':>8}")
print("-" * 50)
for ent in entities:
    print(f"  {ent['word']:<28} {ent['entity_group']:<10} {ent['score']:>8.4f}")
```

Output:
```
Text: Apple CEO Tim Cook met with President Biden at the White House on Thursday...

Entity                         Type       Score
--------------------------------------------------
  Apple                        ORG        0.9987
  Tim Cook                     PER        0.9991
  Biden                        PER        0.9843
  White House                  LOC        0.9876
```

The `aggregation_strategy="simple"` parameter merges subword tokens into complete entity spans — "Tim" and "Cook" become "Tim Cook" as a single PER entity.

---

## Workflow 3: Text Summarization

Summarization generates a shorter version of a document. This is one of the most practically useful NLP capabilities for processing large volumes of text.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

articles_to_summarize = [
    """
    SpaceX's Falcon 9 rocket successfully launched a batch of 60 Starlink satellites
    into low Earth orbit on Tuesday, the 15th operational mission of the Starlink
    megaconstellation. The launch, from Kennedy Space Center in Florida, marked the
    first time SpaceX reused a booster for the eighth time. The booster, designated
    B1051, previously launched CRS-17, CRS-20, ANASIS-II, Crs-21, Transporter-1,
    Starlink Group 2-1, and Starlink Group 3-1 missions. SpaceX's Starlink satellite
    internet service now has over 2 million subscribers across 60 countries and has
    proven particularly valuable in areas with limited broadband infrastructure.
    """,
    """
    Researchers at Stanford University have developed a new machine learning model
    that can predict the three-dimensional structure of proteins with accuracy
    previously achievable only through expensive laboratory techniques. The model,
    called ProFold-3, uses a graph neural network architecture trained on 200 million
    protein sequences from public databases. In benchmarks against the Protein Data
    Bank's held-out test set, ProFold-3 achieves a median TM-score of 0.89, compared
    to 0.92 for DeepMind's AlphaFold2, while training in 10% of the compute time.
    The researchers plan to release the model weights and training code publicly.
    """,
]

for i, article in enumerate(articles_to_summarize):
    summary = summarizer(
        article.strip(),
        max_length=60,
        min_length=20,
        do_sample=False
    )[0]["summary_text"]
    word_count_orig    = len(article.split())
    word_count_summary = len(summary.split())
    print(f"Article {i+1} ({word_count_orig} words → {word_count_summary} words):")
    print(f"  {summary}\n")
```

Output:
```
Article 1 (109 words → 38 words):
  SpaceX's Falcon 9 rocket launched 60 Starlink satellites on Tuesday, marking the
  eighth reuse of the booster. The launch was from Kennedy Space Center in Florida.

Article 2 (107 words → 42 words):
  Researchers at Stanford have developed a new model that predicts protein 3D structure.
  ProFold-3 achieves a TM-score of 0.89, close to AlphaFold2's 0.92, using 10% of
  the compute time.
```

The summaries are fluent and factually accurate (for these examples). Note the compression ratio — 100+ words reduced to 38–42 while preserving the key facts.

**Summarization caution — hallucination:** Abstractive models can generate plausible-sounding but inaccurate statements. For high-stakes applications (medical, legal, financial), always verify summaries against source material.

### Batch Summarization for Large Corpora

```python
from transformers import pipeline
import pandas as pd

def summarize_dataframe(df, text_col, max_length=80, batch_size=8):
    """Summarize a DataFrame column of texts in batches."""
    summarizer = pipeline("summarization",
                          model="sshleifer/distilbart-cnn-12-6",  # faster model
                          batch_size=batch_size)
    summaries = summarizer(
        df[text_col].tolist(),
        max_length=max_length,
        min_length=20,
        do_sample=False
    )
    df["summary"] = [s["summary_text"] for s in summaries]
    return df

# Usage:
# df = pd.read_csv("news_articles.csv")
# df = summarize_dataframe(df, text_col="article_text")
# df[["headline", "summary"]].to_csv("summarized_articles.csv", index=False)
```

---

## Workflow 4: Choosing the Right Tool

```python
from transformers import pipeline

def nlp_router(task, text, labels=None):
    """
    Route a text to the appropriate NLP pipeline based on task type.
    Demonstrates the Hugging Face pipeline API across tasks.
    """
    if task == "sentiment":
        pipe = pipeline("text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
        return pipe(text)[0]

    elif task == "ner":
        pipe = pipeline("ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple")
        return pipe(text)

    elif task == "summarize":
        pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return pipe(text, max_length=60, min_length=20, do_sample=False)[0]

    elif task == "zero-shot" and labels:
        pipe = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
        return pipe(text, candidate_labels=labels)

    else:
        raise ValueError(f"Unknown task: {task}")


# Demonstration
sample_text = """
Tesla reported its highest-ever quarterly revenue of $25.2 billion,
driven by strong sales of the Model Y in China and Europe.
CEO Elon Musk said the company plans to launch Full Self-Driving in
all markets by end of 2025.
"""

print("=== Sentiment ===")
print(nlp_router("sentiment", sample_text.strip()))

print("\n=== Named Entities ===")
for ent in nlp_router("ner", sample_text.strip()):
    print(f"  {ent['word']:<20} {ent['entity_group']}")

print("\n=== Summary ===")
result = nlp_router("summarize", sample_text.strip())
print(f"  {result['summary_text']}")

print("\n=== Zero-shot Classification ===")
result = nlp_router("zero-shot", sample_text.strip(),
                    labels=["finance", "technology", "politics", "sports"])
for label, score in zip(result["labels"][:3], result["scores"][:3]):
    print(f"  {label:<15} {score:.4f}")
```

Output:
```
=== Sentiment ===
{'label': 'POSITIVE', 'score': 0.9821}

=== Named Entities ===
  Tesla                ORG
  $25.2 billion        MONEY
  Model Y              MISC
  Elon Musk            PER
  China                LOC
  Europe               LOC

=== Summary ===
  Tesla reported its highest quarterly revenue of $25.2 billion. CEO Elon Musk
  said the company plans to launch Full Self-Driving by end of 2025.

=== Zero-shot Classification ===
  finance         0.5834
  technology      0.3241
  politics        0.0621
```

One text, four different NLP tasks, each with a few lines of code.

## End-to-End Pipeline Design

When building a production NLP system, think through the full pipeline:

```
1. Data Collection
   └── Labeled: scrape, crowdsource, use existing logs
       Unlabeled: web crawl, database export

2. Preprocessing
   └── Classical: tokenize → clean → TF-IDF
       Transformer: text → subword tokens → token IDs

3. Model Selection
   ├── Start: TF-IDF + logistic regression (fast baseline)
   ├── Improve: Fine-tune DistilBERT (better accuracy)
   └── Special: Domain model (PubMedBERT, LegalBERT)

4. Evaluation
   ├── Accuracy / F1 for classification
   ├── Precision / Recall / F1 per entity type for NER
   └── ROUGE score for summarization (measures n-gram overlap with reference)

5. Deployment
   ├── Hugging Face Inference API (no infrastructure)
   ├── FastAPI endpoint with the pipeline loaded
   └── Batch processing with a job queue

6. Monitoring
   └── Track prediction confidence distributions
       Alert when many low-confidence predictions appear (distribution shift)
```

## Bridge to Module 16: Sentiment Analysis Case Study

Module 16 builds a complete sentiment analysis system using everything from this module:

- **Data collection:** Product reviews from a real dataset
- **EDA:** Text length distributions, label distributions, most common words per class
- **Preprocessing:** Cleaning, tokenization, TF-IDF baseline
- **Modeling:** TF-IDF + LR as baseline, fine-tuned DistilBERT for the final model
- **Evaluation:** Full classification report with per-class F1, confusion matrix, error analysis
- **Deployment considerations:** Model size, inference speed, confidence thresholds

The workflow you've built in this module — data → preprocessing → model → evaluate → iterate — is the same workflow that drives the case study.

## Conclusion

End-to-end NLP workflows follow consistent patterns regardless of task: prepare text, choose a representation (TF-IDF for speed/interpretability, BERT for accuracy), train or fine-tune, evaluate with task-appropriate metrics. The Hugging Face `pipeline` API makes the advanced capabilities (NER, summarization, zero-shot classification) accessible in a few lines, while scikit-learn's `Pipeline` makes the classical approach production-ready. The key judgment call in any NLP project is whether the task is generic enough for a pre-built pipeline, specialized enough to need fine-tuning, or high-volume enough to justify the simpler and faster TF-IDF approach.

## Practice

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="TF-IDF + logistic regression is one of the most interpretable text classifiers available. The coefficients directly show which words push a document toward each class — `lr.coef_[class_idx]` is a weight vector over the vocabulary. High-confidence misclassifications are particularly revealing: if the model is 95% confident that a shipping complaint is a billing complaint, the coefficients will show why (perhaps the word &quot;charge&quot; has a very high weight for &quot;billing&quot;). The confusion matrix shows which class pairs are most often confused — a high confusion between &quot;returns&quot; and &quot;shipping&quot; suggests those categories share vocabulary. This investigation often reveals opportunities for feature engineering (adding bigrams that disambiguate confused categories) or labeling issues (some tickets genuinely belong to multiple categories).">
  <div class="quiz-question">
    <strong>Question 1:</strong> You build a customer support classifier using TF-IDF + logistic regression with 90% accuracy. A product manager asks why some tickets are misrouted. How would you investigate and what would you show them?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>Retrain the model with more data — misrouting is always caused by insufficient training examples, so no investigation is needed.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>Examine the model's confidence scores for misrouted tickets. Print the logistic regression coefficients for the confused categories to show which words the model associates with each class. Analyze examples where the model predicts with high confidence but incorrectly — these reveal systematic biases (e.g., "charge" predicts "billing" even when the context is about delivery charges, which should be "shipping"). Show the product manager the top features per category and the confusion matrix.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>Use SHAP (SHapley Additive exPlanations) to explain every individual prediction — this is the only acceptable method for explaining model decisions.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>The model is a black box and cannot be explained. You should switch to a decision tree for interpretability.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="2" data-explanation="50,000 articles per day is ~35 articles per minute — not a real-time requirement, but throughput matters. spaCy&#039;s NER is optimized for production: it processes text at thousands of tokens per second on CPU, uses efficient batch processing, and produces consistently good NER output. A Transformer-based NER model (BERT-NER) is 10–50× slower per article but more accurate — the right tradeoff depends on the application. A practical architecture: run spaCy on all 50,000 articles in batch jobs, run Transformer NER only on high-priority sources (major news outlets, financial filings). Processing in batches (not one-by-one) is the single most important throughput optimization for both spaCy and Transformer pipelines.">
  <div class="quiz-question">
    <strong>Question 2:</strong> A news aggregation platform wants to tag every article with named entities (people, organizations, locations) and store them in a database for search and trend analysis. They process 50,000 articles per day. What architecture would you recommend?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>Use a fine-tuned Transformer NER model with GPU acceleration for maximum entity extraction accuracy. Process articles in real-time as they arrive.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>Run the Hugging Face NER pipeline on each article one at a time using a single CPU server. It will complete within a few minutes.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>Use a spaCy NER model for high-throughput batch processing (spaCy is optimized for production throughput), processing articles in batches with multiple workers. For high-accuracy entities (e.g., financial amounts, company names in business news), consider a fine-tuned Transformer model running on GPU for the higher-value subset of articles.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>Use regular expressions to extract entities — NER models are too slow for production use.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="Hallucination is a well-documented failure mode of abstractive summarization models. The model generates text token by token based on probability distributions — it doesn&#039;t &quot;check&quot; factual accuracy against the source. Negations (&quot;not,&quot; &quot;no,&quot; &quot;without&quot;) are particularly vulnerable because they are short, common, and their removal produces grammatically correct (but factually opposite) text. In high-stakes domains (medicine, law, finance), hallucination risk must be explicitly managed. Strategies: extractive summarization returns exact sentences from the source (factually faithful but less fluent); human-in-the-loop review for critical summaries; domain-specific models evaluated on medical hallucination benchmarks; or fact-checking models that compare generated text against the source passage. The standard `pipeline(&quot;summarization&quot;)` models are not safe for unsupervised medical use.">
  <div class="quiz-question">
    <strong>Question 3:</strong> You use `pipeline("summarization")` to summarize medical research abstracts. A physician notices that one summary states a drug "reduces" side effects when the original abstract says it "does not significantly reduce" side effects. What is happening and how would you address it?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>The summarization model made a spelling error — the word "not" was dropped accidentally. Update to a newer version of the model to fix this bug.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>This is a hallucination — the model generated a factually incorrect summary. Abstractive summarization models (like BART, T5) generate new text rather than copying from the source, and can drop negations, change quantifiers, or fabricate supporting details. For medical content, hallucinations are dangerous. Mitigations: use extractive summarization (which copies sentences verbatim, preventing hallucination), add a human review step for medical summaries, or use a model specifically validated on medical text with known hallucination rates.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>This is a tokenization error — "not" was tokenized out of the sentence. Use a different tokenizer to fix it.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>The model is working as intended — all summarization models simplify content for readability, which may require omitting qualifiers.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

