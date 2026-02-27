# Introduction to NLP

## Overview

Every lesson in this curriculum has worked with **structured** data — tables where each row is an observation and each column is a feature you can compute statistics on. Module 15 introduces a fundamentally different data type: **unstructured text**. Email inboxes, product reviews, medical notes, social media posts, customer service tickets, research papers — text is how humans record most of their knowledge, and NLP (Natural Language Processing) is how machines learn to read it. This module builds a complete NLP toolkit: from classical preprocessing and TF-IDF all the way to pre-trained Transformer models (BERT), connecting directly to module 16's sentiment analysis case study.

## Learning Objective

By the end of this lesson, you will have learned how to:

- Identify the core NLP task types and the typical pipeline for building text-based machine learning systems.
- Explain what makes text different from tabular data and why it requires specialized preprocessing.

## Key Terms

**Natural Language Processing (NLP):** A field at the intersection of linguistics, computer science, and machine learning concerned with enabling computers to understand, interpret, and generate human language.

**Corpus:** A collection of text documents used for training or evaluation. Plural: corpora.

**Token:** A unit of text — typically a word or subword — that a model processes. "Don't" might be tokenized as ["Don", "'", "t"] or ["Don't"] depending on the tokenizer.

**Vocabulary:** The set of all unique tokens in a corpus. Models with a fixed vocabulary cannot represent tokens outside it (out-of-vocabulary, OOV).

**Feature extraction:** Converting raw text into numerical representations that ML models can process. The core challenge of NLP — the same text can be represented many ways.

**Bag of Words (BoW):** A text representation that counts word occurrences in a document, ignoring order. "The cat sat on the mat" and "The mat sat on the cat" have identical BoW representations.

**TF-IDF (Term Frequency–Inverse Document Frequency):** A weighted word count that downweights words common across all documents (e.g., "the") and upweights words distinctive to specific documents.

**Transformer:** A neural architecture based on self-attention, introduced in 2017, that processes all words in a sequence simultaneously. The foundation of BERT, GPT, T5, and essentially every state-of-the-art NLP model.

**BERT (Bidirectional Encoder Representations from Transformers):** A pre-trained Transformer model (Google, 2018) that reads text in both directions simultaneously. Fine-tuning BERT on a specific task typically achieves state-of-the-art results.

**Hugging Face:** An open-source platform providing pre-trained Transformer models, datasets, and tools through the `transformers` Python library. The standard ecosystem for practical NLP.

## Starter Code

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/01_introduction-to-nlp_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

## What Is NLP?

You use NLP-powered systems every day without noticing:

- **Autocomplete** in your phone's keyboard predicts your next word
- **Spam filters** classify emails as spam or not spam
- **Google Search** understands the meaning of your query, not just its keywords
- **Voice assistants** (Siri, Alexa) transcribe speech and parse intent
- **Machine translation** (Google Translate) converts text between languages
- **GitHub Copilot** generates code from natural language descriptions

All of these systems convert text into numerical representations, then apply machine learning to those representations. The challenge is the conversion step — and it has become dramatically easier with pre-trained Transformer models.

## Core NLP Tasks

NLP covers a broad range of problems. This module focuses on the most practically common:

| Task | Description | Example |
|------|-------------|---------|
| **Text classification** | Assign a category label to a document | Spam/not-spam, topic classification, sentiment |
| **Sentiment analysis** | Determine the emotional tone of text | Positive/negative/neutral product reviews |
| **Named Entity Recognition (NER)** | Identify and classify named entities | "Apple acquired Beats for $3B" → ORG, ORG, MONEY |
| **Text summarization** | Generate a shorter version of a document | Summarize a 1,000-word article in 3 sentences |
| **Question answering** | Extract or generate an answer given a passage | "What year was BERT released?" → "2018" |
| **Machine translation** | Translate between languages | "The cat sat on the mat" → "Le chat s'assit sur le tapis" |
| **Text generation** | Generate coherent new text | GPT completing a sentence or writing code |
| **Zero-shot classification** | Classify into categories not seen during training | Label news articles with user-defined categories |

This module covers the first five tasks (text classification, sentiment, NER, summarization, zero-shot). Machine translation and open-ended text generation require large language models outside this module's scope.

## What Makes Text Hard for Machines

Numerical data is unambiguous: `age=35` means the same thing in every row. Text is deeply ambiguous:

**Lexical ambiguity:** The word "bank" means a financial institution, a riverbank, or the verb to tilt. The word "set" has over 400 meanings in the dictionary.

**Context dependency:** "I saw her duck" — did she duck (dodge), or did I see her duck (the bird)?

**Negation:** "The movie was not at all bad" is a positive sentiment. Naive word counting sees "bad" and may misclassify.

**Long-range dependencies:** In "The cat that sat on the red and green mat refused to move," the subject of "refused" is "cat" — 12 words earlier.

**Variable length:** Reviews range from one word ("Terrible.") to thousands of words (detailed technical reviews). Fixed-size ML models need a fixed-size input.

**Out-of-vocabulary words:** New slang, product names, and technical jargon appear constantly. Models trained on a fixed vocabulary fail on unseen words.

These challenges are why text preprocessing is not a simple normalization step — it involves fundamental design decisions about how to represent meaning numerically.

## The Evolution of NLP

```
1950s–1980s: Rule-based systems
   Linguists hand-wrote grammars and rules.
   Precise but brittle — one unhandled case breaks the system.

1990s–2000s: Statistical NLP
   Models learned probabilities from large text corpora.
   N-gram language models, hidden Markov models.
   More robust, but still required hand-crafted features.

2010s: Neural NLP
   Word2Vec (2013): words as dense vectors; "king − man + woman ≈ queen"
   LSTMs and GRUs for sequence modeling (module 14)
   End-to-end learning replaced most hand-crafted features.

2017–present: Transformer era
   "Attention Is All You Need" (Vaswani et al., 2017)
   BERT, GPT, T5, LLaMA — pre-train on massive text corpora,
   fine-tune on specific tasks.
   Achieves human-level performance on many NLP benchmarks.
```

Pre-training on billions of words allows Transformer models to learn grammar, world knowledge, and even reasoning — from text alone. This is why using a pre-trained BERT model (the approach in lesson 3) typically outperforms a carefully engineered pipeline trained from scratch.

## The NLP Pipeline

Despite the diversity of NLP tasks, the same pipeline structure recurs:

```
Raw text
   ↓
1. Data collection and labeling
   (reviews with star ratings, emails labeled spam/not-spam)
   ↓
2. Text preprocessing
   (cleaning, tokenization, normalization)
   ↓
3. Feature representation
   Classical: TF-IDF → sparse vector
   Modern:    BERT tokenizer → token IDs → contextual embeddings
   ↓
4. Model training
   Classical: Logistic regression, SVM, Naive Bayes
   Modern:    Fine-tune BERT or another pre-trained model
   ↓
5. Evaluation
   Accuracy, F1 score, precision/recall per class
   ↓
6. Deployment
   API endpoint, Hugging Face Hub, streaming inference
```

Lessons 2–4 each cover one or two stages of this pipeline in depth.

## Motivating Comparison: Classical vs. Transformer

To ground the module's scope, here is a preview of the performance gap between classical NLP (TF-IDF + logistic regression) and modern NLP (pre-trained BERT) on the same task:

```python
# 20 Newsgroups: 20-category news article classification
# ~18,000 articles, ranging from sports to religion to science

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset="all")
print(f"Documents: {len(data.data)},  Categories: {len(data.target_names)}")
print(f"Sample categories: {data.target_names[:5]}")
```

Output:
```
Documents: 18846,  Categories: 20
Sample categories: ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
```

In lesson 2 (Text Preprocessing), TF-IDF + logistic regression achieves **~86% accuracy** on this dataset with about 20 lines of code. In lesson 3 (Embeddings & Transformers), a pre-trained DistilBERT fine-tuned on the same data achieves **~91% accuracy** — a significant improvement on a hard 20-class problem. Understanding both approaches, and knowing when each is worth its complexity, is the goal of this module.

## Tools and Libraries

This module uses three NLP tool stacks:

**Classical NLP (lesson 2):**
```python
import nltk                           # Tokenization, stemming, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.linear_model import LogisticRegression          # Classifier
```

**Static Embeddings (lesson 3, part 1):**
```python
import gensim.downloader as api       # Pre-trained Word2Vec/GloVe vectors
```

**Transformers (lessons 3–4):**
```python
from transformers import pipeline     # High-level Hugging Face API
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

## Conclusion

NLP unlocks a category of problems that structured data cannot address: understanding customer feedback at scale, extracting information from unstructured documents, classifying support tickets, summarizing research literature. The challenge — converting variable-length, ambiguous, context-dependent text into numerical features — has been transformed by pre-trained Transformer models, which allow you to achieve state-of-the-art results on most tasks with very little labeled data. In the next lesson, you'll build the classical foundation: cleaning text, computing TF-IDF features, and training logistic regression classifiers — the approach that dominated NLP before 2018 and still works well as a fast, interpretable baseline.

## Practice

### Coding Assessment

Practice the concepts from this lesson using this [notebook](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/15-nlp/01_introduction-to-nlp_practice.ipynb). After completing the exercises, save your notebook to GitHub and submit the link for grading.

### Knowledge Check

<div class="quiz-container" data-correct="1" data-explanation="Routing tickets to departments is multi-class text classification — each ticket gets exactly one label from a fixed set of categories. TF-IDF captures which department-specific keywords appear in the ticket (billing keywords: &quot;invoice,&quot; &quot;charge,&quot; &quot;payment&quot;; technical keywords: &quot;error,&quot; &quot;crash,&quot; &quot;connection&quot;). Logistic regression then learns which keyword patterns map to each department. This approach is the right starting point: it&#039;s fast, interpretable (you can see which words drive each classification), and often good enough to deploy. A pre-trained BERT model adds complexity but typically provides meaningful accuracy improvement only if the baseline underperforms (&lt; 80%) or if subtle phrasing differences matter.">
  <div class="quiz-question">
    <strong>Question 1:</strong> A dataset contains 10,000 customer service tickets. You want to automatically classify each ticket into one of 15 department categories (billing, technical support, returns, etc.) to route it to the right team. Which NLP task is this, and what is the simplest approach to start with?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="0">
      <label>This is named entity recognition (NER). The simplest approach is to extract entity names from each ticket and route based on which entities appear.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="1">
      <label>This is text classification. The simplest approach is TF-IDF vectorization of the ticket text, followed by a logistic regression or linear SVM classifier trained on labeled examples. This baseline often achieves good results and is fast to build and interpret.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="2">
      <label>This is machine translation. You need to translate each ticket into a structured format that a rule-based routing system can parse.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-1" value="3">
      <label>This is text summarization. Summarize each ticket to a single sentence, then route based on keywords in the summary.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="This is the fundamental limitation of static representations (BoW, and also Word2Vec/GloVe). A word has one fixed vector regardless of context. Transformers address this through self-attention: computing &quot;bank&quot;&#039;s representation by attending to all other words in the sentence, weighted by relevance. The word &quot;deposited&quot; and &quot;money&quot; have high attention weights when computing &quot;bank&quot;&#039;s representation in the financial context; &quot;river&quot; and &quot;fished&quot; have high weights in the geographic context. The resulting contextual embedding vectors are literally different numerical vectors for the same word in different contexts.">
  <div class="quiz-question">
    <strong>Question 2:</strong> Why does the word "bank" pose a challenge for a Bag-of-Words model but not for a Transformer model?
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="0">
      <label>BoW models cannot process words with more than four letters, while Transformers have no character limit.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="1">
      <label>In a BoW model, "bank" is represented as a single count — the same number regardless of whether the sentence is "I deposited money at the bank" or "We fished along the river bank." The model has no way to distinguish which sense is intended. A Transformer model processes "bank" in context — attending to surrounding words ("deposited," "money" vs. "river," "fished") — and produces a different embedding vector for each sense. This contextual representation correctly captures the word's meaning in each context.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="2">
      <label>BoW models use a fixed vocabulary and cannot represent "bank" if it wasn't in the training corpus. Transformers use subword tokenization and can represent any word.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-2" value="3">
      <label>BoW models count "bank" as a single word, while Transformers split it into subword tokens "ban" + "k," capturing its etymology.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>


---

<div class="quiz-container" data-correct="1" data-explanation="This is the core argument for transfer learning in NLP. TF-IDF + logistic regression on 100 examples must learn everything from those 100 examples — including basic linguistic patterns that BERT has already encoded from billions of documents. BERT&#039;s representations know that &quot;excellent&quot; and &quot;outstanding&quot; are similar, that negation reverses sentiment, and that technical jargon clusters into domain groups. Fine-tuning with 100 examples adapts these rich representations to your specific classes. In practice, BERT fine-tuned on 100 examples often outperforms TF-IDF + LR trained on 1,000 examples for tasks where linguistic understanding matters (sentiment, intent, paraphrase detection).">
  <div class="quiz-question">
    <strong>Question 3:</strong> You have 100 labeled examples to train a text classifier. Should you train a TF-IDF model from scratch or fine-tune a pre-trained BERT model? Justify your choice.
  </div>
  <div class="quiz-options">
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="0">
      <label>Train TF-IDF from scratch — pre-trained models require at least 10,000 examples to be useful.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="1">
      <label>Fine-tune a pre-trained BERT model. With only 100 labeled examples, any model trained entirely from scratch (including TF-IDF + logistic regression) will likely have high variance because 100 examples are barely enough to learn 15+ class boundaries reliably. BERT has already learned rich linguistic representations from billions of words. Fine-tuning only the classification head on 100 examples adapts these representations to the specific task without needing the full dataset that training from scratch would require.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="2">
      <label>Train TF-IDF from scratch — BERT cannot work with fewer than 1,000 examples per class.</label>
    </label>
    <label class="quiz-option">
      <input type="radio" name="quiz-3" value="3">
      <label>Neither approach will work with 100 examples. You need to collect more data before building any model.</label>
    </label>
  </div>
  <button class="quiz-check-btn">Check Answer</button>
  <div class="quiz-feedback"></div>
</div>

