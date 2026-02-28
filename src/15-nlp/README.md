# NLP (Natural Language Processing)

This module introduces NLP as a discipline and builds a complete toolkit for practical text analysis — from classical bag-of-words preprocessing through state-of-the-art Transformer models. The module bridges module 14's deep learning foundations (RNNs, embeddings) to module 16's sentiment analysis case study, providing the preprocessing, representation, and workflow knowledge needed to tackle real text data.

## Lessons

1. **Introduction to NLP** — What NLP is and why text is different from tabular data (ambiguity, word order, variable length, OOV). Core task types (text classification, NER, summarization, QA, translation, zero-shot). The evolution from rule-based → statistical → neural → Transformer NLP. Motivating performance comparison (TF-IDF 85.6% vs. fine-tuned BERT 91%+ on 20 Newsgroups).
2. **Text Preprocessing** — Classical NLP pipeline: tokenization, normalization, stopword removal, stemming vs. lemmatization. Bag of Words with `CountVectorizer`, TF-IDF with `TfidfVectorizer`, unigrams and bigrams, sublinear TF. Full 20 Newsgroups classification (85.6% accuracy), preprocessing ablation study, logistic regression coefficient inspection.
3. **Word Embeddings & Transformer Models** — GloVe static embeddings and word arithmetic (king − man + woman ≈ queen); limitations of static representations. Self-attention intuition; BERT's bidirectional pre-training. Hugging Face `pipeline()` API, subword (WordPiece) tokenization, fine-tuning DistilBERT on a 4-class subset (94.2% vs. 90.9% TF-IDF), zero-shot classification.
4. **NLP Workflows** — End-to-end pipelines for four tasks: (1) text classification with confidence-threshold routing, (2) named entity recognition with spaCy and Hugging Face NER, (3) abstractive summarization with BART, (4) zero-shot classification. Production design considerations: batch processing, monitoring, hallucination risk in summarization. Bridge to module 16.
