# Case Study: Sentiment Analysis using Deep Learning

This case study applies the NLP preprocessing skills from module 15 and the deep learning architectures from module 14 to a practical sentiment classification problem. You will work through the full pipeline — from raw text and problem framing to a trained and evaluated deep learning model — and confront one of the most common questions in applied machine learning: *does the added complexity of deep learning actually help?*

This lesson encourages experimentation. Use the included [*Colaboratory notebook*](https://colab.research.google.com/github/gabrielsanchez/curriculum-ds/blob/main/notebooks/16-sentiment/01_sentiment-case-study_starter.ipynb) to run the code as you learn, or read the lesson and then freely experiment with the notebook's code.

Sentiment analysis is one of the most widely deployed NLP applications in industry: product review analysis, social media monitoring, and customer support triage all rely on some form of it. This case study makes that concrete — and reveals that the simplest approach is sometimes the best one.

## Lessons

1. **Introduction to the Case Study** — Problem framing, dataset description, loading the IMDB data, class balance, and initial impressions.
2. **Exploratory Data Analysis** — Review length distributions, vocabulary frequency analysis, class-conditional word comparison, and identifying what makes IMDB sentiment tractable.
3. **Text Preprocessing** — Vocabulary size selection, padding and truncation decisions, preparing uniform-length input matrices for a neural network.
4. **Model Building** — Training a bag-of-words baseline, an LSTM, and a bidirectional LSTM with dropout; comparing all three on accuracy and AUC; error analysis of misclassified reviews.
5. **Summary** — Synthesizing findings, reflecting on when deep learning earns its complexity, and considering what a production sentiment classifier needs beyond a good test score.
