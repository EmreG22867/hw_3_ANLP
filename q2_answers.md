# Homework 3 - Written Answers

## Part 1, Question 2

### Main run

```bash
python -m src.train --epochs 50 --margin 0.2 --seed 13
```

| Run | Split | Recall@1 | Recall@3 | MRR |
|---|---|---:|---:|---:|
| Before training | Test | 0.750 | 0.875 | 0.8281 |
| After training | Test | 0.875 | 1.000 | 0.9167 |

The objective computes cosine similarity between the query and the positive document and between the query and each negative document. For every query-negative pair, it applies "max(0, margin + s(q,d_neg) - s(q,d_pos))"; therefore, the loss becomes zero once the positive document scores at least the margin above a negative document, while violations move positive pairs closer and negative pairs more apart.

A successful query was “Which Luna has rabbit food and a yellow collar?”. The correct document D09 was ranked first with a score of 0.741, ahead of the Northside Luna document.

An uncertain query was “Where are Monday Data Science consultations held?” The correct document D13 was only ranked third with a score of 0.398, behind two irrelevant documents. This likely reflects the tiny dense model and limited training data: It has too few examples to learn stable associations for terms such as “Monday,” “Data Science,” and “consultations,” despite using all non-positive documents as negatives.

### Small experiment

The weighted bag-of-words model achieved Recall@1 = 1.0, Recall@3 = 1.0, and MRR = 1.0 on the test set both before and after training. The observation is that this tiny benchmark strongly rewards exact lexical overlap, so a sparse, interpretable bag-of-words representation is more reliable than the small randomly initialized dense mean-embedding model; training only changed the already-perfect ranking slightly.

