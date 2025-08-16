
# 📚 Amazon Hybrid Book Recommendation System

This project implements a **hybrid recommendation engine** for books using the [Amazon Books Reviews dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews).

It combines:

* **Content-Based Filtering** → NLP on summaries, categories, and metadata.
* **Collaborative Filtering** → user–item matrix + nearest neighbors.
* **Embedding-Based Retrieval** → fast approximate nearest neighbors with Annoy.

---

## 💡 About Hybrid Recommender Systems

Recommender systems usually fall into two categories:

* **Content-Based**: Recommend items similar to what the user liked, based on text/content.
* **Collaborative Filtering**: Recommend items based on user–item interactions.

⚡ This project **blends both approaches**.

* Users get recommendations that are **semantically similar (via NLP)**
* AND aligned with **community behavior (via collaborative filtering)**
* While **Annoy index** ensures scalable and fast lookup.

---

## 📁 Dataset Overview

We use the [`mohamedbakhet/amazon-books-reviews`](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) dataset:

* Over **10 million records** (≈1 GB+ uncompressed)
* Fields:

  * `User_id` – reviewer
  * `Title` – book title
  * `review/score` – numeric rating
  * `review/summary` – short review text
  * `categories`, `authors`, `description` – metadata

➡️ Due to space/memory constraints, we **sample 10%** of the dataset for experimentation.

---

## 🧪 Execution Modes

### ✅ 1. Sampled Dataset (Faster, Lightweight)

A preprocessed sample (`ratings2_processed.csv`) is included.
Run the script:

```bash
python hybrid_model.py
```

This will:

* Load the sample
* Clean/normalize titles, categories, authors
* Process summaries/descriptions with **NLTK + lemmatization**
* Build **TF-IDF + SVD embeddings**
* Train **collaborative models**
* Index books with **Annoy**
* Generate recommendations

---

### ⚙️ 2. Full Pipeline From Scratch

You can also run the full pipeline on the **original Kaggle dataset**.

Steps include:

* Download via `kagglehub`
* **Randomly sample 10%** of records
* Clean noisy titles with regex + fuzzy matching (RapidFuzz)
* Normalize metadata (`categories`, `authors`)
* Clean and lemmatize review summaries + compute sentiment (NLTK VADER)
* Vectorize with **TF-IDF + dimensionality reduction (SVD)**
* Build hybrid recommendation models

This mode is heavier but ensures full reproducibility.

---

## 🧹 Data Cleaning Highlights

* 📝 **Title Normalization** → lowercase, remove punctuation, normalize variants like *The Lord of the Rings*.
* 🔍 **Fuzzy Deduplication** → merge duplicate titles via RapidFuzz.
* 📂 **Metadata Cleaning** → authors/categories stripped & standardized.
* 🧾 **Text Preprocessing** → tokenization, stopword removal, lemmatization.
* 😊 **Sentiment Analysis** → NLTK VADER scores included as features.

---

## 🔍 Feature Engineering

* 📖 **TF-IDF on cleaned summaries & descriptions**
* 🔽 **Truncated SVD** → reduces dimensionality for scalability
* ⚡ **Annoy Index** → fast approximate nearest neighbors for book retrieval
* 😊 **Sentiment scores** used to enrich recommendation quality

---

## 🔄 Recommendations

### 📌 Content-Based

* Find semantically similar books (via TF-IDF + cosine similarity).

### 👥 Collaborative

* Use **user–item rating matrix** + **KNN** to recommend books based on similar users.

### ⚡ Hybrid (Final Model)

* Combines content similarity + collaborative scores.
* Annoy index speeds up recommendations across millions of items.

---

## 📊 Evaluation

* Precision\@k and Recall\@k for collaborative filtering.
* Qualitative case studies for hybrid recommendations.
* Fast inference benchmarks using Annoy.

---

## 📦 Dependencies

Install via pip:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn rapidfuzz annoy kagglehub
```

Download NLTK resources:

```python
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
```

---

## ▶️ Running the Project

### 1. With Sampled Dataset

```bash
python hybrid_model.py
```

### 2. With Full Dataset

```bash
python hybrid_model.py --full
```

This will:

* Download the Kaggle dataset
* Sample, clean, and preprocess
* Save a smaller processed file for reuse

---

## 📊 Sample Output

```bash
Top-5 similar books to "Harry Potter":
 1. Harry Potter and the Chamber of Secrets
 2. The Lord of the Rings
 3. Percy Jackson & the Olympians
 ...

Hybrid recommendations for user A12345XYZ:
 - The Hobbit (score: 0.92)
 - The Catcher in the Rye (score: 0.87)
 - Pride and Prejudice (score: 0.84)
```

---

## 💻 Author

**Reihane Montazeri** — Computer Engineer, ML/DL & Data Science Enthusiast
