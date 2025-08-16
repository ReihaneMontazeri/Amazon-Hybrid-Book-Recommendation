#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Recommender for Amazon Books Reviews
===========================================

Modular end-to-end pipeline for large-scale book recommendations on the
Amazon Books Reviews dataset:
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

Main features included (based on your notebook's steps and libraries):
- Data loading + optional sampling
- Robust text cleaning (regex, lowercasing, punctuation, stopwords, lemmatization)
- Canonicalization helpers (e.g., LOTR title normalization)
- Optional fuzzy deduplication of near-duplicate titles (rapidfuzz)
- Content-based recommendations:
  - TF-IDF on titles (and optionally review text)
  - Exact KNN (sklearn) or approximate ANN with Annoy (for scalable similarity)
- User-based collaborative filtering:
  - User–Item matrix + cosine KNN over users
- Hybrid blending of content + user signals
- Evaluation: Precision@K, Recall@K, MAP@K
- Optional quick EDA plots (distributions)
- CLI with configurable options

Author: Reihan (you!)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# Text & NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Content-based
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# KNN for CF (user-based)
from sklearn.neighbors import NearestNeighbors

# Fuzzy / ANN (optional)
from rapidfuzz import fuzz, process as rf_process
from annoy import AnnoyIndex

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Config & Utilities
# -----------------------------

@dataclass
class Config:
    file_path: str
    sample_frac: float = 0.1
    random_state: int = 42

    # content-based
    use_review_text: bool = False
    max_features: int = 100_000
    n_neighbors: int = 25
    use_annoy: bool = False
    annoy_trees: int = 20

    # CF
    cf_n_neighbors: int = 50
    min_user_ratings: int = 3
    min_item_ratings: int = 3

    # hybrid
    top_k: int = 10
    alpha: float = 0.6  # weight for content vs CF (0..1)

    # dedup
    dedup_titles: bool = True
    dedup_threshold: int = 92  # 0..100 (rapidfuzz ratio)

    # eval
    do_eval: bool = True
    eval_test_size: float = 0.2

    # plots
    make_plots: bool = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


# -----------------------------
# Data Loading & Cleaning
# -----------------------------

def load_data(cfg: Config) -> pd.DataFrame:
    """
    Load CSV and optionally sample rows.
    Expected columns (dataset-dependent):
      - 'UserId' (or 'review/userId'), 'ProductId' (or 'asin'), 'Title', 'review/score' (or 'Rating')
    """
    logging.info(f"Loading dataset from {cfg.file_path}")
    df = pd.read_csv(cfg.file_path)

    # Normalize common column names
    col_map = {
        "review/userId": "UserId",
        "review/profileName": "ProfileName",
        "asin": "ProductId",
        "title": "Title",
        "review/score": "Rating",
        "review/text": "ReviewText",
        "review/time": "ReviewTime",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Keep only relevant columns if present
    keep_cols = [c for c in ["UserId", "ProductId", "Title", "Rating", "ReviewText"] if c in df.columns]
    df = df[keep_cols].copy()

    # Basic filtering
    if "Rating" in df.columns:
        # keep valid ratings
        df = df[(df["Rating"].notna()) & (df["Rating"] > 0)]
    df = df.dropna(subset=["UserId", "ProductId", "Title"])

    # Optional sampling
    if 0 < cfg.sample_frac < 1.0:
        df = df.sample(frac=cfg.sample_frac, random_state=cfg.random_state)
        logging.info(f"Sampled dataset shape: {df.shape}")
    else:
        logging.info(f"Full dataset shape: {df.shape}")

    # Ensure types
    df["UserId"] = df["UserId"].astype(str)
    df["ProductId"] = df["ProductId"].astype(str)
    df["Title"] = df["Title"].astype(str)
    if "Rating" in df.columns:
        df["Rating"] = df["Rating"].astype(float)

    return df


# --- Text cleaning helpers ---

_wordnet = None
_stopwords_en = None
_lemmatizer = None


def _ensure_nltk():
    global _wordnet, _stopwords_en, _lemmatizer
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    _stopwords_en = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/extra spaces, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        return ""
    if _lemmatizer is None:
        _ensure_nltk()

    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [w for w in t.split() if w and w not in _stopwords_en]
    lemmas = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(lemmas)


def clean_lotr_titles(title: str) -> str:
    """
    Canonicalize 'The Lord of the Rings' series variants into a single title token.
    """
    if not isinstance(title, str):
        return title
    pattern = r"the\s+lord\s+of\s+the\s+rings.*"
    if re.match(pattern, title.strip().lower()):
        return "The Lord of the Rings"
    return title


def normalize_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonicalization passes (can be extended with more rule-based cleaners).
    """
    df = df.copy()
    df["Title"] = df["Title"].apply(clean_lotr_titles)
    return df


def fuzzy_deduplicate_titles(
    df: pd.DataFrame, threshold: int = 92
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fuzzy-merge near-duplicate titles to a canonical representative.
    Returns deduplicated df and a mapping original_title -> canonical_title.
    """
    titles = df["Title"].astype(str).str.strip().tolist()
    unique_titles = sorted(set(titles))
    canon_map: Dict[str, str] = {}

    # Greedy pass: map close titles to the first "anchor" found
    anchors: List[str] = []
    for t in unique_titles:
        if not anchors:
            anchors.append(t)
            canon_map[t] = t
            continue
        # find best match among anchors
        best = rf_process.extractOne(t, anchors, scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            canon_map[t] = best[0]
        else:
            anchors.append(t)
            canon_map[t] = t

    df = df.copy()
    df["TitleRaw"] = df["Title"]
    df["Title"] = df["Title"].map(canon_map)
    logging.info(f"Fuzzy dedup reduced titles from {len(unique_titles)} to {len(set(df['Title']))}")
    return df, canon_map


# -----------------------------
# Content-Based Models
# -----------------------------

@dataclass
class ContentModel:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    index_to_pid: np.ndarray
    pid_to_index: Dict[str, int]
    title_of_pid: Dict[str, str]
    annoy: Optional[AnnoyIndex] = None
    annoy_dim: Optional[int] = None


def build_content_model(
    df: pd.DataFrame,
    use_review_text: bool = False,
    max_features: int = 100_000,
    use_annoy: bool = False,
    annoy_trees: int = 20,
) -> ContentModel:
    """
    Build a TF-IDF representation and optional Annoy index for approximate search.
    """
    logging.info("Building content-based model...")
    # Prepare text
    if use_review_text and "ReviewText" in df.columns:
        text_series = (
            df["Title"].fillna("") + " " + df["ReviewText"].fillna("")
        ).astype(str).apply(clean_text)
    else:
        text_series = df["Title"].fillna("").astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text_series)

    # Map indices to product IDs and titles
    pid = df["ProductId"].values
    index_to_pid = np.array(pid)
    pid_to_index = {p: i for i, p in enumerate(index_to_pid)}
    title_of_pid = df.set_index("ProductId")["Title"].to_dict()

    content = ContentModel(
        vectorizer=vectorizer,
        matrix=X.tocsr(),
        index_to_pid=index_to_pid,
        pid_to_index=pid_to_index,
        title_of_pid=title_of_pid,
        annoy=None,
        annoy_dim=None,
    )

    if use_annoy:
        dim = X.shape[1]
        logging.info(f"Building Annoy index (dim={dim}, trees={annoy_trees})...")
        ann = AnnoyIndex(dim, metric="angular")  # angular ~ cosine distance
        # Annoy expects dense vectors per item; iterate row-wise
        for i in range(X.shape[0]):
            vec = X.getrow(i).toarray().ravel().astype(np.float32)
            ann.add_item(i, vec)
        ann.build(annoy_trees)
        content.annoy = ann
        content.annoy_dim = dim

    logging.info("Content model ready.")
    return content


def similar_items_content(
    content: ContentModel, query_pid: str, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Return list of (product_id, similarity score) for content neighbors.
    """
    if query_pid not in content.pid_to_index:
        return []

    q_idx = content.pid_to_index[query_pid]

    if content.annoy is not None:
        # Approximate NN via Annoy
        idxs = content.annoy.get_nns_by_item(q_idx, top_k + 1, include_distances=True)
        nn_idx, dists = idxs
        pairs = []
        for idx, dist in zip(nn_idx, dists):
            if idx == q_idx:
                continue
            sim = 1.0 - (dist**2) / 2.0  # approximate cosine sim from angular distance
            pairs.append((content.index_to_pid[idx], float(sim)))
        return pairs[:top_k]
    else:
        q_vec = content.matrix[q_idx]
        sims = cosine_similarity(q_vec, content.matrix).ravel()
        order = np.argsort(-sims)
        recs = []
        for idx in order:
            if idx == q_idx:
                continue
            recs.append((content.index_to_pid[idx], float(sims[idx])))
            if len(recs) >= top_k:
                break
        return recs


# -----------------------------
# Collaborative Filtering (User-based KNN)
# -----------------------------

@dataclass
class CFModel:
    user_encoder: Dict[str, int]
    item_encoder: Dict[str, int]
    inv_user_encoder: Dict[int, str]
    inv_item_encoder: Dict[int, str]
    user_item: sparse.csr_matrix
    knn: NearestNeighbors
    ratings_csr: sparse.csr_matrix  # same as user_item


def _encode_ids(series: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniques = pd.Index(series).astype(str).unique().tolist()
    enc = {v: i for i, v in enumerate(uniques)}
    inv = {i: v for v, i in enc.items()}
    return enc, inv


def build_cf_model(
    df: pd.DataFrame, n_neighbors: int = 50, min_user_r: int = 3, min_item_r: int = 3
) -> CFModel:
    """
    Build a user-based CF model using cosine KNN over the User–Item matrix.
    """
    logging.info("Building CF model (user-based KNN, cosine)...")
    # Filter by minimum activity
    if "Rating" not in df.columns:
        raise ValueError("CF requires a 'Rating' column.")
    g_user = df.groupby("UserId")["Rating"].count()
    g_item = df.groupby("ProductId")["Rating"].count()
    active_users = set(g_user[g_user >= min_user_r].index)
    active_items = set(g_item[g_item >= min_item_r].index)

    dff = df[df["UserId"].isin(active_users) & df["ProductId"].isin(active_items)].copy()
    logging.info(f"CF active subset shape: {dff.shape}")

    user_enc, inv_user = _encode_ids(dff["UserId"])
    item_enc, inv_item = _encode_ids(dff["ProductId"])

    u = dff["UserId"].map(user_enc).values
    i = dff["ProductId"].map(item_enc).values
    r = dff["Rating"].values

    shape = (len(user_enc), len(item_enc))
    mat = sparse.coo_matrix((r, (u, i)), shape=shape).tocsr()

    # Normalize rows (users) to unit length for cosine
    row_norms = np.sqrt(mat.power(2).sum(axis=1)).A.ravel()
    row_norms[row_norms == 0] = 1.0
    mat_norm = sparse.diags(1.0 / row_norms) @ mat

    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n_neighbors, mat_norm.shape[0]))
    knn.fit(mat_norm)

    return CFModel(
        user_encoder=user_enc,
        item_encoder=item_enc,
        inv_user_encoder=inv_user,
        inv_item_encoder=inv_item,
        user_item=mat_norm.tocsr(),
        knn=knn,
        ratings_csr=mat.tocsr(),
    )


def _predict_user_item_score(cf: CFModel, user_id: str, item_id: str) -> float:
    """
    Score by averaging ratings from nearest neighbor users (simple baseline).
    """
    if user_id not in cf.user_encoder or item_id not in cf.item_encoder:
        return 0.0

    u_idx = cf.user_encoder[user_id]
    i_idx = cf.item_encoder[item_id]

    # Find similar users for u_idx
    u_vec = cf.user_item[u_idx]
    distances, indices = cf.knn.kneighbors(u_vec, n_neighbors=min(25, cf.user_item.shape[0]))
    # cosine distance -> similarity
    sims = 1.0 - distances.ravel()
    neigh_idxs = indices.ravel()

    # Gather neighbor ratings for item i_idx
    neighbor_ratings = []
    for s, n_idx in zip(sims, neigh_idxs):
        r = cf.ratings_csr[n_idx, i_idx]
        if r != 0:
            neighbor_ratings.append((float(r), float(s)))

    if not neighbor_ratings:
        return 0.0

    # Weighted average by similarity
    num = sum(r * s for r, s in neighbor_ratings)
    den = sum(abs(s) for _, s in neighbor_ratings) + 1e-8
    return float(num / den)


# -----------------------------
# Hybrid Blending
# -----------------------------

def hybrid_recommendations(
    user_id: str,
    seed_pid: str,
    content: ContentModel,
    cf: CFModel,
    top_k: int = 10,
    alpha: float = 0.6,
) -> List[Tuple[str, float]]:
    """
    Combine content similarity and CF predicted score.
    alpha: weight on content sim; (1 - alpha) on CF score normalized to [0,1].
    """
    content_neighbors = similar_items_content(content, seed_pid, top_k=top_k * 5)
    if not content_neighbors:
        return []

    # Normalize lists
    max_sim = max((s for _, s in content_neighbors), default=1.0) or 1.0
    blended: List[Tuple[str, float]] = []
    for pid, sim in content_neighbors:
        cf_score = _predict_user_item_score(cf, user_id, pid)
        # normalize CF to [0,1] using 5-star assumption
        cf_norm = min(max(cf_score / 5.0, 0.0), 1.0)
        hybrid = alpha * (sim / max_sim) + (1 - alpha) * cf_norm
        blended.append((pid, float(hybrid)))

    blended.sort(key=lambda x: x[1], reverse=True)
    # remove seed if present
    blended = [(p, s) for (p, s) in blended if p != seed_pid]
    return blended[:top_k]


# -----------------------------
# Evaluation
# -----------------------------

def _precision_recall_at_k(
    ground_truth: List[str], ranked_list: List[str], k: int
) -> Tuple[float, float]:
    pred_k = ranked_list[:k]
    hit_set = set(ground_truth)
    hits = sum(1 for p in pred_k if p in hit_set)
    prec = hits / max(1, len(pred_k))
    rec = hits / max(1, len(ground_truth))
    return prec, rec


def _avg_precision_at_k(ground_truth: List[str], ranked_list: List[str], k: int) -> float:
    score = 0.0
    hits = 0
    for i, p in enumerate(ranked_list[:k], start=1):
        if p in ground_truth:
            hits += 1
            score += hits / i
    return score / max(1, len(ground_truth))


def evaluate_recommender(
    df: pd.DataFrame,
    content: ContentModel,
    cf: CFModel,
    k: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Simple holdout by users: for each user, hold out one positive item.
    Evaluate Precision/Recall/MAP@K on that holdout.
    """
    logging.info("Evaluating hybrid recommender (holdout by user)...")
    rng = np.random.default_rng(random_state)
    by_user = df.groupby("UserId")["ProductId"].apply(list)

    users = [u for u, items in by_user.items() if len(items) >= 2]
    rng.shuffle(users)
    n_test = int(len(users) * test_size)
    test_users = set(users[:n_test])

    precs, recs, maps = [], [], []
    for u in test_users:
        items = by_user[u]
        # holdout a random item as "ground truth"
        gt_item = rng.choice(items)
        # pick a seed different from gt if possible, else reuse
        seed = rng.choice([i for i in items if i != gt_item] or [gt_item])

        # Generate hybrid recs using the seed
        ranked = [p for p, _ in hybrid_recommendations(u, seed, content, cf, top_k=max(k, 50))]

        # Evaluate against the single holdout
        gt = [gt_item]
        p, r = _precision_recall_at_k(gt, ranked, k)
        ap = _avg_precision_at_k(gt, ranked, k)
        precs.append(p)
        recs.append(r)
        maps.append(ap)

    results = {
        f"precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        "n_eval_users": int(len(test_users)),
    }
    logging.info(f"Eval results: {results}")
    return results


# -----------------------------
# Quick Plots (optional)
# -----------------------------

def quick_plots(df: pd.DataFrame, out_dir: Optional[str] = None):
    """
    A couple of quick plots (top titles by count, rating distribution).
    """
    logging.info("Generating quick plots...")
    sns.set_theme()

    plt.figure()
    df["Title"].value_counts().head(20).plot(kind="barh")
    plt.title("Top 20 Titles by Count")
    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "top_titles.png")
        plt.savefig(path)
        logging.info(f"Saved {path}")
    else:
        plt.show()

    if "Rating" in df.columns:
        plt.figure()
        df["Rating"].plot(kind="hist", bins=20)
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.tight_layout()
        if out_dir:
            path = os.path.join(out_dir, "rating_distribution.png")
            plt.savefig(path)
            logging.info(f"Saved {path}")
        else:
            plt.show()


# -----------------------------
# Main Pipeline
# -----------------------------

def run_pipeline(cfg: Config):
    # Load
    df = load_data(cfg)

    # Canonicalize + optional fuzzy dedup titles
    df = normalize_titles(df)
    if cfg.dedup_titles:
        df, _ = fuzzy_deduplicate_titles(df, threshold=cfg.dedup_threshold)

    # Build content and CF models
    content = build_content_model(
        df=df,
        use_review_text=cfg.use_review_text,
        max_features=cfg.max_features,
        use_annoy=cfg.use_annoy,
        annoy_trees=cfg.annoy_trees,
    )

    cf = build_cf_model(
        df=df,
        n_neighbors=cfg.cf_n_neighbors,
        min_user_r=cfg.min_user_ratings,
        min_item_r=cfg.min_item_ratings,
    )

    # Example: pick a random user and one of their items as seed
    rng = np.random.default_rng(cfg.random_state)
    users = df["UserId"].unique().tolist()
    user = rng.choice(users)
    user_items = df.loc[df["UserId"] == user, "ProductId"].unique().tolist()
    seed = rng.choice(user_items)

    logging.info(f"Example hybrid recs for user={user} seed_pid={seed}")
    recs = hybrid_recommendations(
        user_id=user,
        seed_pid=seed,
        content=content,
        cf=cf,
        top_k=cfg.top_k,
        alpha=cfg.alpha,
    )
    # Pretty print with titles
    preview = [(pid, score, content.title_of_pid.get(pid, "")) for pid, score in recs]
    for pid, s, title in preview:
        logging.info(f"  {pid:>12} | {s:0.4f} | {title}")

    # Evaluate
    if cfg.do_eval:
        _ = evaluate_recommender(
            df=df,
            content=content,
            cf=cf,
            k=cfg.top_k,
            test_size=cfg.eval_test_size,
            random_state=cfg.random_state,
        )

    # Plots
    if cfg.make_plots:
        quick_plots(df, out_dir="plots")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Hybrid Recommender for Amazon Books Reviews")
    p.add_argument("--file", required=True, help="Path to CSV file")
    p.add_argument("--sample", type=float, default=0.1, help="Sampling fraction (0<frac<=1)")
    p.add_argument("--use-review-text", action="store_true", help="Include review text in TF-IDF")
    p.add_argument("--max-features", type=int, default=100000, help="Max TF-IDF vocabulary size")
    p.add_argument("--neighbors", type=int, default=25, help="Content neighbors (KNN or Annoy)")
    p.add_argument("--annoy", action="store_true", help="Use Annoy approximate index for content similarity")
    p.add_argument("--annoy-trees", type=int, default=20, help="Annoy trees")
    p.add_argument("--cf-neighbors", type=int, default=50, help="CF user KNN neighbors")
    p.add_argument("--min-user", type=int, default=3, help="Minimum ratings per user")
    p.add_argument("--min-item", type=int, default=3, help="Minimum ratings per item")
    p.add_argument("--topk", type=int, default=10, help="Top-K recommendations")
    p.add_argument("--alpha", type=float, default=0.6, help="Blend weight for content vs CF (0..1)")
    p.add_argument("--no-dedup", action="store_true", help="Disable fuzzy deduplication of titles")
    p.add_argument("--dedup-threshold", type=int, default=92, help="Rapidfuzz similarity threshold 0..100")
    p.add_argument("--no-eval", action="store_true", help="Disable evaluation")
    p.add_argument("--eval-size", type=float, default=0.2, help="Test size for holdout")
    p.add_argument("--plots", action="store_true", help="Generate quick plots")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    return Config(
        file_path=args.file,
        sample_frac=args.sample,
        random_state=args.seed,
        use_review_text=args.use_review_text,
        max_features=args.max_features,
        n_neighbors=args.neighbors,
        use_annoy=args.annoy,
        annoy_trees=args.annoy_trees,
        cf_n_neighbors=args.cf_neighbors,
        min_user_ratings=args.min_user,
        min_item_ratings=args.min_item,
        top_k=args.topk,
        alpha=args.alpha,
        dedup_titles=(not args.no_dedup),
        dedup_threshold=args.dedup_threshold,
        do_eval=(not args.no_eval),
        eval_test_size=args.eval_size,
        make_plots=args.plots,
    )


def main():
    setup_logging()
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
