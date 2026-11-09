#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 06: Finalisation and alignment of all six data pools (real + synthetic)
in data/processed/ with strict cross-corpus leakage removal (remove_leakage)
between synthetic and real phishing.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "real")

TARGET_REAL_BASE_SIZE = 20000
TARGET_REAL_PHISH_SIZE = 300
TARGET_SYNTH_BASE_SIZE = 14000
TARGET_SYNTH_PHISH_SIZE = 7000
LEAKAGE_THRESHOLD = 0.75
SEED = 42

# (input_file, input_dir, output_file, class_id, quota_size)
POOL_CONFIG = [
    ("real_ham.csv", REAL_DIR, "real_ham.csv", 0, TARGET_REAL_BASE_SIZE),
    ("real_spam.csv", REAL_DIR, "real_spam.csv", 1, TARGET_REAL_BASE_SIZE),
    ("real_phishing.csv", REAL_DIR, "real_phishing.csv", 2, TARGET_REAL_PHISH_SIZE),
    ("final_synthetic_ham.csv", INTERIM_DIR, "synthetic_ham.csv", 0, TARGET_SYNTH_BASE_SIZE),
    ("final_synthetic_spam.csv", INTERIM_DIR, "synthetic_spam.csv", 1, TARGET_SYNTH_BASE_SIZE),
    ("final_synthetic_phishing.csv", INTERIM_DIR, "synthetic_phishing.csv", 2, TARGET_SYNTH_PHISH_SIZE),
]


def remove_leakage(
    df_synth: pd.DataFrame,
    df_real_phish: pd.DataFrame,
    threshold: float = LEAKAGE_THRESHOLD,
) -> pd.DataFrame:
    print("[INFO] Checking for leakage between synthetic and real phishing pools...")
    real_norm = set(df_real_phish["message"].str.casefold().str.strip())
    synth_norm = df_synth["message"].str.casefold().str.strip()
    exact_mask = synth_norm.isin(real_norm)
    df_clean = df_synth[~exact_mask].reset_index(drop=True)
    print(f"  -> Exact duplicates removed: {int(exact_mask.sum())}")

    if df_clean.empty or df_real_phish.empty:
        return df_clean

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_real = vectorizer.fit_transform(df_real_phish["message"].tolist())
    tfidf_synth = vectorizer.transform(df_clean["message"].tolist())
    sim_matrix = cosine_similarity(tfidf_synth, tfidf_real)
    max_sim = np.max(sim_matrix, axis=1)
    near_dup_mask = max_sim > threshold

    print(f"  -> Maximum TF-IDF cosine similarity: {np.max(max_sim):.4f}")
    print(f"  -> Near-duplicates removed (similarity > {threshold}): {int(near_dup_mask.sum())}")
    return df_clean[~near_dup_mask].reset_index(drop=True)


def prepare_and_align_corpus(
    csv_path: str,
    class_id: int,
    quota_size: int,
    rng_seed: int = SEED,
    leakage_ref_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Read corpus from CSV ('message' or 'text' column), normalize whitespace,
    perform case-insensitive deduplication via .str.casefold(),
    remove leakage against real phishing if specified, and align to target quota.
    """
    raw_df = pd.read_csv(csv_path)
    text_col = "message" if "message" in raw_df.columns else "text"
    if text_col not in raw_df.columns:
        raise KeyError(f"File {csv_path} lacks a 'message' or 'text' column.")

    df = raw_df.dropna(subset=[text_col]).copy()
    df["message"] = (
        df[text_col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["fold_key"] = df["message"].str.casefold()
    df = (
        df[df["fold_key"] != ""]
        .drop_duplicates(subset=["fold_key"])
        .drop(columns=["fold_key"])
        .reset_index(drop=True)
    )
    df["label"] = class_id
    df = df[["message", "label"]]

    print(f"[INFO] {os.path.basename(csv_path)}: {len(df)} unique records before alignment")

    if leakage_ref_df is not None:
        df = remove_leakage(df, leakage_ref_df, threshold=LEAKAGE_THRESHOLD)

    if len(df) < quota_size:
        raise ValueError(
            f"Corpus {csv_path} contains {len(df)} unique records after cleaning; "
            f"minimum required is {quota_size}. Expand raw data or generation."
        )
    if len(df) > quota_size:
        df = df.sample(n=quota_size, random_state=rng_seed).reset_index(drop=True)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pools = {}

    for in_name, in_dir, out_name, class_id, quota_size in POOL_CONFIG:
        if out_name == "synthetic_phishing.csv":
            continue
        csv_path = os.path.join(in_dir, in_name)
        pools[out_name] = prepare_and_align_corpus(
            csv_path=csv_path,
            class_id=class_id,
            quota_size=quota_size,
            rng_seed=SEED,
        )

    synth_phish_in_path = os.path.join(INTERIM_DIR, "final_synthetic_phishing.csv")
    pools["synthetic_phishing.csv"] = prepare_and_align_corpus(
        csv_path=synth_phish_in_path,
        class_id=2,
        quota_size=TARGET_SYNTH_PHISH_SIZE,
        rng_seed=SEED,
        leakage_ref_df=pools["real_phishing.csv"],
    )

    for _, _, out_name, _, _ in POOL_CONFIG:
        output_path = os.path.join(PROCESSED_DIR, out_name)
        pools[out_name].to_csv(output_path, index=False, encoding="utf-8")
        print(f"[INFO] Saved: {output_path} ({len(pools[out_name])} records)")

    print("[SUCCESS] All six pools aligned; static test splitting was omitted.")


if __name__ == "__main__":
    main()
