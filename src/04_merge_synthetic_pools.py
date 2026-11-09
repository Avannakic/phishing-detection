#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 04: Merging baseline and targeted synthetic phishing pools,
cross-class alphanumeric deduplication (Phishing -> Spam -> Ham),
n-gram frequency filtering, and batched TF-IDF semantic deduplication.
"""

from collections import Counter
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
GLOBAL_SEED = 42

# Class-specific deduplication configuration:
# (label_val, max_first_word, max_bigram_prefix, max_trigram, sim_thresh, output_filename)
CLASS_DEDUP_CONFIGS = [
    (2, 40, 3, 25, 0.65, "merged_synthetic_phishing.csv"),
    (1, 80, 6, 40, 0.70, "dedup_synthetic_spam.csv"),
    (0, 80, 6, 40, 0.70, "dedup_synthetic_ham.csv"),
]


def load_messages(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[WARN] File {filepath} not found; returning empty list.")
        return []
    df = pd.read_csv(filepath)
    col = "message" if "message" in df.columns else "text"
    return df[col].dropna().astype(str).tolist()


def deduplicate_class_pool(
    raw_msgs: list,
    label_val: int,
    max_first_word: int,
    max_bigram_prefix: int,
    max_trigram: int,
    sim_thresh: float,
    output_path: str,
    global_seen_norms: set,
) -> pd.DataFrame:
    rng = np.random.default_rng(GLOBAL_SEED + label_val)

    # Stage 1: Cross-class alphanumeric deduplication
    min_words = 2 if label_val == 0 else 4
    stage1 = []
    for line in raw_msgs:
        words = re.findall(r"[а-яёa-z0-9]+", line.lower())
        if len(words) < min_words:
            continue
        alpha_norm = " ".join(words)
        if alpha_norm in global_seen_norms:
            continue
        global_seen_norms.add(alpha_norm)
        stage1.append(line)

    rng.shuffle(stage1)

    # Stage 2: N-gram frequency filtering (prefix unigram, prefix bigram, internal trigram)
    stage2 = []
    fw_counts = Counter()
    bp_counts = Counter()
    tg_counts = Counter()

    for line in stage1:
        words = re.findall(r"[а-яёa-z0-9]+", line.lower())
        if len(words) < min_words:
            continue
        fw = words[0]
        bp = f"{words[0]} {words[1]}"

        if fw_counts[fw] >= max_first_word or bp_counts[bp] >= max_bigram_prefix:
            continue

        tgs = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        if any(tg_counts[tg] >= max_trigram for tg in tgs):
            continue

        fw_counts[fw] += 1
        bp_counts[bp] += 1
        for tg in tgs:
            tg_counts[tg] += 1
        stage2.append(line)

    # Stage 3: Batched TF-IDF semantic deduplication on URL-stripped text
    if len(stage2) >= 2:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=25000,
            dtype=np.float32,
        )
        try:
            X = vec.fit_transform(stage2)
        except ValueError:
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=25000,
                dtype=np.float32,
            )
            X = vec.fit_transform(stage2)
        n = len(stage2)
        removed = set()
        keep_indices = []
        batch_size = 1000

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sim_chunk = (X[start:end] @ X.T).toarray()
            for idx_in_chunk in range(end - start):
                i = start + idx_in_chunk
                if i in removed:
                    continue
                keep_indices.append(i)
                dups = np.where(sim_chunk[idx_in_chunk, i + 1:] >= sim_thresh)[0] + (i + 1)
                removed.update(dups.tolist())

        final_msgs = [stage2[i] for i in keep_indices]
    else:
        final_msgs = list(stage2)

    out_df = pd.DataFrame({
        "message": final_msgs,
        "label": label_val,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    print(
        f"[Class {label_val}] -> {output_path}\n"
        f"  Input: {len(raw_msgs)} | Stage 1 (Alpha-Dedup): {len(stage1)} | "
        f"Stage 2 (N-grams): {len(stage2)} | Stage 3 (TF-IDF): {len(final_msgs)}"
    )
    return out_df


def main():
    base_phishing_path = os.path.join(DATA_DIR, "cleaned_synthetic_base.csv")
    targeted_phishing_path = os.path.join(DATA_DIR, "cleaned_synthetic_targeted.csv")
    spam_path = os.path.join(DATA_DIR, "cleaned_synthetic_spam.csv")
    ham_path = os.path.join(DATA_DIR, "cleaned_synthetic_ham.csv")

    print("[INFO] Loading cleaned datasets from data/interim/...")
    phishing_msgs = load_messages(base_phishing_path) + load_messages(targeted_phishing_path)
    spam_msgs = load_messages(spam_path)
    ham_msgs = load_messages(ham_path)

    pools_by_label = {
        2: phishing_msgs,
        1: spam_msgs,
        0: ham_msgs,
    }

    global_seen_norms = set()
    for label_val, max_fw, max_bp, max_tg, sim_thresh, out_name in CLASS_DEDUP_CONFIGS:
        msgs = pools_by_label[label_val]
        if not msgs:
            print(f"[WARN] Class pool {label_val} is empty; skipping.")
            continue
        out_path = os.path.join(DATA_DIR, out_name)
        deduplicate_class_pool(
            raw_msgs=msgs,
            label_val=label_val,
            max_first_word=max_fw,
            max_bigram_prefix=max_bp,
            max_trigram=max_tg,
            sim_thresh=sim_thresh,
            output_path=out_path,
            global_seen_norms=global_seen_norms,
        )


if __name__ == "__main__":
    main()
