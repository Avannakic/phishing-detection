import os

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
TARGET_SYNTH_PHISH_SIZE = 10000
LEAKAGE_THRESHOLD = 0.85
SEED = 42

POOL_CONFIG = [
    ("real_ham.csv", REAL_DIR, 0, TARGET_REAL_BASE_SIZE),
    ("real_spam.csv", REAL_DIR, 1, TARGET_REAL_BASE_SIZE),
    ("real_phishing.csv", REAL_DIR, 2, TARGET_REAL_PHISH_SIZE),
    ("synthetic_ham.csv", INTERIM_DIR, 0, TARGET_SYNTH_BASE_SIZE),
    ("synthetic_spam.csv", INTERIM_DIR, 1, TARGET_SYNTH_BASE_SIZE),
    ("final_synthetic_phishing.csv", INTERIM_DIR, 2, TARGET_SYNTH_PHISH_SIZE),
]


def clean_pool(df: pd.DataFrame, label: int) -> pd.DataFrame:
    df = df.dropna(subset=["message"]).copy()
    df["message"] = df["message"].astype(str).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    df["norm_text"] = df["message"].str.lower()
    df = (
        df[df["norm_text"] != ""]
        .drop_duplicates(subset=["norm_text"])
        .drop(columns=["norm_text"])
    )
    df["label"] = label
    return df[["message", "label"]].reset_index(drop=True)


def align_pool(df: pd.DataFrame, target_size: int, pool_name: str) -> pd.DataFrame:
    if len(df) < target_size:
        raise ValueError(
            f"Пул {pool_name} содержит {len(df)} уникальных строк после очистки; "
            f"требуется минимум {target_size}. Дополните исходные данные или генерацию."
        )
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=SEED)
    return df.reset_index(drop=True)


def remove_leakage(
    df_synth: pd.DataFrame,
    df_real_phish: pd.DataFrame,
    threshold: float = LEAKAGE_THRESHOLD,
) -> pd.DataFrame:
    print("[INFO] Проверка утечек между синтетикой и всем реальным фишингом...")
    real_norm = set(df_real_phish["message"].str.lower().str.strip())
    synth_norm = df_synth["message"].str.lower().str.strip()
    exact_mask = synth_norm.isin(real_norm)
    df_clean = df_synth[~exact_mask].reset_index(drop=True)
    print(f"  -> Удалено точных совпадений: {int(exact_mask.sum())}")

    if df_clean.empty:
        return df_clean

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_real = vectorizer.fit_transform(df_real_phish["message"].tolist())
    tfidf_synth = vectorizer.transform(df_clean["message"].tolist())
    sim_matrix = cosine_similarity(tfidf_synth, tfidf_real)
    max_sim = np.max(sim_matrix, axis=1)
    near_dup_mask = max_sim > threshold

    print(f"  -> Максимальное косинусное сходство TF-IDF: {np.max(max_sim):.4f}")
    print(f"  -> Удалено нечетких дубликатов (сходство > {threshold}): {int(near_dup_mask.sum())}")
    return df_clean[~near_dup_mask].reset_index(drop=True)


def load_aligned_pool(filename: str, directory: str, label: int, target_size: int) -> pd.DataFrame:
    path = os.path.join(directory, filename)
    df = clean_pool(pd.read_csv(path), label=label)
    print(f"[INFO] {filename}: {len(df)} уникальных строк до выравнивания")
    return align_pool(df, target_size, filename)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pools = {}

    for filename, directory, label, target_size in POOL_CONFIG:
        if filename == "final_synthetic_phishing.csv":
            continue
        pools[filename] = load_aligned_pool(filename, directory, label, target_size)

    synth_phish_path = os.path.join(INTERIM_DIR, "final_synthetic_phishing.csv")
    synth_phish = clean_pool(pd.read_csv(synth_phish_path), label=2)
    synth_phish = remove_leakage(synth_phish, pools["real_phishing.csv"])
    pools["final_synthetic_phishing.csv"] = align_pool(
        synth_phish, TARGET_SYNTH_PHISH_SIZE, "final_synthetic_phishing.csv"
    )

    for filename, _, _, _ in POOL_CONFIG:
        output_path = os.path.join(PROCESSED_DIR, filename)
        pools[filename].to_csv(output_path, index=False, encoding="utf-8")
        print(f"[INFO] Сохранено: {output_path} ({len(pools[filename])} строк)")

    print("[SUCCESS] Все шесть пулов выровнены; статическое Test-разбиение не выполнялось.")


if __name__ == "__main__":
    main()
