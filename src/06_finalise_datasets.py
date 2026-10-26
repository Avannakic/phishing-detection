import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- НАСТРОЙКА ДИРЕКТОРИЙ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    if os.path.basename(BASE_DIR) == "data_generation"
    else BASE_DIR
)

INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "real")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Входные синтетические файлы из предыдущих шагов (00-05)
SYNTH_PHISH_FILE = os.path.join(INTERIM_DIR, "final_synthetic_phishing.csv")
SYNTH_HAM_FILE = os.path.join(INTERIM_DIR, "synthetic_ham.csv")
SYNTH_SPAM_FILE = os.path.join(INTERIM_DIR, "synthetic_spam.csv")

# Входные единые пулы реальных данных
REAL_HAM_FILE = os.path.join(REAL_DIR, "real_ham.csv")
REAL_SPAM_FILE = os.path.join(REAL_DIR, "real_spam.csv")
REAL_PHISH_FILE = os.path.join(REAL_DIR, "real_phishing.csv")

# Выходные очищенные и проверенные на утечки пулы для 5-фолдовой кросс-валидации (Шаг 07)
OUT_REAL_HAM = os.path.join(PROCESSED_DIR, "real_ham.csv")
OUT_REAL_SPAM = os.path.join(PROCESSED_DIR, "real_spam.csv")
OUT_REAL_PHISH = os.path.join(PROCESSED_DIR, "real_phishing.csv")

OUT_SYNTH_HAM = os.path.join(PROCESSED_DIR, "synthetic_ham.csv")
OUT_SYNTH_SPAM = os.path.join(PROCESSED_DIR, "synthetic_spam.csv")
OUT_SYNTH_PHISH = os.path.join(PROCESSED_DIR, "final_synthetic_phishing.csv")

# Целевые размеры синтетических пулов для обучения моделей
TARGET_SYNTH_BASE_SIZE = 14000
TARGET_SYNTH_PHISH_SIZE = 7000
SEED = 42


def clean_pool(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Очищает пул сообщений от пустых строк и точных дубликатов по нормализованному тексту."""
    df = df.dropna(subset=["message"]).copy()
    df["norm_text"] = df["message"].astype(str).str.lower().str.strip()
    df = df[df["norm_text"] != ""].drop_duplicates(subset=["norm_text"]).drop(columns=["norm_text"])
    df["label"] = label
    return df.reset_index(drop=True)


def remove_leakage(df_synth: pd.DataFrame, df_real_phish: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Двухэтапная фильтрация точных и нечетких пересечений синтетического фишинга
    со всем пулом реального фишинга (так как при кросс-валидации каждый реальный
    пример побывает в тестовой выборке).
    """
    print("[INFO] Проверка пересечений между синтетикой и реальным фишингом (Data Leakage Check)...")

    real_norm = set(df_real_phish["message"].astype(str).str.lower().str.strip())
    synth_norm = df_synth["message"].astype(str).str.lower().str.strip()

    # 1. Точные совпадения
    exact_mask = synth_norm.isin(real_norm)
    df_clean = df_synth[~exact_mask].reset_index(drop=True)
    print(f"  -> Удалено точных совпадений: {exact_mask.sum()}")

    # 2. Нечеткие дубликаты (TF-IDF на униграммах и биграммах)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_real = vectorizer.fit_transform(df_real_phish["message"].astype(str))
    tfidf_synth = vectorizer.transform(df_clean["message"].astype(str))

    sim_matrix = cosine_similarity(tfidf_synth, tfidf_real)
    max_sim = np.max(sim_matrix, axis=1)
    near_dup_mask = max_sim > threshold

    print(f"  -> Максимальное косинусное сходство TF-IDF: {np.max(max_sim):.4f}")
    print(f"  -> Удалено нечетких дубликатов (сходство > {threshold}): {near_dup_mask.sum()}")

    return df_clean[~near_dup_mask].reset_index(drop=True)


def main():
    # 1. Загрузка и очистка единого пула РЕАЛЬНОГО фишинга (без отрезания 20%, т.к. далее идет 5-fold CV)
    real_phish = clean_pool(pd.read_csv(REAL_PHISH_FILE), label=2)
    print(f"[INFO] Реальный фишинг очищен: {len(real_phish)} уникальных примеров (готовы к 5-фолдовой CV)")

    # 2. Загрузка синтетического фишинга и фильтрация утечек относительно всего реального пула
    synth_phish = clean_pool(pd.read_csv(SYNTH_PHISH_FILE), label=2)
    synth_phish = remove_leakage(synth_phish, real_phish, threshold=0.85)
    if len(synth_phish) > TARGET_SYNTH_PHISH_SIZE:
        synth_phish = synth_phish.sample(n=TARGET_SYNTH_PHISH_SIZE, random_state=SEED).reset_index(drop=True)

    # 3. Загрузка и очистка реальных и синтетических пулов Ham (0) и Spam (1)
    real_ham = clean_pool(pd.read_csv(REAL_HAM_FILE), label=0)
    real_spam = clean_pool(pd.read_csv(REAL_SPAM_FILE), label=1)

    synth_ham = clean_pool(pd.read_csv(SYNTH_HAM_FILE), label=0)
    synth_spam = clean_pool(pd.read_csv(SYNTH_SPAM_FILE), label=1)

    if len(synth_ham) > TARGET_SYNTH_BASE_SIZE:
        synth_ham = synth_ham.sample(n=TARGET_SYNTH_BASE_SIZE, random_state=SEED).reset_index(drop=True)
    if len(synth_spam) > TARGET_SYNTH_BASE_SIZE:
        synth_spam = synth_spam.sample(n=TARGET_SYNTH_BASE_SIZE, random_state=SEED).reset_index(drop=True)

    # 4. Сохранение подготовленных единых пулов в data/processed/ для 07_train_rubert_cv.py
    real_ham.to_csv(OUT_REAL_HAM, index=False, encoding="utf-8")
    real_spam.to_csv(OUT_REAL_SPAM, index=False, encoding="utf-8")
    real_phish.to_csv(OUT_REAL_PHISH, index=False, encoding="utf-8")

    synth_ham.to_csv(OUT_SYNTH_HAM, index=False, encoding="utf-8")
    synth_spam.to_csv(OUT_SYNTH_SPAM, index=False, encoding="utf-8")
    synth_phish.to_csv(OUT_SYNTH_PHISH, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("ПОДГОТОВЛЕННЫЕ ЕДИНЫЕ ПУЛЫ ДЛЯ КРОСС-ВАЛИДАЦИИ СОХРАНЕНЫ:")
    print(f"  Real Ham Pool:        {OUT_REAL_HAM} ({len(real_ham)} строк)")
    print(f"  Real Spam Pool:       {OUT_REAL_SPAM} ({len(real_spam)} строк)")
    print(f"  Real Phishing Pool:   {OUT_REAL_PHISH} ({len(real_phish)} строк)")
    print(f"  Synth Ham Pool:       {OUT_SYNTH_HAM} ({len(synth_ham)} строк)")
    print(f"  Synth Spam Pool:      {OUT_SYNTH_SPAM} ({len(synth_spam)} строк)")
    print(f"  Synth Phishing Pool:  {OUT_SYNTH_PHISH} ({len(synth_phish)} строк)")
    print("=" * 60)


if __name__ == "__main__":
    main()
