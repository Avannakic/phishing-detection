import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- НАСТРОЙКА ДИРЕКТОРИЙ ---
INTERIM_DIR = os.path.join("data", "interim")
PROCESSED_DIR = os.path.join("data", "processed")
REAL_DIR = os.path.join("data", "real")

os.makedirs(PROCESSED_DIR, exist_ok=True) 

# Синтетические файлы из предыдущих шагов
SYNTH_PHISH_FILE = os.path.join(INTERIM_DIR, "final_synthetic_phishing.csv")
SYNTH_HAM_FILE = os.path.join(INTERIM_DIR, "synthetic_ham.csv")
SYNTH_SPAM_FILE = os.path.join(INTERIM_DIR, "synthetic_spam.csv")

# Реальные данные
REAL_HAM_TRAIN_FILE = os.path.join(REAL_DIR, "real_ham.csv")
REAL_SPAM_TRAIN_FILE = os.path.join(REAL_DIR, "real_spam.csv")
REAL_PHISH_FILE = os.path.join(REAL_DIR, "real_phishing.csv") # Единый файл с фишингом

# Выходные файлы для обучения моделей
OUTPUT_REAL_TRAIN = os.path.join(PROCESSED_DIR, "rubert_train_real_final.csv")
OUTPUT_SYNTH_TRAIN = os.path.join(PROCESSED_DIR, "rubert_train_synth_final.csv")
OUTPUT_HYBRID_TRAIN = os.path.join(PROCESSED_DIR, "rubert_train_hybrid_final.csv")
OUTPUT_TEST_FILE = os.path.join(PROCESSED_DIR, "rubert_test_final.csv") # Файл для финальной проверки

# Флаг выравнивания размера базовых классов (Ham и Spam) между моделями
EQUALIZE_BASE_CLASSES = True
TARGET_BASE_CLASS_SIZE = 10000
SEED = 42


def remove_leakage(df_synth: pd.DataFrame, df_real_test: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Двухэтапная фильтрация точных и нечетких пересечений с тестовым набором."""
    print("[INFO] Проверка пересечений между синтетикой и реальным тестом (Data Leakage Check)...")

    real_norm = set(df_real_test["message"].astype(str).str.lower().str.strip())
    synth_norm = df_synth["message"].astype(str).str.lower().str.strip()

    # 1. Точные совпадения
    exact_mask = synth_norm.isin(real_norm)
    df_clean = df_synth[~exact_mask].reset_index(drop=True)
    print(f"  -> Удалено точных совпадений: {exact_mask.sum()}")

    # 2. Нечеткие дубликаты (TF-IDF на униграммах и биграммах)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_real = vectorizer.fit_transform(df_real_test["message"].astype(str))
    tfidf_synth = vectorizer.transform(df_clean["message"].astype(str))

    sim_matrix = cosine_similarity(tfidf_synth, tfidf_real)
    max_sim = np.max(sim_matrix, axis=1)
    near_dup_mask = max_sim > threshold

    print(f"  -> Максимальное косинусное сходство TF-IDF: {np.max(max_sim):.4f}")
    print(f"  -> Удалено нечетких дубликатов (сходство > {threshold}): {near_dup_mask.sum()}")

    return df_clean[~near_dup_mask].reset_index(drop=True)


def main():
    # 1. Загрузка и разделение РЕАЛЬНОГО фишинга (80% в Train, 20% в строгий Test)
    real_phish_full = pd.read_csv(REAL_PHISH_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    real_phish_full["label"] = 2

    real_phish_train = real_phish_full.sample(frac=0.8, random_state=SEED)
    real_phish_test = real_phish_full.drop(real_phish_train.index).reset_index(drop=True)
    
    # Сохраняем 20% в отдельный файл. На нем вы будете проверять финальные метрики всех моделей
    real_phish_test.to_csv(OUTPUT_TEST_FILE, index=False, encoding="utf-8")
    print(f"[INFO] Реальный фишинг разделен: {len(real_phish_train)} на трейн, {len(real_phish_test)} на тест (сохранен в {OUTPUT_TEST_FILE})")

    # 2. Загрузка синтетического фишинга и проверка на утечки ТОЛЬКО по тестовой части (20%)
    synth_phish = pd.read_csv(SYNTH_PHISH_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    synth_phish = remove_leakage(synth_phish, real_phish_test, threshold=0.85)
    synth_phish["label"] = 2

    # 3. Загрузка синтетических и реальных Ham (0) и Spam (1)
    synth_ham = pd.read_csv(SYNTH_HAM_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    synth_spam = pd.read_csv(SYNTH_SPAM_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    synth_ham["label"] = 0
    synth_spam["label"] = 1

    real_ham = pd.read_csv(REAL_HAM_TRAIN_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    real_spam = pd.read_csv(REAL_SPAM_TRAIN_FILE).dropna(subset=["message"]).drop_duplicates(subset=["message"])
    real_ham["label"] = 0
    real_spam["label"] = 1

    # 4. Опциональное выравнивание объемов базовых классов между моделями
    if EQUALIZE_BASE_CLASSES:
        n_ham = min(TARGET_BASE_CLASS_SIZE, len(synth_ham), len(real_ham))
        n_spam = min(TARGET_BASE_CLASS_SIZE, len(synth_spam), len(real_spam))
        synth_ham = synth_ham.sample(n=n_ham, random_state=SEED)
        real_ham = real_ham.sample(n=n_ham, random_state=SEED)
        synth_spam = synth_spam.sample(n=n_spam, random_state=SEED)
        real_spam = real_spam.sample(n=n_spam, random_state=SEED)
        print(f"[INFO] Базовые классы выровнены: Ham = {n_ham}, Spam = {n_spam}")

    # 5. Сборка выборок
    
    # Real Only: Реальные Ham, Spam + 80% тренировочного реального фишинга
    df_real_only = pd.concat([real_ham, real_spam, real_phish_train], ignore_index=True)
    df_real_only = df_real_only.drop_duplicates(subset=["message"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_real_only.to_csv(OUTPUT_REAL_TRAIN, index=False, encoding="utf-8")

    # Synth Only: Синтетические Ham, Spam, Phishing
    df_synth_only = pd.concat([synth_phish, synth_spam, synth_ham], ignore_index=True)
    df_synth_only = df_synth_only.drop_duplicates(subset=["message"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_synth_only.to_csv(OUTPUT_SYNTH_TRAIN, index=False, encoding="utf-8")

    # Hybrid: Реальные Ham/Spam + Синтетический фишинг + 80% тренировочного реального фишинга
    df_hybrid = pd.concat([synth_phish, real_phish_train, real_spam, real_ham], ignore_index=True)
    df_hybrid = df_hybrid.drop_duplicates(subset=["message"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_hybrid.to_csv(OUTPUT_HYBRID_TRAIN, index=False, encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"Test File сохранен:            {OUTPUT_TEST_FILE} ({len(real_phish_test)} строк фишинга)")
    print(f"Real Base Train сохранен:      {OUTPUT_REAL_TRAIN} ({len(df_real_only)} строк)")
    print(f"Synthetic Only Train сохранен: {OUTPUT_SYNTH_TRAIN} ({len(df_synth_only)} строк)")
    print(f"Hybrid Train сохранен:         {OUTPUT_HYBRID_TRAIN} ({len(df_hybrid)} строк)")
    print("=" * 50)


if __name__ == "__main__":
    main()