import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

base_phishing_file = os.path.join(DATA_DIR, "cleaned_synthetic_base.csv")
targeted_phishing_file = os.path.join(DATA_DIR, "cleaned_synthetic_targeted.csv")
output_file = os.path.join(DATA_DIR, "merged_synthetic_phishing.csv")

print("[INFO] Загрузка очищенных датасетов...")
df_base = pd.read_csv(base_phishing_file)
df_targeted = pd.read_csv(targeted_phishing_file)

print(f"В базовой выборке:  {len(df_base)} сообщений")
print(f"В целевой выборке:  {len(df_targeted)} сообщений")

# 1. Склеиваем оба датафрейма
df_combined = pd.concat([df_base, df_targeted], ignore_index=True)

# 2. Чистим от пустых строк и возможных межитерационных дубликатов
df_combined = df_combined.dropna(subset=["message"])
df_combined = df_combined.drop_duplicates(subset=["message"])

# 3. Перемешиваем (random_state зафиксирован для воспроизводимости)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("-" * 45)
print(f"Итоговый размер объединенной базы: {len(df_combined)} уникальных сообщений")
print("-" * 45)

df_combined.to_csv(output_file, index=False, encoding="utf-8")
print(f"[INFO] Сохранено в: {output_file}")