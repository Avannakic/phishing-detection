import re
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

FILES_TO_CLEAN = [
    (os.path.join(DATA_DIR, "raw_synthetic_base.csv"), os.path.join(DATA_DIR, "cleaned_synthetic_base.csv")),
    (os.path.join(DATA_DIR, "raw_synthetic_targeted.csv"), os.path.join(DATA_DIR, "cleaned_synthetic_targeted.csv")),
]

BAD_STARTS = (
    "я не могу", "i cannot", "к сожалению", "однако", "извините", "простите",
    "as an ai", "я обязан", "сообщение", "сгенерированное сообщение",
    "here is", "here are", "here's", "sure", "certainly", "of course",
    "below", "the following", "вот ", "конечно", "настоящим", "в качестве", "here"
)

# 'example' и '[ссылка' оставлены для процедурной замены на URL в Шаге 4
BAD_WORDS = [
    "test", "sample", "insert", "вставь",
    "phishing", "фишинг", "пример", "prompt", "промпт"
]

BAD_BRACKET_PATTERN = re.compile(r"\[[^\]]*(?:вредоносн|malicious|фишинг)[^\]]*\]", re.IGNORECASE)


def is_valid_russian_phishing(text: str) -> bool:
    if not isinstance(text, str):
        return False
    clean_line = text.strip()
    clean_lower = clean_line.lower()

    if len(clean_line) < 25 or clean_line.startswith("*"):
        return False
    if "емиас" in clean_lower:
        return False

    ru_chars = len(re.findall(r"[А-Яа-яЁё]", clean_line))
    en_chars = len(re.findall(r"[A-Za-z]", clean_line))
    if ru_chars < 15 or en_chars > ru_chars:
        return False

    if clean_lower.startswith(BAD_STARTS) and (":" in clean_line or len(clean_line) < 100):
        return False
    if any(bw in clean_lower for bw in BAD_WORDS) or BAD_BRACKET_PATTERN.search(clean_line):
        return False

    return True


def process_dataset(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"[WARN] Файл {input_file} не найден, пропуск.")
        return
    df = pd.read_csv(input_file)
    initial_count = len(df)

    df_clean = df.dropna(subset=["message"]).copy()
    df_clean = df_clean[df_clean["message"].apply(is_valid_russian_phishing)]
    df_clean = df_clean.drop_duplicates(subset=["message"])
    df_clean = df_clean.sort_values(by="message", key=lambda x: x.str.len()).reset_index(drop=True)

    df_clean.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[INFO] {input_file} -> {output_file} | Исходно: {initial_count}, Осталось: {len(df_clean)}")


if __name__ == "__main__":
    for inp, out in FILES_TO_CLEAN:
        process_dataset(inp, out)