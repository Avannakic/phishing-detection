#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 03: Deep regex-based line-by-line filtering of synthetic pools
(Phishing Base, Phishing Targeted, Spam, Ham), removing legacy URLs,
LLM meta-comments, greetings, placeholders, and truncated sentences.
"""

import os
import re
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

# (input_file, output_file, label_val)
FILES_TO_CLEAN = [
    (
        os.path.join(DATA_DIR, "raw_synthetic_base.csv"),
        os.path.join(DATA_DIR, "cleaned_synthetic_base.csv"),
        2,
    ),
    (
        os.path.join(DATA_DIR, "raw_synthetic_targeted.csv"),
        os.path.join(DATA_DIR, "cleaned_synthetic_targeted.csv"),
        2,
    ),
    (
        os.path.join(DATA_DIR, "synthetic_spam.csv"),
        os.path.join(DATA_DIR, "cleaned_synthetic_spam.csv"),
        1,
    ),
    (
        os.path.join(DATA_DIR, "synthetic_ham.csv"),
        os.path.join(DATA_DIR, "cleaned_synthetic_ham.csv"),
        0,
    ),
]

# 1. Complete stripping of any pre-existing URLs (including Markdown and concatenated links)
FULL_URL_STRIP = re.compile(
    r"\[([^\]]+)\]\([^\)]+\)|"
    r"\S*(?:https?://|t\.me/|vk\.cc/|bit\.ly/|clck\.ru/|goo\.gl/|youtu\.be/)\S*|"
    r"\b[a-z0-9_\-]+\.(?:ru|com|net|org|su|рф|io|me|cc|online|store|tech|pro|center|today|ly|gl|be)(?:/\S*)?",
    flags=re.IGNORECASE,
)

# 2. Unfilled LLM placeholders and Cyrillic character checks
PLACEHOLDER_PATTERN = re.compile(r"[\[\]<>{}]|example\.com", flags=re.IGNORECASE)
BAD_BRACKET_PATTERN = re.compile(r"\[[^\]]*(?:вредоносн|malicious|фишинг)[^\]]*\]", flags=re.IGNORECASE)
HAS_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")

# 3. LLM meta-comments, prompts, and synthetic generation artifacts
META_PATTERNS = re.compile(
    r"(###|\bпример\w*\s*\d|\bобразец\s*\d|\bпримечание|\bконец\s+примеров|\bот\s+лица\s|"
    r"\bдля\s+учебн|\bсгенерирован|\bсоциальн\w+\s+инженер|\bантиспам|\bзлоумышленник|"
    r"\bдатасет|\bальтернативный\s+пример|\bанонимное\s+сообщение|\bтексты\s+создан|"
    r"\bсообщения\s+создан|\bпримеры\s+создан|\bсозданы\s+с\s+целью|\bв\s+соответствии\s+с\s+правилами|"
    r"\bзащитить\s+себя\s+от\s+фишинг|\bбезопасность\s+начинается\s+с\s+вас|\bоформлены\s+максимально\s+правдоподобно)",
    flags=re.IGNORECASE,
)

# 4. Formatting junk, greetings, and sender headers
LEAD_JUNK = re.compile(r"^[\s\"\'\`\*\#\-\–\—\•\d\.\)\:]+")
MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")

PREFIX_STRIP = re.compile(
    r"^(?:[^\wА-Яа-яЁё]*"
    r"(?:уважаем\w+\s+(?:пользовател\w+|клиент\w*|студент\w*|коллег\w+|граждан\w*|геймер\w*|друз\w+|читател\w+|партнер\w*|пациент\w*)|"
    r"здравствуйте|здравствуй|приветствую(?:\s+вас)?|привет(?:ик)?|"
    r"добрый\s+(?:день|вечер)|доброго\s+времени(?:\s+суток)?|"
    r"внимание|срочно(?:е\s+уведомление|\s+сообщение)?|важно(?:е\s+уведомление|\s+сообщение|\s+для\s+ваших\s+счетов)?|"
    r"(?:банк\s+)?(?:втб|сбербанк|сбер|т-банк|т\s+банк|фсб(?:\s+рф|\s+россии)?|цб\s+рф|госуслуги|емиас|telegram)\s*[\:\—\-])"
    r"[!,\.\:\—\-\s]*)+",
    flags=re.IGNORECASE,
)

# 5. Truncated sentence endings (evaluated after URL stripping)
TRUNCATED_END = re.compile(
    r"(\b(и|в|на|с|по|для|от|к|о|из|за|при|до|без|через|или|что|как|но|а|причём|также|ин|зад|об|провер)|[,:;—\-])\s*$",
    flags=re.IGNORECASE,
)

BAD_STARTS = (
    "я не могу", "i cannot", "к сожалению", "однако", "извините", "простите",
    "as an ai", "я обязан", "сгенерированное сообщение",
    "here is", "here are", "sure", "certainly", "of course",
    "вот ", "конечно", "настоящим", "в качестве",
)

BAD_WORDS = [
    "test", "sample", "insert", "вставь",
    "phishing", "фишинг", "пример", "prompt", "промпт",
]


def clean_and_validate_message(raw_text: str, label_val: int) -> Optional[str]:
    if not isinstance(raw_text, str):
        return None
    raw_stripped = raw_text.strip()
    if not raw_stripped:
        return None

    # 1. Filter out LLM meta-comments, example.com placeholders, and bracketed malicious labels
    if (
        META_PATTERNS.search(raw_stripped)
        or BAD_BRACKET_PATTERN.search(raw_stripped)
        or re.search(r"\bexample\.(?:com|ru|org|net)\b", raw_stripped, flags=re.IGNORECASE)
    ):
        return None

    # 2. Strip Markdown bold formatting and remove pre-existing URLs
    line = MD_BOLD.sub(r"\1", raw_stripped).strip()
    if not line or line.startswith("*"):
        return None

    had_url = bool(FULL_URL_STRIP.search(line))
    line = FULL_URL_STRIP.sub("", line)

    # 3. Validate placeholders, language balance, and stop-words after URL stripping
    if PLACEHOLDER_PATTERN.search(line) or BAD_BRACKET_PATTERN.search(line) or not HAS_CYRILLIC.search(line):
        return None

    ru_chars = len(re.findall(r"[А-Яа-яЁё]", line))
    en_chars = len(re.findall(r"[A-Za-z]", line))
    min_ru_chars = 10 if label_val == 0 else 15
    if ru_chars < min_ru_chars or en_chars > ru_chars:
        return None

    line_lower_pre = line.strip().lower()
    if any(bw in line_lower_pre for bw in BAD_WORDS):
        return None
    if label_val == 2:
        if "емиас" in line_lower_pre or line_lower_pre.startswith(BAD_STARTS):
            return None

    # 4. Clean prefix junk, quotes, greetings, and sender headers
    line = LEAD_JUNK.sub("", line).strip()
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()
    elif line.endswith('"') or line.endswith("'"):
        line = line[:-1].strip()

    line = PREFIX_STRIP.sub("", line).strip()
    line = LEAD_JUNK.sub("", line).strip()
    line = re.sub(r"\s+", " ", line).strip()

    if had_url:
        line = re.sub(r"[\s:\-—]+$", "", line).strip()

    if not line:
        return None

    clean_lower = line.lower()
    if any(bw in clean_lower for bw in BAD_WORDS):
        return None
    if label_val == 2:
        if "емиас" in clean_lower or clean_lower.startswith(BAD_STARTS):
            return None

    min_len = 16 if label_val == 0 else 25
    if len(line) < min_len or TRUNCATED_END.search(line):
        return None

    words = re.findall(r"[а-яёa-z0-9]+", clean_lower)
    min_words = 2 if label_val == 0 else 4
    if len(words) < min_words:
        return None

    if line[0].islower() and line[0].isalpha():
        line = line[0].upper() + line[1:]

    return line


def process_dataset(input_file: str, output_file: str, label_val: int):
    if not os.path.exists(input_file):
        print(f"[WARN] File {input_file} not found; skipping.")
        return

    df = pd.read_csv(input_file)
    col = "message" if "message" in df.columns else "text"
    raw_msgs = df[col].dropna().astype(str).tolist()
    initial_count = len(raw_msgs)

    cleaned_msgs = []
    seen = set()
    for msg in raw_msgs:
        cleaned = clean_and_validate_message(msg, label_val=label_val)
        if cleaned is not None and cleaned not in seen:
            seen.add(cleaned)
            cleaned_msgs.append(cleaned)

    df_clean = pd.DataFrame({
        "message": cleaned_msgs,
        "label": label_val,
    })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False, encoding="utf-8")
    print(
        f"[INFO] [Class {label_val}] {input_file} -> {output_file} | "
        f"Initial: {initial_count}, Cleaned: {len(df_clean)}"
    )


if __name__ == "__main__":
    for inp_path, out_path, lbl in FILES_TO_CLEAN:
        process_dataset(inp_path, out_path, lbl)
