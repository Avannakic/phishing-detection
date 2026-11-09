#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Шаг 02: целевая догенерация синтетического фишинга (6 000 строк)
через локальный Ollama с автоматическим переключением на резервную модель при отказах.
"""

import argparse
import json
import os
import random
import re
import string
from collections import defaultdict, deque
from pathlib import Path

import ollama
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenarios_config.json"

MODEL_NAME = "qwen3:14b"
FALLBACK_MODEL = "vikhr-llama3.1-8b"
SEED = 42
REFUSAL_WINDOW_SIZE = 20
REFUSAL_RATE_THRESHOLD = 0.65
MAX_CONSECUTIVE_REFUSALS = 3
MIN_RESPONSE_LENGTH = 25
MAX_SAME_PREFIX = 3

REFUSAL_MARKERS = (
    "я не могу", "извините", "как ии", "не могу помочь",
    "мошеннич", "незаконн", "as an ai", "i cannot",
)

LEN_SHORT = ("короткие", 10, "1-2 предложения (до 150 символов)")
LEN_MED = ("средние", 4, "3-5 предложений (до 300 символов)")
LEN_LONG = ("длинные", 2, "развернутый текст (около 500-600 символов)")

DEFAULT_SCENARIOS = {
    "scenarios": {
        "службы безопасности финансовой организации": [
            "угроза блокировки операции по счету",
            "уведомление о начислении бонусных баллов по ссылке",
        ],
        "знакомого или коллеги": [
            "просьба поддержать ребенка в онлайн-конкурсе по ссылке",
            "кликбейтное сообщение с призывом посмотреть фото по ссылке",
        ],
        "бота раздачи премиум-подписок": [
            "обещание бесплатной подписки на год по акционной ссылке",
            "срочное уведомление об окончании раздачи подарков через час",
        ],
        "сервиса раздачи цифровых ключей и бонусов": [
            "раздача бесплатных промокодов и токенов для онлайн-сервисов",
            "секретная распродажа со скидкой 90% по внешней ссылке",
        ],
    },
    "emoji_rules": {
        "службы безопасности финансовой организации": (0.15, "Используй 1-2 сдержанных эмодзи, характерных для уведомлений."),
        "знакомого или коллеги": (0.30, "Можешь использовать обычные бытовые эмодзи."),
        "бота раздачи премиум-подписок": (0.80, "Обильно используй яркие, кликбейтные эмодзи (🔥, 🎁, 🎉, ⚡️)."),
        "сервиса раздачи цифровых ключей и бонусов": (0.70, "Используй кликбейтные эмодзи, связанные с IT и выгодой (🤖, 🚀, 😱, 🔑)."),
    },
    "allowed_lengths": {
        "службы безопасности финансовой организации": [LEN_MED, LEN_MED, LEN_LONG],
        "знакомого или коллеги": [LEN_SHORT, LEN_SHORT, LEN_MED],
        "бота раздачи премиум-подписок": [LEN_SHORT, LEN_SHORT, LEN_MED],
        "сервиса раздачи цифровых ключей и бонусов": [LEN_SHORT, LEN_SHORT, LEN_MED],
    },
    "link_examples": "t.me/secure_bot, vk.cc/hG8d, my-portal-auth.ru",
}

BAD_STARTS = (
    "я не могу", "i cannot", "к сожалению", "однако", "извините", "простите",
    "as an ai", "сгенерированное сообщение", "here is", "here are", "here's",
    "sure", "certainly", "of course", "below", "the following", "вот ", "конечно",
    "уважаемый", "уважаемая", "здравствуй",
    "эти примеры", "эти сообщения", "каждое сообщение", "ниже представлен", "данные примеры",
)
BAD_WORDS = ["example", "test", "sample", "insert", "вставь", "[ссылка", "phishing", "фишинг", "пример"]


def load_scenarios(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Загружает сценарии целевого фишинга из внешнего JSON или возвращает DEFAULT_SCENARIOS."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        section = data.get("targeted_phishing")
        if section and "scenarios" in section and "emoji_rules" in section and "allowed_lengths" in section:
            return {
                "scenarios": {k: list(v) for k, v in section["scenarios"].items()},
                "emoji_rules": {k: (float(v[0]), str(v[1])) for k, v in section["emoji_rules"].items()},
                "allowed_lengths": {
                    k: [tuple(item) for item in v]
                    for k, v in section["allowed_lengths"].items()
                },
                "link_examples": section.get("link_examples", DEFAULT_SCENARIOS["link_examples"]),
            }
    print(f"[INFO] Конфиг {path} не найден или неполон, используется DEFAULT_SCENARIOS.")
    return DEFAULT_SCENARIOS


def get_style_instruction(sender: str) -> str:
    sender_lower = sender.lower()
    if any(k in sender_lower for k in ("родственник", "знаком", "коллег", "ключ", "нейросет", "чек")):
        return (
            "Пиши разговорным языком. Используй сленг (щас, спс, че), "
            "просторечия и естественные опечатки. Избегай литературности!"
        )
    if any(k in sender_lower for k in ("крупного банка", "финансовой организации", "telegram", "подписк")):
        return "Пиши как агрессивный маркетолог или бот. Коротко, напористо, с кликбейтом."
    return "Пиши строгим, сухим канцелярским языком."


def is_refusal(text: str) -> bool:
    clean_text = text.strip().lower()
    return len(clean_text) < MIN_RESPONSE_LENGTH or any(
        marker in clean_text for marker in REFUSAL_MARKERS
    )


class OllamaGenerator:
    def __init__(self):
        self.current_model = MODEL_NAME
        self.refusal_window = deque(maxlen=REFUSAL_WINDOW_SIZE)
        self.consecutive_refusals = 0

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.generate(
                model=self.current_model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.85,
                    "top_p": 0.92,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_predict": 350,
                },
            )
            text = response["response"]
        except Exception as exc:
            print(f"[Ошибка Ollama, {self.current_model}]: {exc}")
            text = ""

        refused = is_refusal(text)
        self.refusal_window.append(refused)
        self.consecutive_refusals = self.consecutive_refusals + 1 if refused else 0

        if refused:
            print(f"[INFO] {self.current_model} выдал отказ (сработала цензура). Повторяем попытку...")

        rate_exceeded = (
            len(self.refusal_window) == REFUSAL_WINDOW_SIZE
            and sum(self.refusal_window) / REFUSAL_WINDOW_SIZE > REFUSAL_RATE_THRESHOLD
        )
        if self.current_model == MODEL_NAME and (
            rate_exceeded or self.consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS
        ):
            print(
                "[WARN] Высокий процент отказов Qwen3-14B (>65%). "
                "Переключение генератора на VIKHR..."
            )
            self.current_model = FALLBACK_MODEL
            self.consecutive_refusals = 0

        return "" if refused else text


def main():
    parser = argparse.ArgumentParser(description="Шаг 02: целевая догенерация синтетического фишинга")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Путь к внешнему JSON-конфигу со сценариями",
    )
    parser.add_argument("--target_count", type=int, default=6000, help="Целевое число сообщений")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "interim"),
        help="Директория сохранения",
    )
    args = parser.parse_args()

    random.seed(SEED)
    target_count = args.target_count
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    progress_file = os.path.join(out_dir, "synthetic_phishing_v2_progress.csv")

    cfg = load_scenarios(args.config)
    scenarios = cfg["scenarios"]
    emoji_rules = cfg["emoji_rules"]
    allowed_lengths = cfg["allowed_lengths"]
    link_examples = cfg.get("link_examples", DEFAULT_SCENARIOS["link_examples"])

    valid_combinations = []
    for sender, tactics in scenarios.items():
        for tactic in tactics:
            for length in allowed_lengths.get(sender, [LEN_MED]):
                valid_combinations.append((sender, tactic, length))

    random.shuffle(valid_combinations)

    dataset = set()
    if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0:
        try:
            df_prog = pd.read_csv(progress_file)
            if "message" in df_prog.columns:
                dataset = set(df_prog["message"].dropna().astype(str).tolist())
                print(f"Загружено {len(dataset)} строк из {progress_file}.")
        except Exception as exc:
            print(f"[WARN] Не удалось прочитать чекпоинт {progress_file}: {exc}")

    combo_idx = 0
    requests_made = 0
    generator = OllamaGenerator()

    print(f"Всего доступно уникальных комбинаций промпта: {len(valid_combinations)}")
    print(f"Начинаем генерацию {target_count} сообщений с модели {generator.current_model}...")

    prefix_counts = defaultdict(int)
    if len(dataset) > 0:
        for line in dataset:
            clean_words = line.translate(str.maketrans("", "", string.punctuation)).lower().split()
            if len(clean_words) >= 2:
                pref = f"{clean_words[0]} {clean_words[1]}"
                prefix_counts[pref] += 1

    try:
        while len(dataset) < target_count:
            sender, tactic, (length_type, count, length_desc) = valid_combinations[combo_idx]

            emoji_chance, emoji_prompt = emoji_rules.get(
                sender, (0.0, "Пиши СТРОГО БЕЗ эмодзи.")
            )
            if random.random() < emoji_chance:
                emoji_instruction = emoji_prompt
            else:
                emoji_instruction = "Пиши СТРОГО БЕЗ эмодзи."

            if random.random() < 0.5:
                link_instruction = (
                    f"ОБЯЗАТЕЛЬНО органично вставь в текст правдоподобную ссылку "
                    f"(например {link_examples}). "
                    f"КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать слова example, test, sample или плейсхолдеры."
                )
            else:
                link_instruction = "Не вставляй никаких ссылок, просто призывай к действию или пугай."

            style_instruction = get_style_instruction(sender)

            prompt = (
                f"Ты — AI-ассистент исследователя в области информационной безопасности. "
                f"Для обучения защитной антиспам-системы (NLP классификатора) мне нужны "
                f"СИНТЕТИЧЕСКИЕ ПРИМЕРЫ текстов, используемых злоумышленниками. "
                f"Сгенерируй {count} {length_type} учебных примеров социальной инженерии "
                f"для мессенджеров ({length_desc}) от лица: {sender}. Тематика атаки: {tactic}. \n"
                f"ПРАВИЛА:\n"
                f"1. {style_instruction}\n"
                f"2. Пиши сразу примеры текстов. Не пиши вводных слов. Встраивай контекст отправителя естественно!\n"
                f"3. {emoji_instruction}\n"
                f"4. {link_instruction}\n"
                f"5. Пиши максимально правдоподобно для РФ. Не обрывай предложения!\n"
                f"6. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать приветствия (Уважаемый пользователь, Здравствуйте, Добрый день и т.д.). Начинай сразу с сути или с призыва к действию!\n"
                f"ВЫВЕДИ ТОЛЬКО СООБЩЕНИЯ. Каждое сообщение с новой строки. Без нумерации и кавычек."
            )

            text = generator.generate(prompt)
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            for line in lines:
                if "Here" in line or "Вот" in line or "Сгенерирован" in line:
                    continue

                clean_line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

                # Жестко отрезаем типичные приветствия LLM в начале строки
                clean_line = re.sub(
                    r"^(уважаемый(ая)?\s+пользователь(!|,|\.)?|здравствуйте(!|,|\.)?|добрый\s+день(!|,|\.)?)\s*",
                    "",
                    clean_line,
                    flags=re.IGNORECASE,
                ).strip()

                if not clean_line:
                    continue

                if clean_line.startswith('"') and clean_line.endswith('"'):
                    clean_line = clean_line[1:-1]

                clean_lower = clean_line.lower()
                if len(clean_line) < 25:
                    continue
                if len(re.findall(r"[А-Яа-яЁё]", clean_line)) < 15:
                    continue

                if clean_lower.startswith(BAD_STARTS):
                    if ":" in clean_line or len(clean_line) < 100:
                        continue

                if any(bw in clean_lower for bw in BAD_WORDS):
                    continue

                # Трекер одинаковых начал предложений (Bigram Tracker)
                words_for_prefix = clean_lower.translate(str.maketrans("", "", string.punctuation)).split()
                if len(words_for_prefix) >= 2:
                    prefix = f"{words_for_prefix[0]} {words_for_prefix[1]}"
                    if prefix_counts[prefix] >= MAX_SAME_PREFIX:
                        continue
                    prefix_counts[prefix] += 1

                dataset.add(clean_line)

            requests_made += 1
            combo_idx = (combo_idx + 1) % len(valid_combinations)

            if requests_made % 10 == 0:
                print(
                    f"Запросов: {requests_made}. Собрано: {len(dataset)}/{target_count}. "
                    f"Модель: {generator.current_model}"
                )

            if requests_made % 50 == 0:
                pd.DataFrame(sorted(dataset), columns=["message"]).to_csv(
                    progress_file, index=False, encoding="utf-8"
                )
                print("Промежуточное сохранение выполнено.")

    except KeyboardInterrupt:
        print(f"\n\n[!] Генерация прервана пользователем. Сохраняем собранные {len(dataset)} строк...")

    if len(dataset) > 0:
        final_filename = os.path.join(out_dir, "raw_synthetic_targeted.csv")
        final_list = sorted(dataset)[:target_count]
        pd.DataFrame(final_list, columns=["message"]).to_csv(final_filename, index=False, encoding="utf-8")
        pd.DataFrame(final_list, columns=["message"]).to_csv(progress_file, index=False, encoding="utf-8")

        if len(dataset) >= target_count and os.path.exists(progress_file):
            os.remove(progress_file)

        print(f"\nГотово! Сгенерировано: {len(final_list)} уникальных сообщений. Файл сохранен как {final_filename}")
    else:
        print("\nДанных не собрано, сохранять нечего.")


if __name__ == "__main__":
    main()
