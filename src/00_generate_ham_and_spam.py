#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Шаг 00: генерация синтетических фоновых классов 0 (Ham) и 1 (Spam)
через локальный Ollama с автоматическим fallback при отказах Qwen3.
"""

import argparse
import json
import os
import random
import re
from collections import deque
from pathlib import Path

import ollama
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenarios_config.json"

MODEL_NAME = "qwen3:14b"
FALLBACK_MODEL = "vikhr-llama3.1-8b"
TEMPERATURE = 0.8
SEED = 42
REFUSAL_WINDOW_SIZE = 20
REFUSAL_RATE_THRESHOLD = 0.65
MAX_CONSECUTIVE_REFUSALS = 3
MIN_RESPONSE_LENGTH = 25
REFUSAL_MARKERS = (
    "я не могу", "извините", "как ии", "не могу помочь",
    "мошеннич", "незаконн", "as an ai", "i cannot",
)

DEFAULT_SCENARIOS = {
    "ham": {
        "topics_data": [
            (
                "Обсуждение бытовых покупок и списка продуктов в семейном чате",
                ["спокойствие", "забота", "спешка"],
                "перечисление списка покупок и сравнение цен в магазинах",
            ),
            (
                "Обсуждение подготовки к учебному семинару или рабочему проекту",
                ["сосредоточенность", "усталость", "энтузиазм"],
                "согласование сроков сдачи отчета и распределение задач",
            ),
            (
                "Обсуждение планов на выходные и поездки за город",
                ["радость", "предвкушение", "задумчивость"],
                "выбор маршрута поездки и обсуждение прогноза погоды",
            ),
            (
                "Обсуждение прочитанной книги или нового фильма с другом",
                ["интерес", "восхищение", "легкая критика"],
                "впечатления от сюжета и игры актеров",
            ),
        ],
        "lengths": [
            ("короткие", 7, "1-2 слова или одно короткое предложение (от 10 до 50 символов)", False),
            ("средние", 5, "1-2 полных предложения (до 150 символов)", False),
            ("длинные", 2, "развернутое сообщение с объяснением ситуации (около 300-500 символов)", True),
        ],
        "length_weights": [0.50, 0.45, 0.05],
        "styles": [
            "Пиши неформально, в стиле обычной переписки в мессенджере.",
            "Пиши сухо и по делу.",
            "Используй пару нейтральных эмодзи.",
            "Пиши коротко, как будто отвечаешь на ходу.",
        ],
    },
    "spam": {
        "topics_data": [
            (
                "Реклама образовательного онлайн-курса по программированию или дизайну",
                ["деловой маркетинг", "воодушевление"],
                "описание программы курса, скидки на обучение и помощь с портфолио",
            ),
            (
                "Рекламная рассылка интернет-магазина одежды и обуви о сезонной распродаже",
                ["акционный стиль", "выгода"],
                "промокод на скидку при заказе из новой коллекции и бесплатная доставка",
            ),
            (
                "Предложение услуг продвижения сайтов, настройки рекламы и веб-аналитики",
                ["деловое предложение", "навязчивость"],
                "обещание бесплатного аудита сайта и роста числа заявок",
            ),
            (
                "Реклама фитнес-клуба или студии йоги с приглашением на пробное занятие",
                ["бодрость", "мотивация"],
                "выгодный абонемент на полгода и подарок при записи сегодня",
            ),
        ],
        "lengths": [
            ("короткие", 8, "короткий рекламный призыв (до 80 символов)", False),
            ("средние", 5, "рекламное сообщение на 2-3 предложения (до 200 символов)", False),
            ("длинные", 2, "развернутый рекламный пост с описанием преимуществ (около 300-500 символов)", True),
        ],
        "length_weights": [0.20, 0.60, 0.20],
        "styles": [
            "Пиши ярко и продающе, используй рекламные эмодзи (🔥, 🚀, 💰, ⚡).",
            "Выделяй заглавными буквами ключевые слова (СКИДКА, БЕСПЛАТНО, АКЦИЯ).",
            "Используй рекламный заголовок, привлекающий внимание.",
            "Пиши сухо, в стиле автоматической коммерческой рассылки.",
        ],
        "anti_phishing_rule": (
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать про штрафы, блокировку аккаунтов, "
            "службу безопасности финансовой организации, государственные порталы и просить пароли/коды. "
            "Это должен быть коммерческий спам, а не фишинг."
        ),
    },
}

BAD_STARTS = ("я не ", "к сожалению", "вот ", "конечно", "сообщения:", "в качестве")


def load_scenarios(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Загружает конфигурацию сценариев из внешнего JSON или возвращает DEFAULT_SCENARIOS."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "ham" in data and "spam" in data:
            return {
                "ham": {
                    "topics_data": [
                        (item[0], list(item[1]), item[2])
                        for item in data["ham"]["topics_data"]
                    ],
                    "lengths": [tuple(item) for item in data["ham"]["lengths"]],
                    "length_weights": list(data["ham"]["length_weights"]),
                    "styles": list(data["ham"]["styles"]),
                },
                "spam": {
                    "topics_data": [
                        (item[0], list(item[1]), item[2])
                        for item in data["spam"]["topics_data"]
                    ],
                    "lengths": [tuple(item) for item in data["spam"]["lengths"]],
                    "length_weights": list(data["spam"]["length_weights"]),
                    "styles": list(data["spam"]["styles"]),
                    "anti_phishing_rule": data["spam"].get(
                        "anti_phishing_rule",
                        DEFAULT_SCENARIOS["spam"]["anti_phishing_rule"],
                    ),
                },
            }
    print(f"[INFO] Конфиг {path} не найден или неполон, используется DEFAULT_SCENARIOS.")
    return DEFAULT_SCENARIOS


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
                options={"temperature": TEMPERATURE},
            )
            text = response["response"]
        except Exception as exc:
            print(f"[Ошибка Ollama, {self.current_model}]: {exc}")
            text = ""

        refused = is_refusal(text)
        self.refusal_window.append(refused)
        self.consecutive_refusals = self.consecutive_refusals + 1 if refused else 0

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


def build_ham_prompt(
    base_topic: str,
    emotion: str,
    long_detail: str,
    length_type: str,
    count: int,
    length_desc: str,
    is_long: bool,
    style: str,
) -> str:
    prompt = (
        f"Сгенерируй {count} {length_type} сообщений от обычных людей в мессенджере (Telegram/WhatsApp). \n"
        f"Контекст переписки: {base_topic}. \n"
        f"Настроение/эмоция пишущего: {emotion}. \n"
        f"Длина каждого сообщения: {length_desc}. \n"
    )
    if is_long:
        prompt += f"В тексте сообщений можешь раскрыть эту деталь: {long_detail}. \n"
    prompt += (
        f"ПРАВИЛА:\n"
        f"1. {style}\n"
        f"2. Пиши максимально реалистично для современного русскоязычного сегмента.\n"
        f"3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать вводные слова от ИИ (такие как 'вот примеры'), нумерацию (1, 2, 3), кавычки или дефисы в начале строки.\n"
        f"Отвечай ТОЛЬКО самими сообщениями. Каждое сообщение с новой строки, без пустых строк между ними."
    )
    return prompt


def build_spam_prompt(
    base_topic: str,
    emotion: str,
    long_detail: str,
    length_type: str,
    count: int,
    length_desc: str,
    is_long: bool,
    style: str,
    anti_phishing_rule: str = DEFAULT_SCENARIOS["spam"]["anti_phishing_rule"],
) -> str:
    prompt = (
        f"Сгенерируй {count} {length_type} СПАМ-сообщений для рассылки в мессенджере (Telegram/WhatsApp). \n"
        f"Тема спама: {base_topic}. \n"
        f"Настроение/стиль: {emotion}. \n"
        f"Длина каждого сообщения: {length_desc}. \n"
    )
    if is_long:
        prompt += f"В тексте обязательно раскрой эту деталь: {long_detail}. \n"
    prompt += (
        f"ПРАВИЛА:\n"
        f"1. {style}\n"
        f"2. Пиши максимально реалистично для русского сегмента, это назойливая реклама.\n"
        f"3. {anti_phishing_rule}\n"
        f"4. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать вводные слова от ИИ, нумерацию (1, 2, 3), кавычки или дефисы в начале строки.\n"
        f"Отвечай ТОЛЬКО самими сообщениями. Каждое сообщение с новой строки, без пустых строк между ними."
    )
    return prompt


def generate_class_corpus(
    class_name: str,
    label_id: int,
    target_count: int,
    topics_data: list,
    lengths_config: list,
    length_weights: list,
    styles_list: list,
    prompt_builder_fn,
    output_path: str,
    progress_path: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = set()
    if os.path.exists(progress_path) and os.path.getsize(progress_path) > 0:
        try:
            df_prog = pd.read_csv(progress_path)
            if "message" in df_prog.columns:
                dataset = set(df_prog["message"].dropna().astype(str).to_list())
                print(f"[{class_name}] Загружено {len(dataset)} строк из чекпоинта {progress_path}")
        except Exception as exc:
            print(f"[WARN] Не удалось прочитать чекпоинт {progress_path}: {exc}")

    requests_made = 0
    generator = OllamaGenerator()
    print(
        f"[{class_name}] Старт генерации (цель: {target_count} уникальных сообщений, "
        f"модель: {generator.current_model})..."
    )

    try:
        while len(dataset) < target_count:
            topic_info = random.choice(topics_data)
            base_topic = topic_info[0]
            emotion = random.choice(topic_info[1])
            long_detail = topic_info[2]

            length_type, count, length_desc, is_long = random.choices(
                lengths_config, weights=length_weights, k=1
            )[0]
            style = random.choice(styles_list)

            prompt = prompt_builder_fn(
                base_topic, emotion, long_detail, length_type, count, length_desc, is_long, style
            )
            text = generator.generate(prompt)
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            for line in lines:
                clean_line = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
                if clean_line.startswith('"') and clean_line.endswith('"'):
                    clean_line = clean_line[1:-1]
                clean_lower = clean_line.lower()

                if len(clean_line) < 2:
                    continue
                if clean_lower.startswith(BAD_STARTS) or "here are" in clean_lower or "сгенерированные" in clean_lower:
                    continue

                en_chars = len(re.findall(r"[a-z]", clean_lower))
                ru_chars = len(re.findall(r"[а-яё]", clean_lower))
                if en_chars > ru_chars and ru_chars == 0:
                    continue

                dataset.add(clean_line)

            requests_made += 1
            if requests_made % 10 == 0:
                print(
                    f"[{class_name}] Запросов: {requests_made} | "
                    f"Собрано: {len(dataset)}/{target_count} | Модель: {generator.current_model}"
                )

            if requests_made % 100 == 0:
                pd.DataFrame({"message": sorted(dataset)}).to_csv(
                    progress_path, index=False, encoding="utf-8"
                )

    except KeyboardInterrupt:
        print(f"\n[{class_name}] Прервано пользователем. Сохраняем накопленные {len(dataset)} строк...")

    if len(dataset) > 0:
        final_list = sorted(dataset)[:target_count]
        pd.DataFrame({"message": final_list}).to_csv(progress_path, index=False, encoding="utf-8")
        df_out = pd.DataFrame({
            "message": final_list,
            "text": final_list,
            "label": label_id,
        })
        df_out.to_csv(output_path, index=False, encoding="utf-8")
        if len(dataset) >= target_count and os.path.exists(progress_path):
            os.remove(progress_path)
        print(f"[{class_name}] Готово! Сохранено {len(df_out)} строк в {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетических классов Ham (0) и Spam (1)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Путь к внешнему JSON-конфигу со сценариями",
    )
    parser.add_argument("--target_ham", type=int, default=14000, help="Целевое число сообщений Ham")
    parser.add_argument("--target_spam", type=int, default=14000, help="Целевое число сообщений Spam")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "interim"),
        help="Директория сохранения",
    )
    parser.add_argument("--mode", choices=["all", "ham", "spam"], default="all", help="Какой класс генерировать")
    args = parser.parse_args()

    scenarios_cfg = load_scenarios(args.config)
    ham_cfg = scenarios_cfg["ham"]
    spam_cfg = scenarios_cfg["spam"]

    random.seed(SEED)

    if args.mode in ("all", "ham"):
        generate_class_corpus(
            class_name="HAM",
            label_id=0,
            target_count=args.target_ham,
            topics_data=ham_cfg["topics_data"],
            lengths_config=ham_cfg["lengths"],
            length_weights=ham_cfg["length_weights"],
            styles_list=ham_cfg["styles"],
            prompt_builder_fn=build_ham_prompt,
            output_path=os.path.join(args.out_dir, "synthetic_ham.csv"),
            progress_path=os.path.join(args.out_dir, "synthetic_ham_progress.csv"),
        )

    if args.mode in ("all", "spam"):
        anti_phish_rule = spam_cfg.get(
            "anti_phishing_rule",
            DEFAULT_SCENARIOS["spam"]["anti_phishing_rule"],
        )
        generate_class_corpus(
            class_name="SPAM",
            label_id=1,
            target_count=args.target_spam,
            topics_data=spam_cfg["topics_data"],
            lengths_config=spam_cfg["lengths"],
            length_weights=spam_cfg["length_weights"],
            styles_list=spam_cfg["styles"],
            prompt_builder_fn=lambda bt, em, ld, lt, cnt, ldesc, il, st: build_spam_prompt(
                bt, em, ld, lt, cnt, ldesc, il, st, anti_phish_rule
            ),
            output_path=os.path.join(args.out_dir, "synthetic_spam.csv"),
            progress_path=os.path.join(args.out_dir, "synthetic_spam_progress.csv"),
        )


if __name__ == "__main__":
    main()
