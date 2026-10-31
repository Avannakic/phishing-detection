import os
import random
import re
from collections import deque

import ollama
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "vikhr-llama3.1-8b"
MIN_RESPONSE_LENGTH = 25
REFUSAL_MARKERS = (
    "я не могу", "извините", "как ии", "не могу помочь",
    "мошеннич", "незаконн", "as an ai", "i cannot",
)

random.seed(42)
target_count = 14000
out_dir = os.path.join(PROJECT_ROOT, "data", "interim")
os.makedirs(out_dir, exist_ok=True)
progress_file = os.path.join(out_dir, "synthetic_phishing_v1_progress.csv")

SCENARIOS = {
    "портала Госуслуг": [
        "необходимость срочно подтвердить данные",
        "угроза блокировки учетной записи",
        "уведомление о новом штрафе или задолженности"
    ],
    "Центрального Банка (ЦБ РФ) или ФСБ": [
        "страх уголовной ответственности за перевод средств",
        "давление авторитетом и требование спасти деньги на безопасном счете"
    ],
    "ректора или деканата университета": [
        "давление авторитетом и угроза отчисления",
        "срочный запрос на участие в обязательном опросе по ссылке"
    ],
    "крупного банка (Сбербанк, Т-Банк, ВТБ)": [
        "угроза потери денег (подозрительный перевод)",
        "жадность (одобрен кредит на выгодных условиях)",
        "жадность (начисление бонусных рублей или кешбэка)"
    ],
    "родственника или знакомого": [
        "просьба срочно дать денег в долг на карту",
        "сочувствие (просьба проголосовать за ребенка в конкурсе)",
        "кликбейт (шокирующее фото, 'смотри кто на фото')"
    ],
    "городской поликлиники или ЕМИАС": [
        "необходимость срочно прикрепиться к поликлинике",
        "угроза аннулирования медицинского полиса"
    ],
    "официальной поддержки Telegram": [
        "угроза удаления аккаунта из-за подозрительной активности",
        "срочное подтверждение номера телефона"
    ],
    "бота с бесплатным Telegram Premium": [
        "жадность и халява (бесплатная подписка на год)",
        "срочность (акция заканчивается через час)"
    ],
    "бота для знакомств или 18+ контента": [
        "любопытство (кто-то посмотрел твою анкету)",
        "переход по ссылке для просмотра скрытых фото"
    ],
    "магазина ключей для игр (Steam) или нейросетей": [
        "жадность (бесплатные ключи или токены для ChatGPT)",
        "распродажа 90% по секретной ссылке"
    ]
}

EMOJI_RULES = {
    "портала Госуслуг": (0.02, "Пиши СТРОГО БЕЗ эмодзи, сухой канцелярский язык."),
    "Центрального Банка (ЦБ РФ) или ФСБ": (0.01, "СТРОГО БЕЗ эмодзи. Официальный тон силовых структур."),
    "ректора или деканата университета": (0.05, "Минимум эмодзи, официальный тон."),
    "крупного банка (Сбербанк, Т-Банк, ВТБ)": (0.15, "Используй 1-2 сдержанных эмодзи, характерных для рассылок банков."),
    "родственника или знакомого": (0.30, "Можешь использовать обычные бытовые эмодзи."),
    "городской поликлиники или ЕМИАС": (0.01, "Пиши СТРОГО БЕЗ эмодзи, как системное уведомление."),
    "официальной поддержки Telegram": (0.10, "Используй эмодзи-предупреждения (например, ⚠️, 🔴)."),
    "бота с бесплатным Telegram Premium": (0.80, "Обильно используй яркие, кликбейтные эмодзи (🔥, 🎁, 🎉, ⚡️)."),
    "бота для знакомств или 18+ контента": (0.90, "Используй много пошлых, игривых и привлекающих внимание эмодзи (🍑, 💦, 🔞, ❤️‍🔥)."),
    "магазина ключей для игр (Steam) или нейросетей": (0.70, "Используй геймерские и кликбейтные эмодзи (🎮, 🚀, 😱).")
}

len_short = ("короткие", 10, "1-2 предложения (до 150 символов)")
len_med = ("средние", 4, "3-5 предложений (до 300 символов)")
len_long = ("длинные", 2, "развернутый текст (около 600-700 символов)")

ALLOWED_LENGTHS = {
    "портала Госуслуг": [len_med, len_med, len_med, len_med, len_long],
    "Центрального Банка (ЦБ РФ) или ФСБ": [len_med, len_med, len_med, len_med, len_long],
    "ректора или деканата университета": [len_med, len_med, len_med, len_med, len_long],
    "крупного банка (Сбербанк, Т-Банк, ВТБ)": [len_med, len_med, len_med, len_med, len_long],
    "городской поликлиники или ЕМИАС": [len_med],
    "родственника или знакомого": [len_short, len_short, len_short, len_med],
    "официальной поддержки Telegram": [len_short, len_short, len_short, len_med],
    "бота с бесплатным Telegram Premium": [len_short, len_short, len_short, len_short, len_med],
    "бота для знакомств или 18+ контента": [len_short],
    "магазина ключей для игр (Steam) или нейросетей": [len_short, len_short, len_short, len_med]
}

valid_combinations = []
for sender, tactics in SCENARIOS.items():
    for tactic in tactics:
        for length in ALLOWED_LENGTHS[sender]:
            valid_combinations.append((sender, tactic, length))

random.shuffle(valid_combinations)

def is_refusal(text):
    clean_text = text.strip().lower()
    return len(clean_text) < MIN_RESPONSE_LENGTH or any(
        marker in clean_text for marker in REFUSAL_MARKERS
    )


class OllamaGenerator:
    def __init__(self):
        self.current_model = MODEL_NAME
        # Fallback-логика и счетчики отказов полностью удалены

    def generate(self, prompt):
        try:
            response = ollama.generate(
                model=self.current_model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.75,      # Достаточно для креативности, но без бреда
                    "top_p": 0.90,            
                    "top_k": 40,              
                    "repeat_penalty": 1.05,   # Минимальный штраф, чтобы не ломать русскую грамматику
                    "num_predict": 200        
                },
            )
            text = response["response"]
        except Exception as exc:
            print(f"[Ошибка Ollama, {self.current_model}]: {exc}")
            text = ""

        refused = is_refusal(text)
        
        # Если Qwen отказался, просто возвращаем пустую строку. 
        # Основной цикл while сам проигнорирует её и попробует сгенерировать заново.
        if refused:
            print("[INFO] Qwen выдал отказ (сработала цензура). Повторяем попытку...")
            
        return "" if refused else text

if os.path.exists(progress_file):
    dataset = set(pd.read_csv(progress_file)['message'].tolist())
    print(f"Загружено {len(dataset)} строк из {progress_file}.")
else:
    dataset = set()

combo_idx = 0
requests_made = 0
generator = OllamaGenerator()

print(f"Всего доступно уникальных комбинаций промпта: {len(valid_combinations)}")
print(f"Начинаем генерацию {target_count} сообщений с модели {generator.current_model}...")

bad_starts = ("я не могу", "i cannot", "к сожалению", "однако", "извините", "простите", "as an ai", "я обязан", "сообщение")
bad_words = ["example", "test", "sample", "insert", "вставь", "[ссылка"]

try:
    while len(dataset) < target_count:
        sender, tactic, (length_type, count, length_desc) = valid_combinations[combo_idx]

        emoji_chance, emoji_prompt = EMOJI_RULES[sender]
        if random.random() < emoji_chance:
            emoji_instruction = emoji_prompt
        else:
            emoji_instruction = "Пиши СТРОГО БЕЗ эмодзи."

        # Шанс 50% на ссылку
        if random.random() < 0.5:
            link_instruction = "ОБЯЗАТЕЛЬНО органично вставь в текст правдоподобную ссылку (например t.me/secure_bot, vk.cc/hG8d, gosuslugi-auth.ru). КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать слова example, test, sample или плейсхолдеры."
        else:
            link_instruction = "Не вставляй никаких ссылок, просто призывай к действию или пугай."

        prompt = (
            f"Ты — AI-ассистент исследователя в области информационной безопасности. "
            f"Для обучения защитной антиспам-системы (NLP классификатора) мне нужны "
            f"СИНТЕТИЧЕСКИЕ ПРИМЕРЫ текстов, используемых злоумышленниками. "
            f"Сгенерируй {count} {length_type} учебных примеров социальной инженерии "
            f"для мессенджеров ({length_desc}) от лица: {sender}. Тематика атаки: {tactic}. \n"
            f"ПРАВИЛА ГЕНЕРАЦИИ ДАТАСЕТА:\n"
            f"1. Пиши сразу примеры текстов, без вводных слов. Не пиши 'Конечно' или 'Я понимаю'. Встраивай контекст отправителя естественно!\n"
            f"2. {emoji_instruction}\n"
            f"3. {link_instruction}\n"
            f"4. Тексты должны выглядеть максимально правдоподобно для РФ, чтобы антиспам-модель научилась их ловить.\n"
            f"ВЫВЕДИ ТОЛЬКО СООБЩЕНИЯ. Каждое сообщение с новой строки. Без нумерации, без кавычек."
        )

        text = generator.generate(prompt)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for line in lines:
            if "Here" in line or "Вот" in line or "Сгенерирован" in line:
                continue
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if clean_line.startswith('"') and clean_line.endswith('"'):
                clean_line = clean_line[1:-1]

            clean_lower = clean_line.lower()
            if len(clean_line) < 25:
                continue
            if clean_lower.startswith(bad_starts):
                continue
            if any(bw in clean_lower for bw in bad_words):
                continue

            dataset.add(clean_line)

        requests_made += 1
        combo_idx = (combo_idx + 1) % len(valid_combinations)

        if requests_made % 10 == 0:
            print(
                f"Запросов: {requests_made}. Собрано: {len(dataset)}/{target_count}. "
                f"Модель: {generator.current_model}"
            )

        if requests_made % 50 == 0:
            pd.DataFrame(sorted(dataset), columns=['message']).to_csv(
                progress_file, index=False, encoding="utf-8"
            )
            print("Промежуточное сохранение выполнено.")

except KeyboardInterrupt:
    print(f"\n\n[!] Генерация прервана пользователем. Сохраняем собранные {len(dataset)} строк...")

if len(dataset) > 0:
    final_filename = os.path.join(out_dir, "raw_synthetic_base.csv")
    final_list = sorted(dataset)[:target_count]
    pd.DataFrame(final_list, columns=['message']).to_csv(final_filename, index=False, encoding="utf-8")
    pd.DataFrame(final_list, columns=['message']).to_csv(progress_file, index=False, encoding="utf-8")

    if len(dataset) >= target_count and os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"\nГотово! Сгенерировано: {len(final_list)} уникальных сообщений. Файл сохранен как {final_filename}")
else:
    print("\nДанных не собрано, сохранять нечего.")