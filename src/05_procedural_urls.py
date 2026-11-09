#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Шаг 05: контролируемая процедурная инжекция реалистичных и тайпсквоттинговых URL
во все три синтетических класса (Phishing, Spam, Ham) через DynamicURLGenerator
для предотвращения Feature Leakage.
"""

import os
import random
import string

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
GLOBAL_SEED = 42

# (входной_файл, выходной_файл, label_val, url_inject_rate)
URL_INJECTION_CONFIGS = [
    ("merged_synthetic_phishing.csv", "final_synthetic_phishing.csv", 2, 0.65),
    ("dedup_synthetic_spam.csv", "final_synthetic_spam.csv", 1, 0.35),
    ("dedup_synthetic_ham.csv", "final_synthetic_ham.csv", 0, 0.15),
]


class DynamicURLGenerator:
    """Генератор синтетических URL с поддержкой тайпсквоттинга для предотвращения Feature Leakage."""

    def __init__(self, seed=GLOBAL_SEED):
        self.rng = random.Random(seed)
        self.tlds = ["ru", "com", "net", "org", "su", "рф", "io", "me", "cc", "online", "store", "tech", "pro"]
        self.subdomains = ["", "www.", "m.", "shop.", "lk.", "my.", "news.", "forum."]
        self.words = [
            "yandex", "google", "vk", "mail", "ozon", "wildberries", "avito", "drom", "auto",
            "habr", "vc", "pikabu", "lenta", "ria", "rbc", "kinopoisk", "ivi", "okko",
            "sber", "tbank", "vtb", "alfabank", "gosuslugi", "nalog", "pfrf", "mos",
            "sport", "news", "shop", "market", "kino", "music", "play", "game", "dev",
        ]
        self.shorteners = ["t.me", "vk.cc", "clck.ru", "bit.ly", "goo.gl", "youtu.be"]

    def _generate_path(self, is_spam_phish=False):
        if self.rng.random() < 0.4:
            chars = string.ascii_letters + string.digits
            return "/" + "".join(self.rng.choices(chars, k=self.rng.randint(5, 12)))

        path_words = ["article", "post", "item", "product", "catalog", "user", "profile", "video", "news"]
        if is_spam_phish:
            path_words.extend(["promo", "sale", "bonus", "auth", "confirm", "gift", "pay", "login"])

        path = "/" + self.rng.choice(path_words)
        if self.rng.random() < 0.6:
            path += "/" + str(self.rng.randint(1000, 9999999))
        if self.rng.random() < 0.3:
            path += self.rng.choice([".html", ".php", ".aspx"])
        return path

    def generate(self, label_val):
        """label_val: 0 = Ham, 1 = Spam, 2 = Phishing"""
        if self.rng.random() < 0.25:
            base = self.rng.choice(self.shorteners)
            if base == "t.me":
                if label_val in (1, 2) and self.rng.random() < 0.6:
                    bot_prefix = self.rng.choice(["promo", "secure", "auth", "gift", "vip", "support", "bot"])
                    return f"t.me/{bot_prefix}_{self.rng.randint(1, 99)}"
                chars = string.ascii_letters + string.digits
                return "t.me/" + "".join(self.rng.choices(chars, k=self.rng.randint(5, 10)))
            chars = string.ascii_letters + string.digits
            return base + "/" + "".join(self.rng.choices(chars, k=self.rng.randint(4, 7)))

        subdomain = self.rng.choice(self.subdomains)
        domain = self.rng.choice(self.words)

        if label_val == 2 and self.rng.random() < 0.7:
            if self.rng.random() < 0.5:
                domain = domain.replace("o", "0").replace("l", "1").replace("i", "1").replace("s", "5")
            else:
                domain = domain + "-" + self.rng.choice(["auth", "support", "secure", "login", "verify"])
        elif self.rng.random() < 0.3:
            domain = domain + "-" + self.rng.choice(self.words)

        tld = self.rng.choice(self.tlds)
        path = self._generate_path(is_spam_phish=(label_val in (1, 2)))
        protocol = "https://" if self.rng.random() < 0.8 else "http://"
        return f"{protocol}{subdomain}{domain}.{tld}{path}"


def inject_urls_into_pool(input_path: str, output_path: str, label_val: int, url_inject_rate: float):
    if not os.path.exists(input_path):
        print(f"[WARN] Файл {input_path} не найден, пропуск.")
        return

    df = pd.read_csv(input_path)
    col = "message" if "message" in df.columns else "text"
    final_msgs = df[col].dropna().astype(str).tolist()

    rng = np.random.default_rng(GLOBAL_SEED + label_val)
    url_gen = DynamicURLGenerator(seed=GLOBAL_SEED + label_val)

    injected_count = 0
    for i in range(len(final_msgs)):
        if rng.random() < url_inject_rate:
            chosen_url = url_gen.generate(label_val)
            final_msgs[i] = f"{final_msgs[i].rstrip('.!')} {chosen_url}"
            injected_count += 1

    rng.shuffle(final_msgs)
    out_df = pd.DataFrame({
        "message": final_msgs,
        "label": label_val,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    pct = (injected_count / len(final_msgs) * 100.0) if final_msgs else 0.0
    print(
        f"[INFO] [Класс {label_val}] {input_path} -> {output_path} | "
        f"Всего строк: {len(final_msgs)} | URL инжектировано: {injected_count} ({pct:.1f}%)"
    )


def main():
    for in_name, out_name, label_val, rate in URL_INJECTION_CONFIGS:
        in_path = os.path.join(DATA_DIR, in_name)
        out_path = os.path.join(DATA_DIR, out_name)
        inject_urls_into_pool(in_path, out_path, label_val, rate)


if __name__ == "__main__":
    main()
