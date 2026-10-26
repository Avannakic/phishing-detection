import pandas as pd
import re
import random
import string
import os

DATA_DIR = os.path.join("data", "interim")

input_file = os.path.join(DATA_DIR, "merged_synthetic_phishing.csv")
output_file = os.path.join(DATA_DIR, "final_synthetic_phishing.csv")

random.seed(42)
df = pd.read_csv(input_file)
print(f"[INFO] Загружено строк для генерации URL: {len(df)}")


def generate_full_url():
    name_len = random.randint(6, 12)
    domain = "".join(random.choices(string.ascii_lowercase, k=name_len))
    tld = random.choices([".ru", ".com", ".su", ".рф"], weights=[0.70, 0.20, 0.05, 0.05])[0]
    protocol = random.choices(["https://", "http://", ""], weights=[0.8, 0.1, 0.1])[0]
    paths = ["", "/", "/login", "/auth", "/verify", "/secure", "/pay", "/profile", "/update"]
    path = random.choice(paths)

    if random.random() < 0.3:
        token = "".join(random.choices(string.ascii_letters + string.digits, k=16))
        path += f"?token={token}"

    return f"{protocol}{domain}{tld}{path}"


def replace_example_links(text):
    if not isinstance(text, str):
        return text

    def bare_domain_replacer(match):
        name_len = random.randint(6, 12)
        random_name = "".join(random.choices(string.ascii_lowercase, k=name_len))
        tld = random.choices([".ru", ".com"], weights=[0.75, 0.25])[0]
        return f"{random_name}{tld}"

    text = re.sub(r"example\.(com|ru|org|net)", bare_domain_replacer, text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bexample\b",
        lambda m: "".join(random.choices(string.ascii_lowercase, k=random.randint(6, 12))),
        text,
        flags=re.IGNORECASE,
    )

    placeholder_pattern = r"\[\s*[^\]]*(?:link|url|ссылка|сайт)[^\]]*\s*\]"
    text = re.sub(placeholder_pattern, lambda m: generate_full_url(), text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![/:.-])\blink\b(?![/:.-])", lambda m: generate_full_url(), text, flags=re.IGNORECASE)
    return text


df["message"] = df["message"].apply(replace_example_links)
df = df.drop_duplicates(subset=["message"]).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(output_file, index=False, encoding="utf-8")
print(f"[INFO] Финальный пул синтетического фишинга с URL сохранён в: {output_file} ({len(df)} строк)")