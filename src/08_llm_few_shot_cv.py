import argparse
import gc
import os
import re

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_few_shot")

MODELS_CONFIG = {
    "Vikhr": "Vikhrmodels/Vikhr-Llama-3.1-8B-Instruct",
    "Qwen3-14B": "Qwen/Qwen3-14B",
}
PROMPT_MODES = ["Zero-Shot", "Few-Shot_Real", "Few-Shot_Synth"]
CLASS_NAMES = {
    0: "легитимное сообщение (Ham)",
    1: "коммерческий спам (Spam)",
    2: "фишинг или социальная инженерия (Phishing)",
}

TRAIN_BASE_SIZE = 14000
VAL_BASE_SIZE = 3000
TEST_BASE_SIZE = 3000
VAL_PHISH_SIZE = 40
VAL_SYNTH_PHISH_SIZE = 1000
SHOTS_PER_CLASS = 3
MAX_PROMPT_LENGTH = 4096
MAX_NEW_TOKENS = 8
METRIC_NAMES = ["accuracy", "f1_macro", "phish_precision", "phish_recall", "phish_f1"]


def clean_pool(df: pd.DataFrame, label: int) -> pd.DataFrame:
    df = df.dropna(subset=["message"]).copy()
    df["message"] = df["message"].astype(str).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    df["norm_text"] = df["message"].str.lower()
    df = (
        df[df["norm_text"] != ""]
        .drop_duplicates(subset=["norm_text"])
        .drop(columns=["norm_text"])
    )
    df["label"] = label
    return df[["message", "label"]].reset_index(drop=True)


def load_pools(data_dir: str):
    real_ham = clean_pool(pd.read_csv(os.path.join(data_dir, "real_ham.csv")), 0)
    real_spam = clean_pool(pd.read_csv(os.path.join(data_dir, "real_spam.csv")), 1)
    real_phish = clean_pool(pd.read_csv(os.path.join(data_dir, "real_phishing.csv")), 2)
    synth_phish = clean_pool(
        pd.read_csv(os.path.join(data_dir, "final_synthetic_phishing.csv")), 2
    )
    for name, df, expected in [
        ("real_ham.csv", real_ham, 20000),
        ("real_spam.csv", real_spam, 20000),
        ("real_phishing.csv", real_phish, 300),
    ]:
        if len(df) != expected:
            raise ValueError(f"{name}: ожидалось {expected} строк, получено {len(df)}")
    if len(synth_phish) < 10000:
        raise ValueError(
            f"final_synthetic_phishing.csv: требуется минимум 10000 строк, получено {len(synth_phish)}"
        )
    return real_ham, real_spam, real_phish, synth_phish


def split_base_class(df: pd.DataFrame, seed: int):
    total_needed = TRAIN_BASE_SIZE + VAL_BASE_SIZE + TEST_BASE_SIZE
    if len(df) < total_needed:
        raise ValueError(f"Для разбиения нужно {total_needed} строк, доступно {len(df)}")
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_end = TEST_BASE_SIZE
    val_end = test_end + VAL_BASE_SIZE
    train_end = val_end + TRAIN_BASE_SIZE
    return (
        shuffled.iloc[val_end:train_end].reset_index(drop=True),
        shuffled.iloc[test_end:val_end].reset_index(drop=True),
        shuffled.iloc[:test_end].reset_index(drop=True),
    )


def split_synthetic_phishing(df: pd.DataFrame, seed: int):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = min(VAL_SYNTH_PHISH_SIZE, len(shuffled) - 1)
    if val_size < 0:
        raise ValueError("Синтетический фишинг-пул пуст")
    val_reserve = shuffled.iloc[:val_size].reset_index(drop=True)
    train_pool = shuffled.iloc[val_size:].reset_index(drop=True)
    return train_pool, val_reserve


def sample_shot_examples(real_ham_train, real_spam_train, real_phish_train, synth_phish_train, mode, seed):
    if mode == "Zero-Shot":
        return []
    phish_pool = real_phish_train if mode == "Few-Shot_Real" else synth_phish_train
    examples = []
    for label, pool in [(0, real_ham_train), (1, real_spam_train), (2, phish_pool)]:
        if len(pool) < SHOTS_PER_CLASS:
            raise ValueError(f"Недостаточно примеров класса {label} для {mode}")
        sampled = pool.sample(n=SHOTS_PER_CLASS, random_state=seed + label)
        examples.extend((label, message) for message in sampled["message"].tolist())
    return examples


def build_prompt(message, examples):
    lines = [
        "Классифицируй сообщение из мессенджера в один из классов:",
        "0 — легитимное сообщение (Ham);",
        "1 — коммерческий спам (Spam);",
        "2 — фишинг или социальная инженерия (Phishing).",
        "Ответь только одной цифрой: 0, 1 или 2. Не добавляй пояснений.",
    ]
    if examples:
        lines.append("\nПримеры:")
        for label, example in examples:
            lines.append(f"Сообщение: {example}\nКласс: {label}")
    lines.append(f"\nСообщение для классификации: {message}\nКласс:")
    return "\n".join(lines)


def format_prompt(tokenizer, prompt, model_key):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return prompt


def parse_prediction(response):
    text = response.strip()
    if text in {"0", "1", "2"}:
        return int(text)
    labelled = re.findall(r"(?:класс|ответ|label|answer)\s*[:=]?\s*([012])\b", text, re.IGNORECASE)
    if labelled:
        return int(labelled[-1])
    digits = re.findall(r"(?<!\d)([012])(?!\d)", text)
    return int(digits[-1]) if digits else -1


def calculate_metrics(labels, predictions):
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2], zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_macro,
        "phish_precision": p_class[2],
        "phish_recall": r_class[2],
        "phish_f1": f1_class[2],
    }


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"У {model_id} отсутствуют pad_token и eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, max_new_tokens):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def update_summary(summary_path, new_row):
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()
    if not summary.empty:
        summary = summary[
            ~(
                (summary["Model"] == new_row["Model"])
                & (summary["Experiment"] == new_row["Experiment"])
            )
        ]
    summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
    summary = summary.sort_values(["Model", "Experiment"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Zero/Few-Shot оценка LLM на фолдах шага 07")
    parser.add_argument("--models", nargs="+", choices=list(MODELS_CONFIG), default=list(MODELS_CONFIG))
    parser.add_argument("--modes", nargs="+", choices=PROMPT_MODES, default=PROMPT_MODES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--data-dir", default=PROCESSED_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size и max-new-tokens должны быть положительными")

    os.makedirs(args.output_dir, exist_ok=True)
    real_ham, real_spam, real_phish, synth_phish = load_pools(args.data_dir)

    for model_key in args.models:
        model, tokenizer = load_model_and_tokenizer(MODELS_CONFIG[model_key])
        for mode in args.modes:
            print(f"\n{'=' * 70}\n{model_key} | {mode}\n{'=' * 70}")
            fold_metrics = []
            total_cm = np.zeros((3, 3), dtype=int)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (dev_idx, test_idx) in enumerate(
                skf.split(real_phish, real_phish["label"]), 1
            ):
                seed = 42 + fold
                set_seed(seed)
                real_phish_test = real_phish.iloc[test_idx].reset_index(drop=True)
                real_phish_dev = real_phish.iloc[dev_idx].reset_index(drop=True)
                real_phish_train, _ = train_test_split(
                    real_phish_dev, test_size=VAL_PHISH_SIZE, random_state=seed, shuffle=True
                )
                real_ham_train, _, real_ham_test = split_base_class(real_ham, seed)
                real_spam_train, _, real_spam_test = split_base_class(real_spam, seed)
                synth_phish_train, _ = split_synthetic_phishing(synth_phish, seed)
                test_df = pd.concat(
                    [real_ham_test, real_spam_test, real_phish_test], ignore_index=True
                ).sample(frac=1, random_state=seed).reset_index(drop=True)
                examples = sample_shot_examples(
                    real_ham_train,
                    real_spam_train,
                    real_phish_train,
                    synth_phish_train,
                    mode,
                    seed,
                )

                prompts = [
                    format_prompt(tokenizer, build_prompt(message, examples), model_key)
                    for message in test_df["message"].tolist()
                ]
                predictions = []
                raw_responses = []
                for start in range(0, len(prompts), args.batch_size):
                    responses = predict_batch(
                        model,
                        tokenizer,
                        prompts[start:start + args.batch_size],
                        args.max_new_tokens,
                    )
                    raw_responses.extend(responses)
                    predictions.extend(parse_prediction(response) for response in responses)
                    print(
                        f"\rФолд {fold}: {min(start + args.batch_size, len(prompts))}/{len(prompts)}",
                        end="",
                        flush=True,
                    )
                print()

                labels = test_df["label"].tolist()
                metrics = calculate_metrics(labels, predictions)
                unparsed = sum(prediction == -1 for prediction in predictions)
                fold_metrics.append({"Fold": fold, **metrics, "unparsed": unparsed})
                total_cm += confusion_matrix(labels, predictions, labels=[0, 1, 2])
                pd.DataFrame(
                    {
                        "message": test_df["message"],
                        "label": labels,
                        "prediction": predictions,
                        "raw_response": raw_responses,
                    }
                ).to_csv(
                    os.path.join(args.output_dir, f"predictions_{model_key}_{mode}_fold_{fold}.csv"),
                    index=False,
                    encoding="utf-8",
                )
                print(f"Фолд {fold}: {metrics}; не распарсено ответов: {unparsed}")

            df_folds = pd.DataFrame(fold_metrics)
            df_folds.insert(0, "Experiment", mode)
            df_folds.insert(0, "Model", model_key)
            df_folds.to_csv(
                os.path.join(args.output_dir, f"folds_{model_key}_{mode}.csv"),
                index=False,
                encoding="utf-8",
            )
            row = {"Model": model_key, "Experiment": mode, "CM_Sum_All_Folds": str(total_cm.tolist())}
            for metric_name in METRIC_NAMES:
                values = df_folds[metric_name].to_numpy()
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                ci95 = stats.t.ppf(0.975, df=len(values) - 1) * (std / np.sqrt(len(values)))
                row[f"{metric_name}_formatted"] = f"{mean:.4f} ± {std:.4f} (CI: ±{ci95:.4f})"
            update_summary(os.path.join(args.output_dir, "SUMMARY_FEW_SHOT.csv"), row)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Few-Shot результаты сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
