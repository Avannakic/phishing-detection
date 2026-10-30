import argparse
import gc
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

MODELS_CONFIG = {
    "RuBERT": {
        "model_id": "DeepPavlov/rubert-base-cased",
        "type": "encoder",
        "batch_size": 64,
        "grad_accum": 1,
        "lr": 2e-5,
    },
    "RuRoBERTa-large": {
        "model_id": "ai-forever/ruRoberta-large",
        "type": "encoder",
        "batch_size": 32,
        "grad_accum": 2,
        "lr": 1e-5,
    },
    "Vikhr": {
        "model_id": "Vikhrmodels/Vikhr-Llama-3.1-8B-Instruct",
        "type": "llm_lora",
        "batch_size": 8,
        "grad_accum": 8,
        "lr": 1e-4,
    },
    "Qwen3-14B": {
        "model_id": "Qwen/Qwen3-14B",
        "type": "llm_lora",
        "batch_size": 8,
        "grad_accum": 8,
        "lr": 1e-4,
    },
}

EXPERIMENTS = [
    "1_Baseline_Imbalanced",
    "2_Baseline_ClassWeights",
    "3_Baseline_Oversampling",
    "4_Model2_PureSynthetic",
    "5_Model3_Hybrid_Full_100pct",
    "6_Ablation_Hybrid_50pct",
    "7_Ablation_Hybrid_25pct",
    "8_Ablation_Hybrid_10pct",
]

TRAIN_BASE_SIZE = 14000
VAL_BASE_SIZE = 3000
TEST_BASE_SIZE = 3000
TARGET_PHISH_COUNT = 7000
VAL_PHISH_SIZE = 40
VAL_SYNTH_PHISH_SIZE = 1000
VAL_SYNTH_PHISH_MODE = "pure_synth_only"
MAX_LENGTH = 256
NUM_EPOCHS = 3
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
    filenames = {
        "real_ham": "real_ham.csv",
        "real_spam": "real_spam.csv",
        "real_phish": "real_phishing.csv",
        "synth_ham": "synthetic_ham.csv",
        "synth_spam": "synthetic_spam.csv",
        "synth_phish": "final_synthetic_phishing.csv",
    }
    labels = {
        "real_ham": 0,
        "real_spam": 1,
        "real_phish": 2,
        "synth_ham": 0,
        "synth_spam": 1,
        "synth_phish": 2,
    }
    pools = {
        key: clean_pool(pd.read_csv(os.path.join(data_dir, filename)), labels[key])
        for key, filename in filenames.items()
    }

    required_exact = {"real_ham": 20000, "real_spam": 20000, "real_phish": 300}
    required_min = {"synth_ham": 14000, "synth_spam": 14000, "synth_phish": 10000}
    for key, expected in required_exact.items():
        if len(pools[key]) != expected:
            raise ValueError(f"{filenames[key]}: ожидалось {expected} строк, получено {len(pools[key])}")
    for key, expected in required_min.items():
        if len(pools[key]) < expected:
            raise ValueError(f"{filenames[key]}: требуется минимум {expected} строк, получено {len(pools[key])}")

    pools["synth_ham"] = pools["synth_ham"].sample(
        n=TRAIN_BASE_SIZE, random_state=42
    ).reset_index(drop=True)
    pools["synth_spam"] = pools["synth_spam"].sample(
        n=TRAIN_BASE_SIZE, random_state=42
    ).reset_index(drop=True)
    return pools


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


def build_train_dataframe(
    exp_name,
    fold_real_phish_train,
    seed,
    real_ham_train,
    real_spam_train,
    synth_ham,
    synth_spam,
    synth_phish_train,
):
    n_needed_full = TARGET_PHISH_COUNT - len(fold_real_phish_train)
    if exp_name in ["1_Baseline_Imbalanced", "2_Baseline_ClassWeights"]:
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train], ignore_index=True)
    elif exp_name == "3_Baseline_Oversampling":
        oversampled_phish = fold_real_phish_train.sample(
            n=TARGET_PHISH_COUNT, replace=True, random_state=seed
        )
        df = pd.concat([real_ham_train, real_spam_train, oversampled_phish], ignore_index=True)
    elif exp_name == "4_Model2_PureSynthetic":
        s_phish = synth_phish_train.sample(
            n=min(TARGET_PHISH_COUNT, len(synth_phish_train)), random_state=seed
        )
        df = pd.concat([synth_ham, synth_spam, s_phish], ignore_index=True)
    elif exp_name in [
        "5_Model3_Hybrid_Full_100pct",
        "6_Ablation_Hybrid_50pct",
        "7_Ablation_Hybrid_25pct",
        "8_Ablation_Hybrid_10pct",
    ]:
        fractions = {
            "5_Model3_Hybrid_Full_100pct": 1.0,
            "6_Ablation_Hybrid_50pct": 0.50,
            "7_Ablation_Hybrid_25pct": 0.25,
            "8_Ablation_Hybrid_10pct": 0.10,
        }
        n_synth = int(fractions[exp_name] * n_needed_full)
        s_phish = synth_phish_train.sample(
            n=min(n_synth, len(synth_phish_train)), random_state=seed
        )
        df = pd.concat(
            [real_ham_train, real_spam_train, fold_real_phish_train, s_phish],
            ignore_index=True,
        )
    else:
        raise ValueError(f"Неизвестное имя эксперимента: {exp_name}")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_val_dataframe(
    exp_name,
    real_ham_val,
    real_spam_val,
    real_phish_val,
    synth_phish_val,
    mode,
):
    parts = [real_ham_val, real_spam_val]
    if exp_name == "4_Model2_PureSynthetic":
        parts.append(synth_phish_val)
    else:
        parts.append(real_phish_val)
        if mode == "always_include_synth" and exp_name in [
            "5_Model3_Hybrid_Full_100pct",
            "6_Ablation_Hybrid_50pct",
            "7_Ablation_Hybrid_25pct",
            "8_Ablation_Hybrid_10pct",
        ]:
            parts.append(synth_phish_val)
    return pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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


def compute_metrics(pred):
    return calculate_metrics(pred.label_ids, pred.predictions.argmax(-1))


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"У {model_id} отсутствуют pad_token и eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_config, pad_token_id):
    model_id = model_config["model_id"]
    if model_config["type"] == "encoder":
        return AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = pad_token_id
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def make_training_arguments(model_key, model_config, exp_name, fold, seed, output_dir):
    is_llm = model_config["type"] == "llm_lora"
    use_bf16 = is_llm and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints", model_key, f"{exp_name}_fold_{fold}"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"] * 2,
        gradient_accumulation_steps=model_config["grad_accum"],
        learning_rate=model_config["lr"],
        warmup_steps=150,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        seed=seed,
        fp16=not is_llm and torch.cuda.is_available(),
        bf16=use_bf16,
        gradient_checkpointing=is_llm,
        gradient_checkpointing_kwargs={"use_reentrant": False} if is_llm else None,
        dataloader_num_workers=2,
        report_to="none",
    )


def update_summary(summary_path, new_row):
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if "Model" not in summary.columns:
            summary = pd.DataFrame()
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
    parser = argparse.ArgumentParser(description="5-fold CV: энкодеры и 4-bit QLoRA классификаторы")
    parser.add_argument("--models", nargs="+", choices=list(MODELS_CONFIG), default=list(MODELS_CONFIG))
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument("--data-dir", default=PROCESSED_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--val-synth-phish-mode",
        choices=["pure_synth_only", "always_include_synth"],
        default=VAL_SYNTH_PHISH_MODE,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pools = load_pools(args.data_dir)
    print(
        "[INFO] Пулы: "
        + " | ".join(f"{key}={len(value)}" for key, value in pools.items())
    )

    for model_key in args.models:
        model_config = MODELS_CONFIG[model_key]
        tokenizer = load_tokenizer(model_config["model_id"])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        for exp_name in args.experiments:
            print(f"\n{'=' * 70}\n{model_key} | {exp_name}\n{'=' * 70}")
            fold_metrics = []
            total_cm = np.zeros((3, 3), dtype=int)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (dev_idx, test_idx) in enumerate(
                skf.split(pools["real_phish"], pools["real_phish"]["label"]), 1
            ):
                seed = 42 + fold
                set_seed(seed)
                real_phish_test = pools["real_phish"].iloc[test_idx].reset_index(drop=True)
                real_phish_dev = pools["real_phish"].iloc[dev_idx].reset_index(drop=True)
                real_phish_train, real_phish_val = train_test_split(
                    real_phish_dev, test_size=VAL_PHISH_SIZE, random_state=seed, shuffle=True
                )
                real_ham_train, real_ham_val, real_ham_test = split_base_class(
                    pools["real_ham"], seed
                )
                real_spam_train, real_spam_val, real_spam_test = split_base_class(
                    pools["real_spam"], seed
                )
                synth_phish_train, synth_phish_val = split_synthetic_phishing(
                    pools["synth_phish"], seed
                )

                train_df = build_train_dataframe(
                    exp_name,
                    real_phish_train,
                    seed,
                    real_ham_train,
                    real_spam_train,
                    pools["synth_ham"],
                    pools["synth_spam"],
                    synth_phish_train,
                )
                val_df = build_val_dataframe(
                    exp_name,
                    real_ham_val,
                    real_spam_val,
                    real_phish_val,
                    synth_phish_val,
                    args.val_synth_phish_mode,
                )
                test_df = pd.concat(
                    [real_ham_test, real_spam_test, real_phish_test], ignore_index=True
                ).sample(frac=1, random_state=seed).reset_index(drop=True)
                print(
                    f"Фолд {fold}: Train={len(train_df)} | Val={len(val_df)} | "
                    f"Test={len(test_df)} | Synth Val reserve={len(synth_phish_val)}"
                )

                encoded = {}
                datasets = {}
                for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                    encoded[split_name] = tokenizer(
                        split_df["message"].tolist(), truncation=True, padding=False, max_length=MAX_LENGTH
                    )
                    datasets[split_name] = TextDataset(
                        encoded[split_name], split_df["label"].tolist()
                    )

                model = load_model(model_config, tokenizer.pad_token_id)
                training_args = make_training_arguments(
                    model_key, model_config, exp_name, fold, seed, args.output_dir
                )
                trainer_class = WeightedTrainer if exp_name == "2_Baseline_ClassWeights" else Trainer
                trainer_kwargs = {}
                if exp_name == "2_Baseline_ClassWeights":
                    w_phish = float(len(real_ham_train)) / len(real_phish_train)
                    trainer_kwargs["class_weights"] = torch.tensor(
                        [1.0, 1.0, w_phish], dtype=torch.float
                    )
                trainer = trainer_class(
                    model=model,
                    args=training_args,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets["val"],
                    compute_metrics=compute_metrics,
                    data_collator=data_collator,
                    **trainer_kwargs,
                )
                trainer.train()
                preds_output = trainer.predict(datasets["test"])
                predictions = preds_output.predictions.argmax(-1)
                labels = test_df["label"].tolist()
                metrics = calculate_metrics(labels, predictions)
                fold_metrics.append({"Fold": fold, **metrics})
                total_cm += confusion_matrix(labels, predictions, labels=[0, 1, 2])

                del model, trainer, datasets, encoded, preds_output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            df_folds = pd.DataFrame(fold_metrics)
            df_folds.insert(0, "Experiment", exp_name)
            df_folds.insert(0, "Model", model_key)
            folds_path = os.path.join(args.output_dir, f"folds_{model_key}_{exp_name}.csv")
            df_folds.to_csv(folds_path, index=False, encoding="utf-8")

            row = {"Model": model_key, "Experiment": exp_name, "CM_Sum_All_Folds": str(total_cm.tolist())}
            for metric_name in METRIC_NAMES:
                values = df_folds[metric_name].to_numpy()
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                ci95 = stats.t.ppf(0.975, df=len(values) - 1) * (std / np.sqrt(len(values)))
                row[f"{metric_name}_formatted"] = f"{mean:.4f} ± {std:.4f} (CI: ±{ci95:.4f})"
            update_summary(os.path.join(args.output_dir, "SUMMARY.csv"), row)

        del tokenizer, data_collator
        gc.collect()

    print(f"\n[SUCCESS] Результаты сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
