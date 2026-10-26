import os
import gc
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Определяем папку скрипта для корректной работы путей при запуске откуда угодно
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# СПИСОК ВСЕХ ЭКСПЕРИМЕНТОВ (КРОСС-ВАЛИДАЦИЯ 5 ФОЛДОВ)
# ==========================================
EXPERIMENTS = [
    "1_Baseline_Imbalanced",
    "2_Baseline_ClassWeights",
    "3_Baseline_Oversampling",
    "4_Model2_PureSynthetic",
    "5_Model3_Hybrid_Full_100pct",
    "6_Ablation_Hybrid_50pct",
    "7_Ablation_Hybrid_25pct",
    "8_Ablation_Hybrid_10pct"
]

# Целевые размеры выборок в каждом фолде
TRAIN_BASE_SIZE = 14000
VAL_BASE_SIZE = 3000
TEST_BASE_SIZE = 3000

TARGET_PHISH_COUNT = 7000
VAL_PHISH_SIZE = 40  # Из 240 не-тестовых примеров: 40 идут в Val, 200 остаются в Train


def resolve_file(*candidates: str) -> str:
    """
    Находит файл в стандартных подпапках data/ (приоритет у data/processed/ после шага 06,
    затем data/real/ и data/interim/).
    """
    search_dirs = [
        os.path.join(BASE_DIR, "data", "processed"),
        os.path.join(BASE_DIR, "data", "real"),
        os.path.join(BASE_DIR, "data", "interim"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "processed")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "real")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "interim")),
        BASE_DIR,
    ]
    for name in candidates:
        if not name:
            continue
        for d in search_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"Не найден ни один из файлов {candidates} в {BASE_DIR}")


def clean_pool(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Гарантирует отсутствие пустых строк и дубликатов, устанавливает метку класса."""
    df = df.dropna(subset=["message"]).copy()
    df["norm_text"] = df["message"].astype(str).str.lower().str.strip()
    df = df[df["norm_text"] != ""].drop_duplicates(subset=["norm_text"]).drop(columns=["norm_text"])
    df["label"] = label
    return df.reset_index(drop=True)


def split_base_class(df: pd.DataFrame, train_size: int, val_size: int, test_size: int, seed: int):
    """
    Разделяет единый пул базового класса (real_ham или real_spam) в каждом фолде
    на строго непересекающиеся выборки Train (14 000), Val (3 000) и Test (3 000).
    Поскольку seed фиксирован для каждого фолда, все модели внутри одного фолда
    обучаются, валидируются и тестируются на абсолютно идентичных данных.
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total_needed = train_size + val_size + test_size

    if len(df_shuffled) < total_needed:
        ratio = len(df_shuffled) / total_needed
        test_size = int(test_size * ratio)
        val_size = int(val_size * ratio)
        train_size = len(df_shuffled) - test_size - val_size

    df_test = df_shuffled.iloc[:test_size].reset_index(drop=True)
    df_val = df_shuffled.iloc[test_size : test_size + val_size].reset_index(drop=True)
    df_train = df_shuffled.iloc[test_size + val_size : test_size + val_size + train_size].reset_index(drop=True)
    return df_train, df_val, df_test


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
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels_ids = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_ids, preds, average="macro", zero_division=0
    )
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        labels_ids, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    acc = accuracy_score(labels_ids, preds)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "phish_precision": p_class[2],
        "phish_recall": r_class[2],
        "phish_f1": f1_class[2]
    }


def build_train_dataframe(
    exp_name, fold_real_phish_train, seed,
    real_ham_train, real_spam_train, synth_ham, synth_spam, synth_phish
):
    n_needed_full = TARGET_PHISH_COUNT - len(fold_real_phish_train)  # 7000 - 200 = 6800

    if exp_name in ["1_Baseline_Imbalanced", "2_Baseline_ClassWeights"]:
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train], ignore_index=True)

    elif exp_name == "3_Baseline_Oversampling":
        # Дублируем 200 реальных обучающих примеров фишинга до 7000
        oversampled_phish = fold_real_phish_train.sample(
            n=TARGET_PHISH_COUNT, replace=True, random_state=seed
        )
        df = pd.concat([real_ham_train, real_spam_train, oversampled_phish], ignore_index=True)

    elif exp_name == "4_Model2_PureSynthetic":
        # Чисто синтетическая модель (14k Synth Ham + 14k Synth Spam + 7k Synth Phish)
        s_phish = synth_phish.sample(n=min(TARGET_PHISH_COUNT, len(synth_phish)), random_state=seed)
        df = pd.concat([synth_ham, synth_spam, s_phish], ignore_index=True)

    elif exp_name in ["5_Model3_Hybrid_Full_100pct", "6_Model3B_Hybrid_Full_100pct"]:
        # 100% целевого объема синтетики (6800 примеров / 34x от 200 реальных)
        s_phish = synth_phish.sample(n=min(n_needed_full, len(synth_phish)), random_state=seed)
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train, s_phish], ignore_index=True)

    elif exp_name == "6_Ablation_Hybrid_50pct":
        # 50% целевого объема синтетики (3400 примеров / 17x от 200 реальных)
        n_synth_50pct = int(0.50 * n_needed_full)
        s_phish = synth_phish.sample(n=min(n_synth_50pct, len(synth_phish)), random_state=seed)
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train, s_phish], ignore_index=True)

    elif exp_name == "7_Ablation_Hybrid_25pct":
        # 25% целевого объема синтетики (1700 примеров / 8.5x от 200 реальных)
        n_synth_25pct = int(0.25 * n_needed_full)
        s_phish = synth_phish.sample(n=min(n_synth_25pct, len(synth_phish)), random_state=seed)
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train, s_phish], ignore_index=True)

    elif exp_name == "8_Ablation_Hybrid_10pct":
        # 10% целевого объема синтетики (680 примеров / 3.4x от 200 реальных)
        n_synth_10pct = int(0.10 * n_needed_full)
        s_phish = synth_phish.sample(n=min(n_synth_10pct, len(synth_phish)), random_state=seed)
        df = pd.concat([real_ham_train, real_spam_train, fold_real_phish_train, s_phish], ignore_index=True)

    else:
        raise ValueError(f"Неизвестное имя эксперимента: {exp_name}")

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. Загрузка единых пулов РЕАЛЬНЫХ данных (real_ham.csv, real_spam.csv, real_phishing.csv)
    real_ham_pool = clean_pool(pd.read_csv(resolve_file("real_ham.csv")), label=0)
    real_spam_pool = clean_pool(pd.read_csv(resolve_file("real_spam.csv")), label=1)
    real_phish_all = clean_pool(pd.read_csv(resolve_file("real_phishing.csv")), label=2)

    print(
        f"[INFO] Загружены пулы реальных данных -> "
        f"Ham: {len(real_ham_pool)} | Spam: {len(real_spam_pool)} | Phishing: {len(real_phish_all)}"
    )

    # 2. Загрузка СИНТЕТИЧЕСКИХ данных из предыдущих шагов пайплайна
    synth_ham = clean_pool(pd.read_csv(resolve_file("synthetic_ham.csv")), label=0)
    synth_ham = synth_ham.sample(n=min(TRAIN_BASE_SIZE, len(synth_ham)), random_state=42)

    synth_spam = clean_pool(pd.read_csv(resolve_file("synthetic_spam.csv")), label=1)
    synth_spam = synth_spam.sample(n=min(TRAIN_BASE_SIZE, len(synth_spam)), random_state=42)

    synth_phish = clean_pool(pd.read_csv(resolve_file("final_synthetic_phishing.csv")), label=2)

    print(
        f"[INFO] Загружены пулы синтетических данных -> "
        f"Synth Ham: {len(synth_ham)} | Synth Spam: {len(synth_spam)} | Synth Phishing: {len(synth_phish)}"
    )

    # 3. Разбиение 300 реальных примеров фишинга на 5 фолдов (по 60 тестовых примеров в каждом фолде)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dummy_y = np.zeros(len(real_phish_all))
    summary_rows = []

    use_fp16 = torch.cuda.is_available()

    for exp_name in EXPERIMENTS:
        print(f"\n{'='*65}\nСТАРТ ЭКСПЕРИМЕНТА: {exp_name}\n{'='*65}")
        fold_metrics = []
        total_cm = np.zeros((3, 3), dtype=int)

        for fold, (dev_idx, test_idx) in enumerate(skf.split(real_phish_all, dummy_y), 1):
            print(f"\n--- {exp_name} | ФОЛД {fold} / 5 ---")
            seed = 42 + fold

            # А) Разделение реального фишинга в фолде: 60 в слепой Test, 200 в Train, 40 в Val
            fold_real_phish_test = real_phish_all.iloc[test_idx].reset_index(drop=True)
            fold_real_phish_dev = real_phish_all.iloc[dev_idx].reset_index(drop=True)
            fold_real_phish_train, fold_real_phish_val = train_test_split(
                fold_real_phish_dev, test_size=VAL_PHISH_SIZE, random_state=seed, shuffle=True
            )

            # Б) Разделение единых пулов real_ham и real_spam в фолде (14k Train / 3k Val / 3k Test)
            real_ham_train, real_ham_val, real_ham_test = split_base_class(
                real_ham_pool, TRAIN_BASE_SIZE, VAL_BASE_SIZE, TEST_BASE_SIZE, seed=seed
            )
            real_spam_train, real_spam_val, real_spam_test = split_base_class(
                real_spam_pool, TRAIN_BASE_SIZE, VAL_BASE_SIZE, TEST_BASE_SIZE, seed=seed
            )

            # В) Собираем 3 изолированные выборки: Train, Val (для эпох) и Test (для слепого экзамена)
            train_df = build_train_dataframe(
                exp_name, fold_real_phish_train, seed,
                real_ham_train, real_spam_train, synth_ham, synth_spam, synth_phish
            )
            val_df = pd.concat([real_ham_val, real_spam_val, fold_real_phish_val], ignore_index=True)
            test_df = pd.concat([real_ham_test, real_spam_test, fold_real_phish_test], ignore_index=True)

            print(f"Размеры выборок -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

            # Токенизация
            train_enc = tokenizer(
                train_df["message"].astype(str).tolist(), truncation=True, padding=True, max_length=256
            )
            val_enc = tokenizer(
                val_df["message"].astype(str).tolist(), truncation=True, padding=True, max_length=256
            )
            test_enc = tokenizer(
                test_df["message"].astype(str).tolist(), truncation=True, padding=True, max_length=256
            )

            train_dataset = TextDataset(train_enc, train_df["label"].tolist())
            val_dataset = TextDataset(val_enc, val_df["label"].tolist())
            test_dataset = TextDataset(test_enc, test_df["label"].tolist())

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

            training_args = TrainingArguments(
                output_dir=os.path.join(BASE_DIR, "results", f"{exp_name}_fold_{fold}"),
                num_train_epochs=3,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=128,
                warmup_steps=150,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                save_total_limit=1,
                seed=seed,
                fp16=use_fp16,
                dataloader_num_workers=2,
                report_to="none"
            )

            if exp_name == "2_Baseline_ClassWeights":
                w_phish = float(len(real_ham_train)) / len(fold_real_phish_train)
                weights = torch.tensor([1.0, 1.0, w_phish], dtype=torch.float)
                trainer = WeightedTrainer(
                    class_weights=weights,
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics
                )

            trainer.train()

            # Финальное слепое тестирование лучшего чекпоинта на изолированном test_dataset
            preds_output = trainer.predict(test_dataset)
            eval_res = preds_output.metrics
            fold_metrics.append(eval_res)

            preds = preds_output.predictions.argmax(-1)
            total_cm += confusion_matrix(test_df["label"].tolist(), preds, labels=[0, 1, 2])

            del model, trainer, train_dataset, val_dataset, test_dataset, train_enc, val_enc, test_enc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df_folds = pd.DataFrame(fold_metrics)
        df_folds.to_csv(os.path.join(BASE_DIR, f"folds_{exp_name}.csv"), index=False)

        row = {"Experiment": exp_name, "CM_Sum_All_Folds": str(total_cm.tolist())}
        for col in ["test_accuracy", "test_f1_macro", "test_phish_precision", "test_phish_recall", "test_phish_f1"]:
            vals = df_folds[col].values
            mean, std = np.mean(vals), np.std(vals, ddof=1)
            ci95 = stats.t.ppf(0.975, df=len(vals) - 1) * (std / np.sqrt(len(vals)))
            clean_name = col.replace("test_", "")
            row[f"{clean_name}_formatted"] = f"{mean:.4f} ± {std:.4f} (CI: ±{ci95:.4f})"

        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(BASE_DIR, "SUMMARY.csv"), index=False)

    print("\n[ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ] Итоговый файл сохранен: SUMMARY.csv")


if __name__ == "__main__":
    main()
