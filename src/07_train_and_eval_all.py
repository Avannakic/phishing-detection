#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 07: Unified 5-fold cross-validation pipeline for model fine-tuning and Few-Shot evaluation.
Combines classifier training (encoder family: RuBERT, RuRoBERTa; llm_lora family: Vikhr-8B, Qwen3-14B)
and generative Zero/Few-Shot evaluation.

Model definitions and hyperparameters are fully decoupled into configs/models_config.json
with automatic architecture classification (encoder vs llm_lora).
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Hardware TF32 acceleration on NVIDIA Tensor Cores
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_LENGTH = 128
TRAIN_BASE_SIZE = 14000
VAL_BASE_SIZE = 3000
TEST_BASE_SIZE = 3000
VAL_PHISH_SIZE = 40
TARGET_PHISH_COUNT = 7000
SHOTS_PER_CLASS = 3
MAX_PROMPT_LENGTH = 2048

EXIT_CONTINUE = 98
EXIT_OOM_RETRY = 99
EXIT_USER_ABORT = 130

# Standalone fallback model configuration template for offline operation and reproducible execution
DEFAULT_MODELS_CONFIG = {
    "defaults": {
        "encoder": {
            "type": "encoder",
            "epochs": 3,
            "learning_rate": 2e-5,
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "grad_accum": 1,
            "max_length": 128,
            "optim": "adamw_torch_fused",
            "grad_ckpt": False,
            "supports_train_eval": True,
            "supports_few_shot": False,
            "needs_vram_worker_reset": False,
        },
        "llm_lora": {
            "type": "llm_lora",
            "epochs": 2,
            "learning_rate": 1e-4,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "grad_accum": 2,
            "max_length": 128,
            "optim": "paged_adamw_8bit",
            "grad_ckpt": False,
            "supports_train_eval": True,
            "supports_few_shot": True,
            "needs_vram_worker_reset": True,
            "few_shot_batch_size": 32,
            "few_shot_max_new_tokens": 8,
            "max_prompt_length": 2048,
            "enable_thinking": False,
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "modules_to_save": ["score"],
            },
        },
    },
    "models": {
        "Vikhr-8B": {
            "enabled": True,
            "model_id": "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
            "type": "llm_lora",
        },
        "RuBERT": {
            "enabled": True,
            "model_id": "DeepPavlov/rubert-base-cased",
            "type": "encoder",
        },
        "RuRoBERTa-large": {
            "enabled": True,
            "model_id": "ai-forever/ruRoberta-large",
            "type": "encoder",
            "learning_rate": 1.5e-5,
        },
        "Qwen3-14B": {
            "enabled": False,
            "model_id": "Qwen/Qwen3-14B",
            "type": "llm_lora",
            "train_batch_size": 16,
            "eval_batch_size": 32,
            "grad_accum": 4,
            "enable_thinking": False,
        },
    },
}

EXPERIMENTS = [
    "1_Baseline_Imbalanced",
    "2_Baseline_ClassWeights",
    "3_Baseline_Oversampling",
    "4_Pure_Synthetic",
    "5_Hybrid_3.4x_10pct",
    "6_Hybrid_8.5x_25pct",
    "7_Hybrid_17x_50pct",
    "8_Hybrid_34x_100pct",
]

HYBRID_SYNTH_COUNTS = {
    "5_Hybrid_3.4x_10pct": 680,
    "6_Hybrid_8.5x_25pct": 1700,
    "7_Hybrid_17x_50pct": 3400,
    "8_Hybrid_34x_100pct": 6800,
}

PROMPT_MODES = ["Zero-Shot", "Few-Shot_Real", "Few-Shot_Synth"]

METRIC_NAMES = [
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "phish_precision",
    "phish_recall",
    "phish_f1",
    "spam_precision",
    "spam_recall",
    "spam_f1",
    "ham_precision",
    "ham_recall",
    "ham_f1",
]

RAW_FIELDNAMES = [
    "RunType",
    "Model",
    "Config",
    "Fold",
    "Train_Size",
    "Val_Size",
    "Test_Size",
    *METRIC_NAMES,
    "CM",
    "unparsed",
]

_LAST_SIGINT_TIME = 0.0


class SkipConfiguration(BaseException):
    """Internal exception for single Ctrl+C (skip current experimental configuration)."""


class StopAllExecution(BaseException):
    """Internal exception for double Ctrl+C (abort complete pipeline execution)."""


def handle_sigint(signum, frame):
    global _LAST_SIGINT_TIME
    now = time.monotonic()
    if now - _LAST_SIGINT_TIME < 2.0:
        raise StopAllExecution
    _LAST_SIGINT_TIME = now
    raise SkipConfiguration


def is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cudnn_status_alloc_failed" in msg
        or "cublas_status_alloc_failed" in msg
    )


def resolve_model_family(model_id: str, declared_type: str = "auto") -> str:
    """
    Classify model architecture family: 'encoder' (BERT-like) vs 'llm_lora' (decoder LLMs).
    If declared_type is explicitly 'encoder' or 'llm_lora', returns it directly.
    If declared_type is 'auto', inspects configuration via AutoConfig.from_pretrained.
    """
    if declared_type in ("encoder", "llm_lora"):
        return declared_type

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load AutoConfig for '{model_id}' during auto-detection: {exc}. "
            f"Please specify 'type': 'encoder' or 'type': 'llm_lora' explicitly in the config."
        )

    model_type = getattr(config, "model_type", "").lower()
    archs = [a.lower() for a in getattr(config, "architectures", [])]

    encoder_types = {
        "bert",
        "roberta",
        "xlm-roberta",
        "deberta",
        "deberta-v2",
        "electra",
        "distilbert",
        "albert",
        "camembert",
    }
    llm_types = {
        "llama",
        "qwen2",
        "qwen3",
        "mistral",
        "gemma",
        "gemma2",
        "phi",
        "phi3",
        "falcon",
        "gpt2",
        "gpt_neox",
    }

    if model_type in encoder_types or any(
        "formaskedlm" in a or ("forsequenceclassification" in a and "causal" not in a)
        for a in archs
    ):
        return "encoder"
    if model_type in llm_types or any("forcausallm" in a for a in archs):
        return "llm_lora"

    raise ValueError(
        f"Unable to unambiguously classify architecture for '{model_id}' "
        f"(model_type='{model_type}', architectures={archs}). "
        f"Please explicitly specify 'type': 'encoder' or 'type': 'llm_lora' in configs/models_config.json."
    )


def load_models_config(config_path=None):
    """
    Load model configuration from external JSON file with fallback to defaults.
    If file is absent, uses standalone embedded DEFAULT_MODELS_CONFIG.
    """
    if config_path is not None:
        target_path = Path(config_path)
    else:
        target_path = PROJECT_ROOT / "configs" / "models_config.json"

    if target_path.is_file() and target_path.stat().st_size > 0:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        print(f"[CONFIG] Loaded external model configuration: {target_path}")
    else:
        raw_config = copy.deepcopy(DEFAULT_MODELS_CONFIG)
        print(f"[CONFIG] External config not found ({target_path}); using standalone DEFAULT_MODELS_CONFIG.")

    defaults = raw_config.get("defaults", {})
    models_raw = raw_config.get("models", {})

    resolved_models = {}
    for model_name, model_info in models_raw.items():
        declared_type = model_info.get("type", "auto")
        model_id = model_info.get("model_id", "")
        family = resolve_model_family(model_id, declared_type)

        template = copy.deepcopy(defaults.get(family, {}))

        # Deep merge quantization and lora dictionaries
        for k, v in model_info.items():
            if isinstance(v, dict) and k in template and isinstance(template[k], dict):
                template[k] = {**template[k], **v}
            else:
                template[k] = v

        template["type"] = family
        resolved_models[model_name] = template

    return resolved_models


def load_verified_pools(data_dir=None):
    """
    Load 6 verified data pools produced by Step 06 (06_finalise_datasets.py).
    By default reads from PROJECT_ROOT / 'data' / 'processed', falling back to 'data'.
    Verifies existence and target quotas without redundant cleaning.
    """
    search_dirs = []
    if data_dir is not None:
        search_dirs.append(Path(data_dir))
    search_dirs.append(PROJECT_ROOT / "data" / "processed")
    search_dirs.append(PROJECT_ROOT / "data")

    file_specs = [
        ("real_ham", ["real_ham.csv"], 0, 20000, "exact"),
        ("real_spam", ["real_spam.csv"], 1, 20000, "exact"),
        ("real_phish", ["real_phishing.csv"], 2, 300, "exact"),
        ("synth_ham", ["synthetic_ham.csv"], 0, 14000, "min"),
        ("synth_spam", ["synthetic_spam.csv"], 1, 14000, "min"),
        (
            "synth_phish",
            ["synthetic_phishing.csv", "final_synthetic_phishing.csv"],
            2,
            7000,
            "min_sample",
        ),
    ]

    pools = {}
    for key, candidate_names, label, target_size, check_type in file_specs:
        found_path = None
        for d in search_dirs:
            for name in candidate_names:
                candidate_path = d / name
                if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                    found_path = candidate_path
                    break
            if found_path is not None:
                break

        if found_path is None:
            names_str = " or ".join(candidate_names)
            searched_str = ", ".join(str(d) for d in search_dirs)
            raise FileNotFoundError(
                f"Data file {names_str} not found in search paths: {searched_str}"
            )

        df = pd.read_csv(found_path)
        if "message" not in df.columns:
            if "text" in df.columns:
                df = df.rename(columns={"text": "message"})
            else:
                raise ValueError(f"{found_path}: missing 'message' column")

        df = df.dropna(subset=["message"]).copy()
        df["message"] = df["message"].astype(str)
        df["label"] = int(label)

        if check_type == "exact":
            if len(df) < target_size:
                raise ValueError(
                    f"{found_path.name}: expected exactly {target_size} rows, found {len(df)}"
                )
            if len(df) > target_size:
                df = df.sample(n=target_size, random_state=42).reset_index(drop=True)
        elif check_type in ("min", "min_sample"):
            if len(df) < target_size:
                raise ValueError(
                    f"{found_path.name}: requires at least {target_size} rows, found {len(df)}"
                )
            if len(df) > target_size:
                df = df.sample(n=target_size, random_state=42).reset_index(drop=True)

        pools[key] = df[["message", "label"]].reset_index(drop=True)

    print("[DATA] Verified data pools loaded:")
    for key, df in pools.items():
        print(f"  - {key}: {len(df)} rows (label={df['label'].iloc[0]})")
    return pools


def split_base_class(df, seed):
    """Split base classes (20k rows) into Test (3k), Val (3k), and Train (14k)."""
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:TEST_BASE_SIZE].reset_index(drop=True)
    val_df = shuffled.iloc[TEST_BASE_SIZE : TEST_BASE_SIZE + VAL_BASE_SIZE].reset_index(drop=True)
    train_df = shuffled.iloc[TEST_BASE_SIZE + VAL_BASE_SIZE : TEST_BASE_SIZE + VAL_BASE_SIZE + TRAIN_BASE_SIZE].reset_index(drop=True)
    return train_df, val_df, test_df


def build_folds(pools):
    """
    Construct 5 stratified folds:
    - StratifiedKFold on 300 rows of real_phish (60 test, 40 val, 200 train);
    - real_ham and real_spam (20k each) partitioned into 3000 test, 3000 val, 14000 train;
    - val_df (6,040 rows) and test_df (6,060 rows) are ALWAYS strictly real.
    """
    folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    real_phish = pools["real_phish"]

    for fold, (dev_idx, test_idx) in enumerate(skf.split(real_phish, real_phish["label"]), 1):
        seed = 42 + fold
        real_phish_test = real_phish.iloc[test_idx].reset_index(drop=True)
        real_phish_dev = real_phish.iloc[dev_idx].reset_index(drop=True)
        real_phish_train, real_phish_val = train_test_split(
            real_phish_dev,
            test_size=VAL_PHISH_SIZE,
            random_state=seed,
            shuffle=True,
        )

        real_ham_train, real_ham_val, real_ham_test = split_base_class(pools["real_ham"], seed)
        real_spam_train, real_spam_val, real_spam_test = split_base_class(pools["real_spam"], seed)

        val_df = pd.concat(
            [real_ham_val, real_spam_val, real_phish_val], ignore_index=True
        ).sample(frac=1, random_state=seed).reset_index(drop=True)

        test_df = pd.concat(
            [real_ham_test, real_spam_test, real_phish_test], ignore_index=True
        ).sample(frac=1, random_state=seed).reset_index(drop=True)

        assert len(real_phish_train) == 200, f"Fold {fold}: real_phish_train != 200"
        assert len(real_phish_val) == 40, f"Fold {fold}: real_phish_val != 40"
        assert len(real_phish_test) == 60, f"Fold {fold}: real_phish_test != 60"
        assert len(val_df) == 6040, f"Fold {fold}: val_df != 6040 ({len(val_df)})"
        assert len(test_df) == 6060, f"Fold {fold}: test_df != 6060 ({len(test_df)})"

        folds.append(
            {
                "fold": fold,
                "seed": seed,
                "real_ham_train": real_ham_train,
                "real_spam_train": real_spam_train,
                "real_phish_train": real_phish_train.reset_index(drop=True),
                "val": val_df,
                "test": test_df,
            }
        )
    return folds


def build_train_dataframe(config_name, fold_data, pools):
    """
    Construct training set for specified experimental configuration (1..8).
    """
    seed = fold_data["seed"]
    real_parts = [
        fold_data["real_ham_train"],
        fold_data["real_spam_train"],
        fold_data["real_phish_train"],
    ]

    if config_name in ("1_Baseline_Imbalanced", "2_Baseline_ClassWeights"):
        parts = real_parts
    elif config_name == "3_Baseline_Oversampling":
        oversampled_phish = fold_data["real_phish_train"].sample(
            n=TARGET_PHISH_COUNT, replace=True, random_state=seed
        ).reset_index(drop=True)
        parts = real_parts[:2] + [oversampled_phish]
    elif config_name == "4_Pure_Synthetic":
        parts = [
            pools["synth_ham"].sample(n=TRAIN_BASE_SIZE, random_state=seed).reset_index(drop=True),
            pools["synth_spam"].sample(n=TRAIN_BASE_SIZE, random_state=seed).reset_index(drop=True),
            pools["synth_phish"].sample(n=TARGET_PHISH_COUNT, random_state=seed).reset_index(drop=True),
        ]
    elif config_name in HYBRID_SYNTH_COUNTS:
        synth_phish = pools["synth_phish"].sample(
            n=HYBRID_SYNTH_COUNTS[config_name], random_state=seed
        ).reset_index(drop=True)
        parts = real_parts + [synth_phish]
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

    return pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weights)(
            logits.view(-1, logits.shape[-1]), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def calculate_metrics(labels, predictions):
    """Compute 13 classification performance metrics (Macro, Ham, Spam, Phishing)."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2], zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "phish_precision": float(precision[2]),
        "phish_recall": float(recall[2]),
        "phish_f1": float(f1[2]),
        "spam_precision": float(precision[1]),
        "spam_recall": float(recall[1]),
        "spam_f1": float(f1[1]),
        "ham_precision": float(precision[0]),
        "ham_recall": float(recall[0]),
        "ham_f1": float(f1[0]),
    }


def compute_metrics(pred):
    predictions = pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return calculate_metrics(pred.label_ids, predictions.argmax(-1))


def load_training_tokenizer(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
    if model_config["type"] == "llm_lora":
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_config['model_id']} has no eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    elif tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_config['model_id']} has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_model(model_config, pad_token_id, oom_safe=False):
    model_id = model_config["model_id"]
    model_type = model_config["type"]

    if model_type == "encoder":
        return AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3,
            torch_dtype=torch.bfloat16,
        )

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    q_cfg = model_config.get("quantization", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = pad_token_id
    use_gc = True if oom_safe else model_config.get("grad_ckpt", False)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)

    lora_dict = model_config.get("lora", {})
    lora_config = LoraConfig(
        r=lora_dict.get("r", 16),
        lora_alpha=lora_dict.get("lora_alpha", 32),
        lora_dropout=lora_dict.get("lora_dropout", 0.05),
        bias=lora_dict.get("bias", "none"),
        task_type=TaskType.SEQ_CLS,
        target_modules=lora_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        modules_to_save=lora_dict.get("modules_to_save", ["score"]),
    )
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = pad_token_id
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def make_training_arguments(model_config, seed, fold_output_dir, oom_safe=False):
    use_gc = True if oom_safe else model_config.get("grad_ckpt", False)
    train_bs = model_config["train_batch_size"]
    grad_accum = model_config["grad_accum"]
    eval_bs = model_config["eval_batch_size"]

    if oom_safe and model_config["type"] == "llm_lora" and train_bs >= 32:
        train_bs = train_bs // 2
        grad_accum = grad_accum * 2
        eval_bs = max(32, eval_bs // 2)
        print(
            f"[OOM-SAFE] Memory-safe profile activated: train_batch_size={train_bs}, "
            f"grad_accum={grad_accum} (effective_batch={train_bs * grad_accum}), grad_ckpt={use_gc}"
        )

    use_tf32 = bool(
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and torch.cuda.get_device_capability()[0] >= 8
    )

    args = TrainingArguments(
        output_dir=str(fold_output_dir),
        num_train_epochs=model_config["epochs"],
        learning_rate=model_config["learning_rate"],
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        eval_accumulation_steps=16,
        optim=model_config["optim"],
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        bf16=True,
        tf32=use_tf32,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        train_sampling_strategy="random",
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    args.group_by_length = False
    return args


def encode_dataframe(tokenizer, df, max_length=MAX_LENGTH):
    encodings = tokenizer(
        df["message"].tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    return TextDataset(encodings, df["label"].tolist())


def direct_test_inference(model, tokenizer, test_df, batch_size, max_length=MAX_LENGTH):
    """
    Direct sequential batched inference.
    Guarantees strict 1:1 message, label, and prediction ordering independent of Trainer.predict.
    """
    model.eval()
    device = next(model.parameters()).device
    messages = test_df["message"].tolist()
    labels = test_df["label"].to_numpy(dtype=int)
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(messages), batch_size):
            batch_messages = messages[start : start + batch_size]
            batch = tokenizer(
                batch_messages,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            batch_predictions = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(batch_predictions.tolist())

            del batch, outputs, logits, batch_predictions
            if start == 0 or (start // batch_size) % 10 == 0:
                print(
                    f"\r[TEST INFERENCE] {min(start + batch_size, len(messages))}/{len(messages)}",
                    end="",
                    flush=True,
                )

    print(f"\r[TEST INFERENCE] {len(messages)}/{len(messages)}", flush=True)
    predictions = np.asarray(predictions, dtype=int)
    if len(predictions) != len(labels):
        raise RuntimeError(
            f"Prediction count ({len(predictions)}) does not match label count ({len(labels)})"
        )

    metrics = calculate_metrics(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return metrics, cm, predictions


def save_predictions(path, test_df, predictions):
    """Atomically persist record-level predictions for test verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(
        {
            "message": test_df["message"].tolist(),
            "label": test_df["label"].to_numpy(dtype=int),
            "prediction": np.asarray(predictions, dtype=int),
        }
    ).to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, path)


def read_raw_results(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    required = {"RunType", "Model", "Config", "Fold", "CM", *METRIC_NAMES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File {path} missing required columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def completed_keys(raw_rows):
    return {
        (str(row["RunType"]), str(row["Model"]), str(row["Config"]), int(row["Fold"]))
        for row in raw_rows
    }


def append_raw_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        output.flush()
        os.fsync(output.fileno())


def update_summary(raw_rows, summary_path):
    """Update aggregate summary table (mean, std, 95% CI, summed confusion matrices)."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_columns = ["RunType", "Model", "Config", "N_Folds", "CM_Sum_All_Folds"]
    for metric_name in METRIC_NAMES:
        summary_columns.extend(
            [f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_ci95"]
        )
    if not raw_rows:
        pd.DataFrame(columns=summary_columns).to_csv(summary_path, index=False)
        return

    df = pd.DataFrame(raw_rows)
    rows = []
    for keys, group in df.groupby(["RunType", "Model", "Config"], sort=True):
        matrices = [np.asarray(json.loads(value), dtype=int) for value in group["CM"]]
        row = {
            "RunType": keys[0],
            "Model": keys[1],
            "Config": keys[2],
            "N_Folds": int(len(group)),
            "CM_Sum_All_Folds": json.dumps(
                np.sum(matrices, axis=0).tolist(), ensure_ascii=False
            ),
        }
        for metric_name in METRIC_NAMES:
            values = pd.to_numeric(group[metric_name]).to_numpy(dtype=float)
            mean = float(np.mean(values))
            if len(values) > 1:
                std = float(np.std(values, ddof=1))
                ci95 = float(
                    stats.t.ppf(0.975, df=len(values) - 1) * std / np.sqrt(len(values))
                )
            else:
                std = 0.0
                ci95 = 0.0
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std
            row[f"{metric_name}_ci95"] = ci95
        rows.append(row)
    summary = pd.DataFrame(rows, columns=summary_columns)
    temp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, summary_path)


def is_better_fold(candidate, incumbent):
    """
    Model checkpoint selection policy:
    - Priority given to folds with phish_precision >= 0.50;
    - Maximizes phish_recall among eligible folds;
    - Otherwise falls back to maximizing phish_f1.
    """
    if incumbent is None:
        return True
    candidate_eligible = float(candidate["phish_precision"]) >= 0.50
    incumbent_eligible = float(incumbent["phish_precision"]) >= 0.50
    if candidate_eligible != incumbent_eligible:
        return candidate_eligible
    if candidate_eligible:
        candidate_recall = float(candidate["phish_recall"])
        incumbent_recall = float(incumbent["phish_recall"])
        if not math.isclose(candidate_recall, incumbent_recall, abs_tol=1e-12):
            return candidate_recall > incumbent_recall
    return float(candidate["phish_f1"]) > float(incumbent["phish_f1"])


def best_history_row(raw_rows, model_key, config_name):
    best = None
    for row in raw_rows:
        if (
            str(row["RunType"]) == "TrainEval"
            and str(row["Model"]) == model_key
            and str(row["Config"]) == config_name
            and is_better_fold(row, best)
        ):
            best = row
    return best


def save_best_model(model, tokenizer, row, best_models_dir, model_config):
    """Save best-performing fold checkpoint with metadata for universal re-validation."""
    best_dir = Path(best_models_dir) / str(row["Model"]) / str(row["Config"])
    temp_dir = best_dir.with_name(best_dir.name + ".tmp")
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer.save_pretrained(temp_dir)
    metadata = {key: row[key] for key in ("RunType", "Model", "Config", "Fold")}
    metadata.update({name: float(row[name]) for name in METRIC_NAMES})
    metadata["model_id"] = model_config["model_id"]
    metadata["model_type"] = model_config["type"]
    metadata["max_length"] = model_config.get("max_length", MAX_LENGTH)
    metadata["quantization"] = model_config.get("quantization", None)

    with (temp_dir / "best_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    shutil.rmtree(best_dir, ignore_errors=True)
    os.replace(temp_dir, best_dir)
    print(f"[BEST] Saved best fold {row['Fold']} checkpoint: {best_dir}")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def run_training(args, pools, folds, raw_rows, raw_path, summary_path):
    """Execute supervised classifier fine-tuning across configured experimental matrices."""
    completed = completed_keys(raw_rows)
    folds_done_in_worker = 0
    predictions_dir = Path(args.output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_config = args.models_config_dict

    for model_key in args.models:
        model_config = models_config[model_key]
        if not model_config.get("supports_train_eval", True):
            print(f"[SKIP] Model {model_key} marked with supports_train_eval=False; skipping training.")
            continue

        pending_for_model = any(
            ("TrainEval", model_key, cfg, fd["fold"]) not in completed
            for cfg in args.configs
            for fd in folds
        )
        if not pending_for_model:
            print(f"[RESUME] Model {model_key} already fully trained; skipping.")
            continue

        tokenizer = load_training_tokenizer(model_config)
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        )
        model_max_length = model_config.get("max_length", MAX_LENGTH)

        for config_name in args.configs:
            print(f"\n{'=' * 80}\n{model_key} | {config_name}\n{'=' * 80}")
            try:
                for fold_data in folds:
                    fold = fold_data["fold"]
                    key = ("TrainEval", model_key, config_name, fold)
                    if key in completed:
                        print(f"[RESUME] Skipping completed fold: {key}")
                        continue

                    seed = fold_data["seed"]
                    set_seed(seed)
                    train_df = build_train_dataframe(config_name, fold_data, pools)
                    val_df = fold_data["val"]
                    test_df = fold_data["test"]
                    fold_output_dir = (
                        Path(args.output_dir)
                        / "tmp_checkpoints"
                        / model_key
                        / config_name
                        / f"fold_{fold}"
                    )
                    shutil.rmtree(fold_output_dir, ignore_errors=True)
                    model = None
                    trainer = None
                    train_dataset = None
                    val_dataset = None

                    try:
                        print(
                            f"[FOLD {fold}] Train={len(train_df)} | Val={len(val_df)} | "
                            f"Test={len(test_df)}"
                        )
                        train_dataset = encode_dataframe(tokenizer, train_df, max_length=model_max_length)
                        val_dataset = encode_dataframe(tokenizer, val_df, max_length=model_max_length)

                        model = load_training_model(
                            model_config, tokenizer.pad_token_id, oom_safe=args.oom_safe
                        )
                        training_args = make_training_arguments(
                            model_config, seed, fold_output_dir, oom_safe=args.oom_safe
                        )

                        trainer_kwargs = {}
                        trainer_class = Trainer
                        if config_name == "2_Baseline_ClassWeights":
                            counts = np.bincount(train_df["label"], minlength=3).astype(float)
                            if np.any(counts == 0):
                                raise ValueError("WeightedTrainer: missing class in training partition")
                            weights = counts.max() / counts
                            trainer_kwargs["class_weights"] = torch.tensor(
                                weights, dtype=torch.float32
                            )
                            trainer_class = WeightedTrainer
                            print(f"[WEIGHTS] {weights.tolist()}")

                        trainer = trainer_class(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                            **trainer_kwargs,
                        )
                        trainer.train()

                        eval_bs = (
                            max(32, model_config["eval_batch_size"] // 2)
                            if args.oom_safe
                            else model_config["eval_batch_size"]
                        )
                        metrics, cm, predictions = direct_test_inference(
                            model=model,
                            tokenizer=tokenizer,
                            test_df=test_df,
                            batch_size=eval_bs,
                            max_length=model_max_length,
                        )

                        row = {
                            "RunType": "TrainEval",
                            "Model": model_key,
                            "Config": config_name,
                            "Fold": fold,
                            "Train_Size": len(train_df),
                            "Val_Size": len(val_df),
                            "Test_Size": len(test_df),
                            **metrics,
                            "CM": json.dumps(cm.tolist()),
                            "unparsed": 0,
                        }

                        incumbent = best_history_row(raw_rows, model_key, config_name)
                        if is_better_fold(row, incumbent):
                            save_best_model(model, tokenizer, row, args.best_models_dir, model_config)

                        pred_path = (
                            predictions_dir
                            / f"{model_key}_{config_name}_fold_{fold}.csv"
                        )
                        save_predictions(pred_path, test_df, predictions)

                        append_raw_row(raw_path, row)
                        raw_rows.append(row)
                        completed.add(key)
                        update_summary(raw_rows, summary_path)
                        print(f"[TEST] {metrics}")
                        folds_done_in_worker += 1

                    except BaseException as exc:
                        if is_oom_exception(exc):
                            print(
                                f"\n[OOM-GUARD] CUDA Out of Memory on {key}! "
                                f"Cleaning temporary checkpoint and restarting worker with --oom-safe...",
                                flush=True,
                            )
                            shutil.rmtree(fold_output_dir, ignore_errors=True)
                            sys.exit(EXIT_OOM_RETRY)
                        raise
                    finally:
                        shutil.rmtree(fold_output_dir, ignore_errors=True)
                        if trainer is not None:
                            trainer.model = None
                            trainer.optimizer = None
                            trainer.lr_scheduler = None
                        del model, trainer, train_dataset, val_dataset
                        cleanup_memory()

                    if (
                        args.worker
                        and model_config.get("needs_vram_worker_reset", False)
                        and args.max_folds_per_worker > 0
                        and folds_done_in_worker >= args.max_folds_per_worker
                    ):
                        print(
                            f"[VRAM-RESET] Completed {folds_done_in_worker} folds for model with needs_vram_worker_reset=True. "
                            f"Initiating scheduled worker restart to release VRAM to 0 MiB...",
                            flush=True,
                        )
                        sys.exit(EXIT_CONTINUE)

            except SkipConfiguration:
                update_summary(raw_rows, summary_path)
                print(
                    f"\n[CTRL+C] Configuration {config_name} skipped via single Ctrl+C. "
                    "Proceeding to next configuration (press twice within 2s to abort completely).",
                    flush=True,
                )
                cleanup_memory()
                continue

        del tokenizer, data_collator
        cleanup_memory()


def sample_shot_examples(fold_data, pools, mode):
    if mode == "Zero-Shot":
        return []
    if mode == "Few-Shot_Real":
        shot_pools = {
            0: fold_data["real_ham_train"],
            1: fold_data["real_spam_train"],
            2: fold_data["real_phish_train"],
        }
    elif mode == "Few-Shot_Synth":
        shot_pools = {0: pools["synth_ham"], 1: pools["synth_spam"], 2: pools["synth_phish"]}
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    examples = []
    for label in (0, 1, 2):
        pool = shot_pools[label]
        if len(pool) < SHOTS_PER_CLASS:
            raise ValueError(f"Insufficient examples for class {label} in {mode}")
        sampled = pool.sample(
            n=SHOTS_PER_CLASS, random_state=fold_data["seed"] + label
        )
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


def format_prompt(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return prompt


def parse_prediction(response):
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text in {"0", "1", "2"}:
        return int(text)
    labelled = re.findall(
        r"(?:класс|ответ|label|answer)\s*[:=]?\s*([012])\b", text, re.IGNORECASE
    )
    if labelled:
        return int(labelled[-1])
    digits = re.findall(r"(?<!\d)([012])(?!\d)", text)
    return int(digits[-1]) if digits else -1


def load_few_shot_model(model_config):
    model_id = model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.eos_token is None:
        raise ValueError(f"{model_id} has no eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    q_cfg = model_config.get("quantization", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, max_new_tokens, max_prompt_length=MAX_PROMPT_LENGTH):
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, encoded["input_ids"].shape[1] :]
        result = tokenizer.batch_decode(generated, skip_special_tokens=True)
        del encoded, output, generated
        return result
    except BaseException as exc:
        if is_oom_exception(exc) and len(prompts) > 1:
            cleanup_memory()
            mid = len(prompts) // 2
            print(
                f"\n[FEW-SHOT OOM-SPLIT] Long context: splitting batch {len(prompts)} -> {mid} + {len(prompts) - mid}",
                flush=True,
            )
            left = predict_batch(model, tokenizer, prompts[:mid], max_new_tokens, max_prompt_length)
            right = predict_batch(model, tokenizer, prompts[mid:], max_new_tokens, max_prompt_length)
            return left + right
        raise


def save_few_shot_predictions(path, test_df, predictions, raw_responses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(
        {
            "message": test_df["message"].tolist(),
            "label": test_df["label"].tolist(),
            "prediction": predictions,
            "raw_response": raw_responses,
        }
    ).to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, path)


def run_few_shot(args, pools, folds, raw_rows, raw_path, summary_path):
    """Execute Zero-Shot, Few-Shot_Real, and Few-Shot_Synth evaluation for generative LLMs."""
    completed = completed_keys(raw_rows)
    predictions_dir = Path(args.output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_config = args.models_config_dict

    for model_key in args.few_shot_models:
        model_config = models_config[model_key]
        if not model_config.get("supports_few_shot", False):
            print(f"[SKIP] Model {model_key} marked with supports_few_shot=False; skipping Few-Shot.")
            continue

        pending_fs = any(
            ("FewShot", model_key, mode, fd["fold"]) not in completed
            for mode in args.few_shot_modes
            for fd in folds
        )
        if not pending_fs:
            print(f"[RESUME] Few-Shot for {model_key} already completed; skipping.")
            continue

        fs_batch_size = model_config.get("few_shot_batch_size", args.few_shot_batch_size)
        fs_max_tokens = model_config.get("few_shot_max_new_tokens", args.few_shot_max_new_tokens)
        max_prompt_len = model_config.get("max_prompt_length", MAX_PROMPT_LENGTH)
        enable_thinking = model_config.get("enable_thinking", False)

        model = None
        tokenizer = None
        try:
            model, tokenizer = load_few_shot_model(model_config)
            for mode in args.few_shot_modes:
                print(f"\n{'=' * 80}\n{model_key} | {mode}\n{'=' * 80}")
                try:
                    for fold_data in folds:
                        fold = fold_data["fold"]
                        key = ("FewShot", model_key, mode, fold)
                        if key in completed:
                            print(f"[RESUME] Skipping completed fold: {key}")
                            continue
                        set_seed(fold_data["seed"])
                        test_df = fold_data["test"]
                        examples = sample_shot_examples(fold_data, pools, mode)
                        prompts = [
                            format_prompt(
                                tokenizer,
                                build_prompt(message, examples),
                                enable_thinking=enable_thinking,
                            )
                            for message in test_df["message"].tolist()
                        ]
                        predictions = []
                        raw_responses = []
                        try:
                            for start in range(0, len(prompts), fs_batch_size):
                                responses = predict_batch(
                                    model,
                                    tokenizer,
                                    prompts[start : start + fs_batch_size],
                                    fs_max_tokens,
                                    max_prompt_length=max_prompt_len,
                                )
                                raw_responses.extend(responses)
                                predictions.extend(parse_prediction(response) for response in responses)
                                print(
                                    f"\r[FOLD {fold}] "
                                    f"{min(start + fs_batch_size, len(prompts))}/{len(prompts)}",
                                    end="",
                                    flush=True,
                                )
                            print()
                        except BaseException as exc:
                            if is_oom_exception(exc):
                                print(
                                    f"\n[OOM-GUARD] OOM encountered during Few-Shot {key}; restarting worker...",
                                    flush=True,
                                )
                                sys.exit(EXIT_OOM_RETRY)
                            raise

                        labels = test_df["label"].to_numpy(dtype=int)
                        metrics = calculate_metrics(labels, predictions)
                        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
                        row = {
                            "RunType": "FewShot",
                            "Model": model_key,
                            "Config": mode,
                            "Fold": fold,
                            "Train_Size": 0,
                            "Val_Size": 0,
                            "Test_Size": len(test_df),
                            **metrics,
                            "CM": json.dumps(cm.tolist()),
                            "unparsed": int(sum(value == -1 for value in predictions)),
                        }
                        prediction_path = (
                            predictions_dir
                            / f"{model_key}_{mode}_fold_{fold}.csv"
                        )
                        save_few_shot_predictions(
                            prediction_path, test_df, predictions, raw_responses
                        )
                        append_raw_row(raw_path, row)
                        raw_rows.append(row)
                        completed.add(key)
                        update_summary(raw_rows, summary_path)
                        print(f"[TEST] {metrics}; unparsed={row['unparsed']}")
                        cleanup_memory()

                except SkipConfiguration:
                    update_summary(raw_rows, summary_path)
                    print(
                        f"\n[CTRL+C] Few-Shot mode {mode} skipped via single Ctrl+C.",
                        flush=True,
                    )
                    cleanup_memory()
                    continue

        finally:
            del model, tokenizer
            cleanup_memory()


def print_dry_run(folds, pools, args):
    models_config = args.models_config_dict
    print("[DRY-RUN] Models and tokenizers are not loaded; GPU is not utilized.")
    print(f"[DRY-RUN] Model configuration file: {args.models_config}")
    print(f"[DRY-RUN] Models scheduled for training (TrainEval): {args.models}")
    for m in args.models:
        cfg = models_config[m]
        print(f"  - {m}: type={cfg['type']}, model_id={cfg['model_id']}, batch={cfg['train_batch_size']}, lr={cfg['learning_rate']}")
    print(f"[DRY-RUN] Models scheduled for Few-Shot: {args.few_shot_models}")

    for fold_data in folds:
        fold = fold_data["fold"]
        print(
            f"Fold {fold}: real train ham={len(fold_data['real_ham_train'])}, "
            f"spam={len(fold_data['real_spam_train'])}, phish=200; "
            f"Val={len(fold_data['val'])}; Test={len(fold_data['test'])}"
        )
        for config_name in args.configs:
            train_df = build_train_dataframe(config_name, fold_data, pools)
            counts = train_df["label"].value_counts().sort_index().to_dict()
            print(f"  {config_name}: total={len(train_df)}, classes={counts}")
        for mode in PROMPT_MODES:
            examples = sample_shot_examples(fold_data, pools, mode)
            print(f"  {mode}: shots={len(examples)}")


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # 1. Preliminary argument parsing to determine models configuration JSON path
    # allow_abbrev=False is essential to prevent --models being parsed as prefix of --models-config
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument(
        "--models-config",
        default=str(PROJECT_ROOT / "configs" / "models_config.json"),
        help="Path to models configuration JSON file",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)

    models_config = load_models_config(pre_args.models_config)
    available_model_keys = list(models_config.keys())
    default_train_models = [
        k for k, v in models_config.items()
        if v.get("enabled", True) and v.get("supports_train_eval", True)
    ]
    few_shot_eligible_keys = [
        k for k, v in models_config.items()
        if v.get("supports_few_shot", False)
    ]

    parser = argparse.ArgumentParser(
        description="Unified 5-fold cross-validation pipeline for classifier fine-tuning and Zero/Few-Shot evaluation",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--models-config",
        default=pre_args.models_config,
        help="Path to models configuration JSON file (default: configs/models_config.json)",
    )
    parser.add_argument("--data-dir", default=None, help="Path to data directory (default: data/processed, with fallback to data)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--best-models-dir", default=str(PROJECT_ROOT / "saved_best_models")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_model_keys,
        default=default_train_models,
        help=f"Models to train (available: {available_model_keys})",
    )
    parser.add_argument("--configs", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument(
        "--few-shot-models",
        nargs="+",
        choices=available_model_keys,
        default=None,
        help=f"Models for Few-Shot evaluation (supported: {few_shot_eligible_keys})",
    )
    parser.add_argument(
        "--few-shot-modes", nargs="+", choices=PROMPT_MODES, default=PROMPT_MODES
    )
    parser.add_argument("--few-shot-batch-size", type=int, default=32)
    parser.add_argument("--few-shot-max-new-tokens", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-few-shot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag for worker subprocess (managed by supervisor)",
    )
    parser.add_argument(
        "--oom-safe",
        action="store_true",
        help="Enabled automatically when restarting after OOM",
    )
    parser.add_argument(
        "--max-folds-per-worker",
        type=int,
        default=3,
        help="Number of LLM folds before restarting worker process to reset VRAM to 0 MiB",
    )
    args = parser.parse_args(argv)
    args.models_config_dict = models_config

    if args.few_shot_models is None:
        args.few_shot_models = [
            m for m in args.models
            if models_config[m].get("supports_few_shot", False)
        ]

    if args.few_shot_batch_size < 1 or args.few_shot_max_new_tokens < 1:
        parser.error("Few-Shot batch size and max new tokens must be positive integers")
    return args


def run_supervisor(args):
    """
    Subprocess supervisor:
    - Periodically restarts worker after N folds to release VRAM to 0 MiB via the NVIDIA driver.
    - Captures CUDA OOM (return code 99) and restarts worker under --oom-safe profile.
    - Captures double Ctrl+C (code 130) and terminates execution gracefully.
    """
    target_script = Path(__file__).resolve()
    base_cmd = [sys.executable, str(target_script), "--worker"] + [
        arg for arg in sys.argv[1:] if arg not in ("--worker", "--oom-safe")
    ]
    oom_safe_next = args.oom_safe
    consecutive_failures = 0
    max_failures = 3

    while True:
        cmd = list(base_cmd)
        if oom_safe_next:
            cmd.append("--oom-safe")

        try:
            proc = subprocess.run(cmd)
            rc = proc.returncode
        except KeyboardInterrupt:
            print("\n[SUPERVISOR] Interrupted by user (KeyboardInterrupt).", flush=True)
            return

        if rc == 0:
            print("\n[SUPERVISOR] All training and Few-Shot evaluation tasks completed successfully!")
            return
        elif rc == EXIT_USER_ABORT or rc == 130:
            print("\n[SUPERVISOR] Aborted by user (double Ctrl+C). Terminating.", flush=True)
            return
        elif rc == EXIT_CONTINUE:
            oom_safe_next = False
            consecutive_failures = 0
            time.sleep(2)
            continue
        elif rc == EXIT_OOM_RETRY:
            print(
                "\n[SUPERVISOR] Detected OOM exit code (99). Releasing VRAM "
                "and restarting worker with protective --oom-safe profile...",
                flush=True,
            )
            oom_safe_next = True
            time.sleep(4)
            continue
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(
                    f"\n[SUPERVISOR] Worker failed with unexpected return code {rc} "
                    f"({consecutive_failures} consecutive failures). Aborting execution to prevent infinite loop.",
                    flush=True,
                )
                sys.exit(rc)
            print(
                f"\n[SUPERVISOR] Worker exited with code {rc} "
                f"(failure {consecutive_failures}/{max_failures}). "
                f"Restarting in 5 seconds while preserving checkpoint progress...",
                flush=True,
            )
            oom_safe_next = True
            time.sleep(5)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.best_models_dir).mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "ALL_FOLDS_RAW.csv"
    summary_path = output_dir / "FINAL_SUMMARY_METRICS.csv"

    if args.dry_run:
        pools = load_verified_pools(args.data_dir)
        folds = build_folds(pools)
        print_dry_run(folds, pools, args)
        return

    if not args.worker:
        run_supervisor(args)
        return

    # Install signal handlers in worker process
    signal.signal(signal.SIGINT, handle_sigint)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Use --dry-run for schema validation.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU with bfloat16 support is required for this pipeline.")

    pools = load_verified_pools(args.data_dir)
    folds = build_folds(pools)
    raw_rows = read_raw_results(raw_path)
    update_summary(raw_rows, summary_path)

    try:
        if not args.skip_training:
            run_training(args, pools, folds, raw_rows, raw_path, summary_path)
        if not args.skip_few_shot:
            run_few_shot(args, pools, folds, raw_rows, raw_path, summary_path)
    except StopAllExecution:
        update_summary(raw_rows, summary_path)
        print("\n[STOP] Received double Ctrl+C. Halting execution and saving results.", flush=True)
        cleanup_memory()
        sys.exit(EXIT_USER_ABORT)

    update_summary(raw_rows, summary_path)
    print(f"\n[SUCCESS] Results written to: {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
