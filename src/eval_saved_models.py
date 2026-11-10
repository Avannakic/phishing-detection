#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-validation script for saved best model checkpoints in saved_best_models.
Universally compatible with all models configured in configs/models_config.json.
Automatically routes loading for encoder (BERT/RoBERTa) vs llm_lora (Vikhr/Qwen/Llama).
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# Ensure access to src modules regardless of working directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_train_pipeline():
    candidates = sorted(SRC_DIR.glob("*train_and_eval_all*.py"))
    if not candidates:
        raise FileNotFoundError(f"Training pipeline script (*train_and_eval_all*.py) not found in {SRC_DIR}")
    script_path = candidates[0]
    spec = importlib.util.spec_from_file_location("pipeline_train_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pipeline = _load_train_pipeline()
PROJECT_ROOT = _pipeline.PROJECT_ROOT
build_folds = _pipeline.build_folds
cleanup_memory = _pipeline.cleanup_memory
direct_test_inference = _pipeline.direct_test_inference
load_verified_pools = _pipeline.load_verified_pools
save_predictions = _pipeline.save_predictions

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-validation of saved best model checkpoints",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--best-models-dir",
        default=str(PROJECT_ROOT / "saved_best_models"),
        help="Path to saved best models directory",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory (default: data/processed, with fallback to data)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter by model names from configs/models_config.json",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Filter by experimental configurations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for direct inference",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "results" / "REEVAL_BEST_FOLD_METRICS.csv"),
        help="Output CSV path for re-validation metrics summary",
    )
    return parser.parse_args(argv)


def load_saved_best_model(model_dir: Path, meta: dict):
    """
    Universal model loader by architecture type defined in metadata or adapter_config.json,
    without hardcoded model identifiers.
    """
    model_type = meta.get("model_type")
    adapter_config_path = model_dir / "adapter_config.json"
    if not model_type:
        model_type = "llm_lora" if adapter_config_path.is_file() else "encoder"

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "[PAD]"
    tok.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "encoder":
        print(f"[LOAD] Loading encoder model from {model_dir}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device).eval()
        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        return model, tok

    elif model_type == "llm_lora":
        from peft import PeftModel

        base_model_id = meta.get("model_id")
        if not base_model_id and adapter_config_path.is_file():
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path")

        if not base_model_id:
            raise ValueError(f"Could not determine base_model_id for checkpoint {model_dir}")

        q_cfg = meta.get("quantization") or {}
        use_4bit = torch.cuda.is_available() and q_cfg.get("load_in_4bit", True)
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
            )
            if use_4bit
            else None
        )

        print(f"[LOAD] Loading base LLM {base_model_id}...")
        base_kwargs = {
            "num_labels": 3,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if bnb_config is not None:
            base_kwargs["quantization_config"] = bnb_config
            base_kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            base_kwargs["device_map"] = "auto"

        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            **base_kwargs,
        )
        if not torch.cuda.is_available():
            base = base.to(device)
        base.config.pad_token_id = tok.pad_token_id

        print(f"[LOAD] Attaching LoRA adapter from {model_dir}...")
        model = PeftModel.from_pretrained(base, str(model_dir)).eval()
        if not torch.cuda.is_available():
            model = model.to(device)
        model.config.pad_token_id = tok.pad_token_id
        return model, tok

    else:
        raise ValueError(f"Unknown model_type: {model_type} for {model_dir}")


def main():
    args = parse_args()
    best_models_root = Path(args.best_models_dir)
    if not best_models_root.is_dir():
        print(f"[ERROR] Saved models directory not found: {best_models_root}")
        sys.exit(1)

    print("[INIT] Loading verified data pools and building folds...")
    pools = load_verified_pools(args.data_dir)
    folds = build_folds(pools)

    # Search for all subdirectories matching model_name / config_name
    model_dirs = []
    for model_path in sorted(best_models_root.iterdir()):
        if not model_path.is_dir():
            continue
        model_name = model_path.name
        if args.models and model_name not in args.models:
            continue
        for config_path in sorted(model_path.iterdir()):
            if not config_path.is_dir():
                continue
            config_name = config_path.name
            if args.configs and config_name not in args.configs:
                continue
            model_dirs.append((model_name, config_name, config_path))

    if not model_dirs:
        print(f"[WARN] No models matching filters found in {best_models_root}.")
        return

    print(f"[INFO] Discovered models for re-validation: {len(model_dirs)}")
    revalidation_rows = []
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_csv.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for model_name, config_name, model_dir in model_dirs:
        print(f"\n{'=' * 80}\nRe-validating: {model_name} | {config_name}\nPath: {model_dir}\n{'=' * 80}")
        best_metrics_file = model_dir / "best_metrics.json"
        saved_metadata = {}
        target_fold = 1
        if best_metrics_file.is_file():
            with open(best_metrics_file, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
            target_fold = int(saved_metadata.get("Fold", 1))
            print(f"[INFO] According to best_metrics.json, model was trained on Fold {target_fold}")

        fold_data = folds[target_fold - 1]
        test_df = fold_data["test"]
        max_length = int(saved_metadata.get("max_length", 128))

        model = None
        tokenizer = None
        try:
            model, tokenizer = load_saved_best_model(model_dir, saved_metadata)
            metrics, cm, predictions = direct_test_inference(
                model=model,
                tokenizer=tokenizer,
                test_df=test_df,
                batch_size=args.batch_size,
                max_length=max_length,
            )

            row = {
                "Model": model_name,
                "Config": config_name,
                "Fold": target_fold,
                "Test_Size": len(test_df),
                **metrics,
                "CM": json.dumps(cm.tolist()),
            }

            if saved_metadata:
                orig_f1 = float(saved_metadata.get("f1_macro", 0.0))
                orig_phish_f1 = float(saved_metadata.get("phish_f1", 0.0))
                print(
                    f"[COMPARE] Macro F1: {metrics['f1_macro']:.4f} (original: {orig_f1:.4f}) | "
                    f"Phish F1: {metrics['phish_f1']:.4f} (original: {orig_phish_f1:.4f})"
                )

            revalidation_rows.append(row)
            pred_path = predictions_dir / f"reval_{model_name}_{config_name}_fold_{target_fold}.csv"
            save_predictions(pred_path, test_df, predictions)
            print(f"[RESULT] {metrics}")

        except Exception as exc:
            print(f"[ERROR] Error during re-validation of {model_name} / {config_name}: {exc}")
        finally:
            del model, tokenizer
            cleanup_memory()

    if revalidation_rows:
        reval_df = pd.DataFrame(revalidation_rows)
        reval_df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n[SUCCESS] Re-validation summary written to {output_csv}")


if __name__ == "__main__":
    main()
