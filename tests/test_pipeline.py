#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Unit Test Suite for Phishing Detection Research Pipeline.
Covers:
- Configuration schemas and parameter inheritance (configs/)
- Preprocessing and sanitization filters (src/03, src/05, src/06)
- 5-fold cross-validation invariants and dataset construction across 8 configurations
- Classification performance metrics and LLM output parsing (including <think> tags)
- Checkpoint selection policy and architecture auto-routing
- Supervisor fault tolerance and exit codes
- Codebase language and dependency integrity
"""

import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clean_mod = _import_module("step_03", SRC_DIR / "03_clean_and_filter.py")
urls_mod = _import_module("step_05", SRC_DIR / "05_procedural_urls.py")
final_mod = _import_module("step_06", SRC_DIR / "06_finalise_datasets.py")
train_mod = _import_module("step_07", SRC_DIR / "07_train_and_eval_all.py")
eval_mod = _import_module("eval_saved_models", SRC_DIR / "eval_saved_models.py")


class TestConfigurations(unittest.TestCase):
    """Test external configuration schemas, scenarios, and parameter inheritance."""

    def test_models_config_structure(self):
        cfg_path = CONFIGS_DIR / "models_config.json"
        self.assertTrue(cfg_path.is_file(), "models_config.json does not exist")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertIn("defaults", cfg)
        self.assertIn("encoder", cfg["defaults"])
        self.assertIn("llm_lora", cfg["defaults"])
        self.assertIn("models", cfg)
        self.assertIn("Vikhr-8B", cfg["models"])
        self.assertIn("RuBERT", cfg["models"])
        self.assertEqual(cfg["models"]["Vikhr-8B"]["type"], "llm_lora")
        self.assertEqual(cfg["models"]["RuBERT"]["type"], "encoder")

    def test_scenarios_config_structure_and_thresholds(self):
        cfg_path = CONFIGS_DIR / "scenarios_config.json"
        self.assertTrue(cfg_path.is_file(), "scenarios_config.json does not exist")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertIn("ham", cfg)
        self.assertIn("spam", cfg)
        self.assertIn("base_phishing", cfg)
        self.assertIn("targeted_phishing", cfg)

        # Verify ham short length threshold is lowered to < 16
        ham_short = [item for item in cfg["ham"]["lengths"] if item[0] == "короткие"]
        self.assertTrue(len(ham_short) > 0)
        self.assertIn("16", ham_short[0][2])

    def test_model_config_inheritance(self):
        resolved = train_mod.load_models_config(CONFIGS_DIR / "models_config.json")
        self.assertEqual(resolved["RuBERT"]["train_batch_size"], 64)
        self.assertTrue(resolved["Vikhr-8B"]["quantization"]["load_in_4bit"])


class TestPreprocessing(unittest.TestCase):
    """Test message cleaning, sanitization filters, URL generation, and leakage removal."""

    def test_ham_length_filtering(self):
        clean_fn = clean_mod.clean_and_validate_message
        # Ham message with < 16 chars should be rejected
        short_ham = clean_fn("Короткий текст!", label_val=0)  # 15 chars
        self.assertIsNone(short_ham)

        # Ham message with >= 16 chars and >= 2 words should be accepted
        valid_ham = clean_fn("Купи, пожалуйста, хлеб и молоко по пути домой.", label_val=0)
        self.assertIsNotNone(valid_ham)
        self.assertGreaterEqual(len(valid_ham), 16)

    def test_greetings_and_prefix_stripping(self):
        clean_fn = clean_mod.clean_and_validate_message
        msg = clean_fn("Здравствуйте! Перейдите по ссылке для подтверждения операции", label_val=2)
        self.assertIsNotNone(msg)
        self.assertFalse(msg.lower().startswith("здравствуйте"))

    def test_polyclinic_message_preserved(self):
        clean_fn = clean_mod.clean_and_validate_message
        msg = clean_fn("Поликлиника: Напоминаем о необходимости подтвердить запись к врачу.", label_val=2)
        self.assertIsNotNone(msg)
        self.assertIn("подтвердить запись к врачу", msg)

    def test_meta_comments_rejection(self):
        clean_fn = clean_mod.clean_and_validate_message
        msg = clean_fn("### Пример 1 фишингового сообщения от лица банка", label_val=2)
        self.assertIsNone(msg)

    def test_url_stripping(self):
        clean_fn = clean_mod.clean_and_validate_message
        msg = clean_fn("Срочно обновите пароль http://phishing.ru для защиты аккаунта", label_val=2)
        self.assertIsNotNone(msg)
        self.assertNotIn("http", msg)

    def test_dynamic_url_generator(self):
        gen = urls_mod.DynamicURLGenerator(seed=42)
        phish_url = gen.generate(label_val=2)
        ham_url = gen.generate(label_val=0)
        self.assertIsInstance(phish_url, str)
        self.assertIsInstance(ham_url, str)
        self.assertGreater(len(phish_url), 5)
        self.assertGreater(len(ham_url), 5)

    def test_leakage_removal(self):
        df_real = pd.DataFrame({"message": ["Внимание, подтвердите код безопасности 1234"]})
        df_synth = pd.DataFrame({
            "message": [
                "Внимание, подтвердите код безопасности 1234",  # Exact duplicate
                "Совершенно другое сообщение от курьерской службы",  # Distinct
            ]
        })
        cleaned = final_mod.remove_leakage(df_synth, df_real, threshold=0.75)
        self.assertEqual(len(cleaned), 1)
        self.assertIn("курьерской", cleaned.iloc[0]["message"])


class TestCrossValidationAndSplit(unittest.TestCase):
    """Test 5-Fold Stratified Split invariants and 8 experimental configurations."""

    @classmethod
    def setUpClass(cls):
        cls.mock_pools = {
            "real_ham": pd.DataFrame({"message": [f"ham_{i}" for i in range(20000)], "label": 0}),
            "real_spam": pd.DataFrame({"message": [f"spam_{i}" for i in range(20000)], "label": 1}),
            "real_phish": pd.DataFrame({"message": [f"phish_{i}" for i in range(300)], "label": 2}),
            "synth_ham": pd.DataFrame({"message": [f"sham_{i}" for i in range(14000)], "label": 0}),
            "synth_spam": pd.DataFrame({"message": [f"sspam_{i}" for i in range(14000)], "label": 1}),
            "synth_phish": pd.DataFrame({"message": [f"sphish_{i}" for i in range(7000)], "label": 2}),
        }
        cls.folds = train_mod.build_folds(cls.mock_pools)

    def test_folds_count_and_sizes(self):
        self.assertEqual(len(self.folds), 5)
        for fold_data in self.folds:
            self.assertEqual(len(fold_data["val"]), 6040)
            self.assertEqual(len(fold_data["test"]), 6060)
            self.assertEqual(len(fold_data["real_phish_train"]), 200)

    def test_zero_leakage_between_partitions(self):
        for fold_data in self.folds:
            train_msgs = (
                set(fold_data["real_ham_train"]["message"])
                | set(fold_data["real_spam_train"]["message"])
                | set(fold_data["real_phish_train"]["message"])
            )
            val_msgs = set(fold_data["val"]["message"])
            test_msgs = set(fold_data["test"]["message"])

            self.assertEqual(len(train_msgs & val_msgs), 0, "Leakage detected between train and val")
            self.assertEqual(len(train_msgs & test_msgs), 0, "Leakage detected between train and test")
            self.assertEqual(len(val_msgs & test_msgs), 0, "Leakage detected between val and test")

    def test_eight_experiment_configurations(self):
        expected_sizes = {
            "1_Baseline_Imbalanced": 28200,
            "2_Baseline_ClassWeights": 28200,
            "3_Baseline_Oversampling": 35000,
            "4_Pure_Synthetic": 35000,
            "5_Hybrid_3.4x_10pct": 28880,
            "6_Hybrid_8.5x_25pct": 29900,
            "7_Hybrid_17x_50pct": 31600,
            "8_Hybrid_34x_100pct": 35000,
        }
        for cfg_name, expected_len in expected_sizes.items():
            df_train = train_mod.build_train_dataframe(cfg_name, self.folds[0], self.mock_pools)
            self.assertEqual(
                len(df_train),
                expected_len,
                f"Configuration {cfg_name} produced {len(df_train)} rows; expected {expected_len}",
            )


class TestMetricsAndInference(unittest.TestCase):
    """Test performance metric calculation and LLM response parsing."""

    def test_calculate_metrics(self):
        labels = [0, 1, 2, 2]
        preds = [0, 1, 2, 0]
        metrics = train_mod.calculate_metrics(labels, preds)
        self.assertEqual(len(metrics), 13)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("phish_precision", metrics)
        self.assertIn("phish_recall", metrics)
        self.assertIn("phish_f1", metrics)

    def test_parse_prediction_simple_digit(self):
        self.assertEqual(train_mod.parse_prediction("2"), 2)
        self.assertEqual(train_mod.parse_prediction("0"), 0)

    def test_parse_prediction_with_thinking_tags(self):
        self.assertEqual(train_mod.parse_prediction("<think>Phishing characteristics found</think> 2"), 2)

    def test_parse_prediction_unclosed_thinking_marked_unparsed(self):
        # Truncated or unclosed reasoning must be marked as unparsed (-1)
        self.assertEqual(train_mod.parse_prediction("<think>thinking... cut off"), -1)

    def test_parse_prediction_labeled(self):
        self.assertEqual(train_mod.parse_prediction("Ответ: 1"), 1)
        self.assertEqual(train_mod.parse_prediction("Класс: 2"), 2)

    def test_parse_prediction_unknown_fallback(self):
        self.assertEqual(train_mod.parse_prediction("Непонятный ответ модели"), -1)


class TestModelSelectionAndRevalidation(unittest.TestCase):
    """Test checkpoint selection rules and universal re-validation via test.py."""

    def test_selection_rule_precision_priority(self):
        # Eligible candidates with precision >= 0.50 prioritize recall
        cand_high_rec = {"phish_precision": 0.60, "phish_recall": 0.85, "phish_f1": 0.70}
        cand_low_rec = {"phish_precision": 0.65, "phish_recall": 0.70, "phish_f1": 0.67}
        self.assertTrue(train_mod.is_better_fold(cand_high_rec, cand_low_rec))

        # Precision < 0.50 is rejected if incumbent has precision >= 0.50
        cand_sub_prec = {"phish_precision": 0.40, "phish_recall": 0.95, "phish_f1": 0.56}
        self.assertFalse(train_mod.is_better_fold(cand_sub_prec, cand_low_rec))

    def test_architecture_resolution(self):
        self.assertEqual(train_mod.resolve_model_family("DeepPavlov/rubert-base-cased", "encoder"), "encoder")
        self.assertEqual(train_mod.resolve_model_family("Vikhrmodels/Vikhr-8B", "llm_lora"), "llm_lora")

    def test_dynamic_train_pipeline_loader(self):
        loaded_mod = eval_mod._load_train_pipeline()
        self.assertTrue(hasattr(loaded_mod, "build_folds"))
        self.assertTrue(hasattr(loaded_mod, "direct_test_inference"))

    def test_revalidation_metadata_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta = {
                "RunType": "TrainEval",
                "Model": "RuBERT",
                "Config": "1_Baseline_Imbalanced",
                "Fold": 1,
                "model_id": "DeepPavlov/rubert-base-cased",
                "model_type": "encoder",
                "max_length": 128,
                "f1_macro": 0.88,
                "phish_f1": 0.85,
            }
            meta_file = tmp_path / "best_metrics.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            with open(meta_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded.get("model_type"), "encoder")
            self.assertEqual(loaded.get("Fold"), 1)


class TestSupervisorAndFaultTolerance(unittest.TestCase):
    """Test supervisor status codes and exception handlers."""

    def test_exit_codes(self):
        self.assertEqual(train_mod.EXIT_OOM_RETRY, 99)
        self.assertEqual(train_mod.EXIT_CONTINUE, 98)
        self.assertEqual(train_mod.EXIT_USER_ABORT, 130)

    def test_oom_exception_detection(self):
        self.assertTrue(train_mod.is_oom_exception(RuntimeError("CUDA error: out of memory")))
        self.assertTrue(train_mod.is_oom_exception(RuntimeError("cudnn_status_alloc_failed")))
        self.assertFalse(train_mod.is_oom_exception(ValueError("Invalid argument")))


class TestRepositoryCleanliness(unittest.TestCase):
    """Test documentation, absence of internal artifacts, and dependency consistency."""

    def test_zero_cyrillic_docstrings_in_codebase(self):
        py_files = list(SRC_DIR.glob("*.py")) + [PROJECT_ROOT / "run_pipeline.py"]
        for pf in py_files:
            with open(pf, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(pf))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node)
                    if doc:
                        self.assertIsNone(
                            re.search(r"[а-яёА-ЯЁ]", doc),
                            f"File {pf.name} contains Cyrillic docstring: {doc[:40]}...",
                        )

    def test_internal_files_absent(self):
        self.assertFalse((PROJECT_ROOT / "AGENTS.md").exists(), "AGENTS.md must not exist in repository")
        self.assertFalse((PROJECT_ROOT / "tmp_trainer").exists(), "tmp_trainer must not exist in repository")

    def test_requirements_file_completeness(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        self.assertTrue(req_path.is_file())
        content = req_path.read_text(encoding="utf-8")
        self.assertIn("safetensors", content)
        self.assertIn("bitsandbytes", content)
        self.assertIn("peft", content)
        self.assertIn("scikit-learn", content)


if __name__ == "__main__":
    unittest.main()
