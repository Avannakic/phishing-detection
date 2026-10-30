"""Последовательный запуск этапов подготовки данных и, по запросу, обучения."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATION_STEPS = [
    "00_generate_ham_and_spam.py",
    "01_generate_base_phishing.py",
    "02_generate_targeted_addition.py",
]
PROCESSING_STEPS = [
    "03_clean_and_filter.py",
    "04_merge_synthetic_pools.py",
    "05_procedural_urls.py",
    "06_finalise_datasets.py",
]
TRAINING_STEP = "07_train_rubert_and_results.py"
FEW_SHOT_STEP = "08_llm_few_shot_cv.py"


def run_step(script_path: Path):
    print(f"\n>>> ЗАПУСК: {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=False
    )
    if result.returncode != 0:
        print(
            f"[ERROR] Скрипт {script_path.name} завершился с ошибкой "
            f"(код {result.returncode}). Остановка пайплайна."
        )
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Оркестратор исследовательского пайплайна")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Включить долгие шаги генерации 00–02 через Ollama",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="После подготовки данных запустить обучение всех моделей (шаг 07)",
    )
    parser.add_argument(
        "--include-few-shot",
        action="store_true",
        help="После подготовки данных запустить Few-Shot оценку (шаг 08)",
    )
    parser.add_argument("--scripts-dir", default="src", help="Папка со скриптами пайплайна")
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir)
    if not scripts_dir.is_absolute():
        scripts_dir = PROJECT_ROOT / scripts_dir
    steps = []
    if args.include_generation:
        steps.extend(GENERATION_STEPS)
    steps.extend(PROCESSING_STEPS)
    if args.include_training:
        steps.append(TRAINING_STEP)
    if args.include_few_shot:
        steps.append(FEW_SHOT_STEP)

    print(f"Старт пайплайна. Всего шагов: {len(steps)}")
    for script_name in steps:
        script_path = scripts_dir / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Скрипт не найден: {script_path}")
        run_step(script_path)
    print("\n[SUCCESS] Все выбранные этапы выполнены.")


if __name__ == "__main__":
    main()
