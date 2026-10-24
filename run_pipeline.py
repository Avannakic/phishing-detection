"""
Мастер-скрипт (оркестратор) для последовательного запуска пайплайна подготовки данных.
По умолчанию выполняет быстрые шаги очистки, верификации и сборки (Шаги 03-06).
Для запуска с генерацией в Ollama используйте флаг: python run_pipeline.py --include-generation
"""

import argparse
import subprocess
import sys
from pathlib import Path

GENERATION_STEPS = [
    "01_generate_base_phishing.py",
    "02_generate_targeted_add.py",
]

PROCESSING_STEPS = [
    "03_clean_and_filter.py",
    "04_merge_synthetic_pools.py",
    "05_procedural_urls.py",
    "06_build_training_datasets.py",
]


def run_step(script_path: Path):
    print(f"\n>>> ЗАПУСК: {script_path.name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print(f"[ERROR] Скрипт {script_path.name} завершился с ошибкой (код {result.returncode}). Остановка пайплайна.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Запуск пайплайна подготовки синтетических и гибридных данных")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Включить долгие шаги генерации через Ollama (01 и 02)"
    )
    parser.add_argument(
        "--scripts-dir",
        default="src/data_generation",
        help="Папка со скриптами пайплайна"
    )
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir)
    steps_to_run = (GENERATION_STEPS + PROCESSING_STEPS) if args.include_generation else PROCESSING_STEPS

    print("=" * 60)
    print(f"Старт пайплайна. Всего шагов к выполнению: {len(steps_to_run)}")
    print("=" * 60)

    for script_name in steps_to_run:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            # Если мастер-скрипт лежит в одной папке со скриптами
            script_path = Path(script_name)
        run_step(script_path)

    print("\n[SUCCESS] Все этапы пайплайна успешно выполнены!")


if __name__ == "__main__":
    main()