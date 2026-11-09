"""
Sequential pipeline orchestrator for data generation, preprocessing, model fine-tuning, and evaluation.
"""

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
TRAINING_STEP = "07_train_and_eval_all.py"


def run_step(script_path: Path, extra_args=None):
    """Executes a pipeline step in a subprocess with working directory at PROJECT_ROOT."""
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    cmd_str = f"{script_path.name}" + (f" {' '.join(extra_args)}" if extra_args else "")
    print(f"\n>>> RUNNING: {cmd_str}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(
            f"[ERROR] Script {script_path.name} terminated with error "
            f"(exit code {result.returncode}). Pipeline aborted."
        )
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Research pipeline master orchestrator")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Include synthetic generation steps 00-02 via Ollama",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Run model fine-tuning after preprocessing (step 07)",
    )
    parser.add_argument(
        "--include-few-shot",
        action="store_true",
        help="Run Few-Shot evaluation after preprocessing (step 07)",
    )
    parser.add_argument("--scripts-dir", default="src", help="Path to directory containing pipeline scripts")
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir)
    if not scripts_dir.is_absolute():
        scripts_dir = PROJECT_ROOT / scripts_dir

    steps = []
    if args.include_generation:
        steps.extend(GENERATION_STEPS)
    steps.extend(PROCESSING_STEPS)

    print(f"Starting pipeline execution. Total preprocessing steps: {len(steps)}")
    for script_name in steps:
        script_path = scripts_dir / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")
        run_step(script_path)

    if args.include_training or args.include_few_shot:
        train_script = scripts_dir / TRAINING_STEP
        if not train_script.is_file():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        if args.include_training and args.include_few_shot:
            run_step(train_script)
        elif args.include_training:
            run_step(train_script, ["--skip-few-shot"])
        elif args.include_few_shot:
            run_step(train_script, ["--skip-training"])

    print("\n[SUCCESS] All selected pipeline stages completed successfully.")


if __name__ == "__main__":
    main()
