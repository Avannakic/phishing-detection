# Russian Phishing Detection: Synthetic Data Generation, QLoRA, and Few-Shot Evaluation

A research pipeline for classifying Russian-language messenger messages into three mutually exclusive categories:

- `0` — Legitimate personal or operational message (**Ham**);
- `1` — Commercial unsolicited advertisement (**Spam**);
- `2` — Phishing, credential harvesting, and social engineering (**Phishing**).

This repository integrates Russian synthetic text generation via Ollama, decoupled scenario and model configurations in `configs/`, artifact cleaning, procedural URL injection to eliminate feature leakage, 5-fold cross-validation across encoder and decoder LLM architectures using 4-bit QLoRA, and Zero-Shot / Few-Shot LLM evaluation without parameter updates.

---

## 1. Data Lineage and Pool Quotas

All CSV datasets require a `message` column; Step 06 establishes the canonical `label` column. Encoding is strictly UTF-8.

| Dataset Pool | File in `data/processed/` | Label | Target Quota | Source Pipeline Step |
|---|---|:---:|---:|---|
| Real Ham | `real_ham.csv` | `0` | 20,000 | Ingestion from `data/real/real_ham.csv` |
| Real Spam | `real_spam.csv` | `1` | 20,000 | Ingestion from `data/real/real_spam.csv` |
| Real Phishing | `real_phishing.csv` | `2` | 300 | Ingestion from `data/real/real_phishing.csv` |
| Synthetic Ham | `synthetic_ham.csv` | `0` | 14,000 | Generated in 00 -> Cleaned in 03-05 -> Finalized in 06 |
| Synthetic Spam | `synthetic_spam.csv` | `1` | 14,000 | Generated in 00 -> Cleaned in 03-05 -> Finalized in 06 |
| Synthetic Phishing | `synthetic_phishing.csv` | `2` | 7,000 | Generated in 01-02 -> Cleaned in 03-05 -> Finalized in 06 |

### Storage Structure:
- `data/real/`: Raw external corpora (`real_ham.csv`, `real_spam.csv`, `real_phishing.csv`).
- `data/interim/`: Raw generated batches (`raw_synthetic_base.csv`, `raw_synthetic_targeted.csv`), intermediate cleaned files (`cleaned_*.csv`), deduplicated pools (`dedup_*.csv`, `merged_*.csv`), and URL-injected sets (`final_synthetic_*.csv`).
- `data/processed/`: Balanced, leak-free, validated datasets produced exclusively by Step 06.

---

## 2. Decoupled Configurations (`configs/`)

Operational parameters, generation prompts, refusal detection markers, and model training settings are completely decoupled into two central JSON files in `configs/`.

> [!NOTE]
> The JSON schemas and snippets below represent **examples of configuration structure and usage**. The pipeline dynamically parses and resolves these configs at runtime, enabling users to supply custom scenario files (`--config <path>`) or model configs (`--models-config <path>`) without modifying the codebase.

### 2.1. Scenario & Generation Config: [`configs/scenarios_config.json`](configs/scenarios_config.json)

Manages prompts, topics, length distributions, and guardrails for Steps 00, 01, and 02 (overrideable via `--config`). Below is an overview of the schema structure and default usage:

- **`ham`**:
  - `topics_data`: Household groceries, study/work deadlines, weekend travel plans, movie/book discussions;
  - `lengths`: Short (`"короткие"`, 16–50 characters, 1–2 words or 1 phrase), Medium (`"средние"`, up to 150 characters), Long (`"длинные"`, 300–500 characters);
  - `length_weights`: `[0.50, 0.45, 0.05]` (50% short messenger messages);
  - `styles`: Casual messenger style, concise, emoji-supported, on-the-go quick replies.
- **`spam`**:
  - `topics_data`: IT courses, retail seasonal sales, SEO/web analytics services, fitness memberships;
  - `lengths`: Short (up to 80 chars), Medium (up to 200 chars), Long (300–500 chars);
  - `length_weights`: `[0.20, 0.60, 0.20]`;
  - `anti_phishing_rule`: Strict negative constraints prohibiting fake fines, account blocks, bank security alerts, or government portal impersonation.
- **`base_phishing`** & **`targeted_phishing`**:
  - `scenarios`: 10 baseline vectors (online service support, banking security, university administration, peer requests) + 6 targeted spear-phishing vectors (bonus clubs, digital key giveaways, urgent voting);
  - `emoji_rules`: Contextual emoji probabilities and tone rules (e.g., official banking alerts vs. aggressive clickbait);
  - `allowed_lengths`: Scenario-specific length allocations;
  - `link_examples`: Plausible Russian domain/shortener references (`t.me/...`, `vk.cc/...`, `my-portal-auth.ru`).

### 2.2. Model & Routing Config: [`configs/models_config.json`](configs/models_config.json)

Manages architectures, optimization hyperparameters, quantization, and LoRA adapters for Steps 07 and `test.py` (overrideable via `--models-config`). 

The following snippet illustrates the configuration structure and usage pattern:

```json
{
  "defaults": {
    "encoder": {
      "epochs": 3,
      "learning_rate": 2e-5,
      "train_batch_size": 64,
      "eval_batch_size": 128,
      "grad_accum": 1,
      "max_length": 128,
      "optim": "adamw_torch_fused",
      "supports_train_eval": true,
      "supports_few_shot": false,
      "needs_vram_worker_reset": false
    },
    "llm_lora": {
      "epochs": 2,
      "learning_rate": 1e-4,
      "train_batch_size": 32,
      "eval_batch_size": 64,
      "grad_accum": 2,
      "max_length": 128,
      "optim": "paged_adamw_8bit",
      "supports_train_eval": true,
      "supports_few_shot": true,
      "needs_vram_worker_reset": true,
      "few_shot_batch_size": 32,
      "few_shot_max_new_tokens": 8,
      "max_prompt_length": 2048,
      "enable_thinking": false,
      "quantization": {
        "load_in_4bit": true,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": true
      },
      "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save": ["score"]
      }
    }
  },
  "models": {
    "Vikhr-8B": {
      "enabled": true,
      "model_id": "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
      "type": "llm_lora"
    },
    "RuBERT": {
      "enabled": true,
      "model_id": "DeepPavlov/rubert-base-cased",
      "type": "encoder"
    },
    "RuRoBERTa-large": {
      "enabled": true,
      "model_id": "ai-forever/ruRoberta-large",
      "type": "encoder",
      "learning_rate": 1.5e-5
    },
    "Qwen3-14B": {
      "enabled": false,
      "model_id": "Qwen/Qwen3-14B",
      "type": "llm_lora",
      "train_batch_size": 16,
      "eval_batch_size": 32,
      "grad_accum": 4,
      "enable_thinking": false
    }
  }
}
```

---

## 3. End-to-End Pipeline Steps

```text
├── configs/
│   ├── scenarios_config.json         # Generation scenarios, topics, refusal parameters
│   └── models_config.json            # Model registry, architecture routing, hyperparameters
├── src/
│   ├── 00_generate_ham_and_spam.py   # Ham (14k) and Spam (14k) via Qwen3 + VIKHR fallback
│   ├── 01_generate_base_phishing.py  # 10 baseline phishing vectors (14k raw)
│   ├── 02_generate_targeted_addition.py # 6 targeted spear-phishing vectors (6k raw)
│   ├── 03_clean_and_filter.py        # Deep regex artifact, lead junk, and length filtering
│   ├── 04_merge_synthetic_pools.py   # Cross-class and TF-IDF semantic deduplication
│   ├── 05_procedural_urls.py         # Procedural URL injection (prevents URL-leakage)
│   ├── 06_finalise_datasets.py       # Pool balancing, leakage purge against real phishing
│   ├── 07_train_and_eval_all.py      # Unified 5-fold CV (BERT, LoRA) and Few-Shot evaluation
│   └── test.py                       # Standalone re-validation of saved best checkpoints
└── run_pipeline.py                   # Master pipeline orchestrator
```

### Detailed Execution Flow:

#### Step 00: `00_generate_ham_and_spam.py`
- Generates 14,000 Ham and 14,000 Spam messages via Ollama (`qwen3:14b` with fallback to `vikhr-llama3.1-8b`).
- Monitors refusal indicators (`я не могу`, `as an ai`, etc.) and short responses (`< 25` characters). Switches permanently to fallback if refusal rate exceeds 65% across 20 calls or on 3 consecutive refusals.
- Output: `data/interim/synthetic_ham.csv`, `data/interim/synthetic_spam.csv`.

#### Step 01: `01_generate_base_phishing.py`
- Generates 14,000 baseline phishing samples across 10 scenarios.
- Output: `data/interim/raw_synthetic_base.csv`.

#### Step 02: `02_generate_targeted_addition.py`
- Generates 6,000 targeted spear-phishing samples across 6 focused scenarios.
- Output: `data/interim/raw_synthetic_targeted.csv`.

#### Step 03: `03_clean_and_filter.py`
- Strips Markdown bold formatting, URLs, AI meta-talk, and sender headers (`PREFIX_STRIP`).
- Validates Cyrillic character density (minimum 10 Cyrillic chars for Ham, 15 for Spam/Phishing).
- **Length cutoffs**:
  - **Ham**: Cutoff boundary is set to `< 16` characters (messages `>= 16` chars are preserved) with minimum 2 words (`min_words = 2`), accommodating authentic short messenger replies (*"Хорошо, договорились"*, *"Давай созвонимся"*).
  - **Spam & Phishing**: Cutoff boundary is `< 25` characters with minimum 4 words (`min_words = 4`).
- Output: `data/interim/cleaned_*.csv`.

#### Step 04: `04_merge_synthetic_pools.py`
- Combines base and targeted synthetic phishing into a single candidate pool.
- Performs sequential cross-class alphanumeric deduplication: Phishing -> Spam -> Ham.
- Implements N-gram frequency capping (first word, prefix bigram, internal trigram) with `min_words = 2` for Ham and `min_words = 4` for Spam/Phishing.
- Executes batched TF-IDF semantic deduplication (`char_wb` 3–5 ngrams; cosine threshold 0.65 for Phishing, 0.70 for Spam/Ham).
- Output: `data/interim/merged_synthetic_phishing.csv`, `data/interim/dedup_synthetic_spam.csv`, `data/interim/dedup_synthetic_ham.csv`.

#### Step 05: `05_procedural_urls.py`
- Prevents classifier overfitting on URL presence (*Feature Leakage*) by injecting realistic, shortener, and typosquatted URLs generated on the fly via `DynamicURLGenerator`:
  - Phishing URL injection rate: **65%** (typosquatted domains, token links, authentication bots);
  - Spam URL injection rate: **35%** (promotional landing pages, sale paths, Telegram channels);
  - Ham URL injection rate: **15%** (neutral news, search, video, and social portals).
- Output: `data/interim/final_synthetic_phishing.csv`, `data/interim/final_synthetic_spam.csv`, `data/interim/final_synthetic_ham.csv`.

#### Step 06: `06_finalise_datasets.py`
- Ingests real corpora from `data/real/` and clean synthetic corpora from `data/interim/`.
- Normalizes whitespace and enforces case-insensitive deduplication using `.str.casefold()`.
- **Anti-Leakage Purge**: Compares synthetic phishing against **all 300 real phishing messages**:
  - Removes exact duplicate matches;
  - Removes near-duplicates with TF-IDF cosine similarity strictly `> 0.75` (`ngram_range=(1, 2)`).
- Validates strict quotas (20k/20k/300 real, 14k/14k/7k synthetic). Raises `ValueError` on any shortfall.
- Output: `data/processed/*.csv`.

#### Step 07: `07_train_and_eval_all.py`
- Unified 5-fold cross-validation and generative Few-Shot evaluation.
- Stratified partitioning: 60 Test, 40 Val, 200 Train per fold from 300 real phishing messages.
- Real Val (**6,040** rows) and Real Test (**6,060** rows) contain strictly 0 synthetic samples.
- Direct sequential batched inference `direct_test_inference()` without padding leakage and without `trainer.predict()`.
- VRAM Supervisor and automated CUDA OOM recovery via `--oom-safe`.
- Few-Shot prompt parsing with `<think>` tag stripping and recursive batch halving under memory pressure.
- Output: `results/ALL_FOLDS_RAW.csv`, `results/FINAL_SUMMARY_METRICS.csv`, `results/predictions/`, `saved_best_models/`.

#### Step 08 / Re-validation: `src/test.py`
- Independent re-evaluation of optimal checkpoints from `saved_best_models/`.
- Universal architecture routing (`encoder` vs `llm_lora`) without hardcoded model names.
- Output: `results/REEVAL_BEST_FOLD_METRICS.csv`.

---

## 4. Pipeline Orchestration (`run_pipeline.py`)

The pipeline can be executed selectively or end-to-end via `run_pipeline.py`:

```bash
# Default execution: data processing only (Steps 03 -> 06)
python run_pipeline.py

# Include synthetic data generation (Steps 00 -> 06)
python run_pipeline.py --include-generation

# Full pipeline: generation, processing, and model training
python run_pipeline.py --include-generation --include-training

# Full pipeline including Few-Shot LLM evaluation
python run_pipeline.py --include-generation --include-training --include-few-shot

# Training and Few-Shot only on finalized datasets
python run_pipeline.py --include-training --include-few-shot
```

---

## 5. Model Training & 8 Experimental Configurations

### 5-Fold Stratification Scheme:
- **Test set** (strictly real, 6,060 rows): 3,000 Ham, 3,000 Spam, 60 Phishing. Across 5 folds, exactly 100% (300/300) of real phishing messages are evaluated without overlap.
- **Validation set** (strictly real, 6,040 rows): 3,000 Ham, 3,000 Spam, 40 Phishing.
- **Train set**: Partitioned across 8 experimental configurations to measure the impact of synthetic phishing:

| ID | Configuration Name | Ham (Class 0) | Spam (Class 1) | Phish (Class 2) | Total Train Rows | Composition & Purpose |
|---|---|---:|---:|---:|---:|---|
| 1 | `1_Baseline_Imbalanced` | 14,000 | 14,000 | 200 | 28,200 | Strictly real train data, severe class imbalance |
| 2 | `2_Baseline_ClassWeights` | 14,000 | 14,000 | 200 | 28,200 | Real train data with weighted CrossEntropyLoss `[1, 1, 70]` |
| 3 | `3_Baseline_Oversampling` | 14,000 | 14,000 | 7,000 | 35,000 | Real phishing sampled with replacement from the 200 train rows |
| 4 | `4_Pure_Synthetic` | 14,000 | 14,000 | 7,000 | 35,000 | 100% synthetic training data (0 real train samples) |
| 5 | `5_Hybrid_3.4x_10pct` | 14,000 | 14,000 | 880 | 28,880 | 200 real Phish + 680 synthetic Phish (10% synthetic quota) |
| 6 | `6_Hybrid_8.5x_25pct` | 14,000 | 14,000 | 1,900 | 29,900 | 200 real Phish + 1,700 synthetic Phish (25% synthetic quota) |
| 7 | `7_Hybrid_17x_50pct` | 14,000 | 14,000 | 3,600 | 31,600 | 200 real Phish + 3,400 synthetic Phish (50% synthetic quota) |
| 8 | `8_Hybrid_34x_100pct` | 14,000 | 14,000 | 7,000 | 35,000 | 200 real Phish + 6,800 synthetic Phish (100% synthetic quota) |

### Standalone Training CLI:

```bash
# Full training and Few-Shot across all active models
python src/07_train_and_eval_all.py

# Override model configuration
python src/07_train_and_eval_all.py --models-config configs/models_config.json

# Train specific models only
python src/07_train_and_eval_all.py --models RuBERT RuRoBERTa-large --skip-few-shot

# Run Few-Shot evaluation only (Zero-Shot, Few-Shot_Real, Few-Shot_Synth)
python src/07_train_and_eval_all.py --skip-training

# Verify fold splits and data schemas without GPU
python src/07_train_and_eval_all.py --dry-run
```

### Standalone Re-validation CLI:

```bash
# Re-validate all saved checkpoints from saved_best_models/
python src/test.py

# Re-validate specific model and configuration
python src/test.py --models Vikhr-8B --configs 1_Baseline_Imbalanced
```

---

## 6. Installation & Verification

### Setup:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models for synthetic generation
ollama pull qwen3:14b
ollama pull vikhr-llama3.1-8b
```

### Verification & Testing:

1. **Static Syntax & Bytecode Compilation**:
   ```bash
   python -m compileall -q run_pipeline.py src
   ```

2. **Automated Unit & Integration Test Suite**:
   The test suite systematically verifies all 6 blocks of edge cases in an isolated temporary environment:
   - **Block 1**: AST code audit verifying absence of duplicate functions (`clean_pool`, `split_synthetic_phishing`), absence of unsafe `trainer.predict()`, and exact adherence of `TrainingArguments` (`random` sampling, `group_by_length=False`, `bf16=True`);
   - **Block 2**: `models_config.json` inheritance, fallback mode, CLI swap test with prefix collision protection (`--models` vs `--models-config`), and `resolve_model_family` routing;
   - **Block 3**: Quota assertions in `load_verified_pools`, zero synthetic leakage in Val/Test, 100% test coverage of real phishing across 5 folds, and exact row counts across all 8 training configurations;
   - **Block 4**: Equivalence of batched inference (`batch_size=1` vs `16`) for Llama and BERT, label order preservation, and `WeightedTrainer` loss computation with `bfloat16` logits;
   - **Block 5**: Purity of shot sampling (`Zero-Shot`, `Few-Shot_Real`, `Few-Shot_Synth`), `<think>` tag stripping in `parse_prediction`, unparsed counter tracking, and recursive batch halving under CUDA OOM;
   - **Block 6**: Round-trip save and load parity between Step 07 and `src/test.py`, backward compatibility for LoRA adapters, `is_better_fold` selection logic, `update_summary` Student's $t$ confidence intervals, `--oom-safe` parameter halving, and graceful `Ctrl+C` signal handling (`SkipConfiguration` on single interrupt, `StopAllExecution` on double interrupt).

---

## 7. Safety and Ethical Guidelines

All synthetic phishing texts and scripts in this repository are developed strictly for academic research and defensive detection engineering. Do not use generated materials for phishing campaigns, unsolicited messaging, or social engineering. Never navigate to generated or real URLs. Research corpora and empirical outputs must be reviewed prior to external dissemination.
