# Role-Aware Crisis Summarization

This project builds a role-aware crisis tweet summarization pipeline using the
FReCS dataset. It processes raw FReCS records into a tweet-level training schema,
generates synthetic role-aware summaries, trains a baseline `t5-small` model,
evaluates the baseline with summarization metrics, and scores outputs with a
transparent role-aware reward function.

The project is script-first: the runnable workflow lives in `scripts/`, while
data artifacts, prompts, reports, and documentation live in top-level folders.

## Project Structure

```text
role-aware-crisis-summarization/
  data/
    raw/          Original FReCS CSV.
    processed/    Cleaned tweet-level schema artifacts.
    generated/    OpenAI-generated synthetic summary JSONL batches.
    modeling/     T5 baseline train/validation/test splits and predictions.
    rewards/      Reward-score outputs from the role-aware scorer.
    preferences/  DPO-style chosen/rejected preference-pair datasets.

  docs/           Research/specification documents.
  prompts/        External prompts used by summary generation scripts.
  reports/
    tables/       Dataset inventories, metrics tables, reward summaries.
    figures/      Future report figures.

  scripts/        Runnable project pipeline scripts.
  notebooks/      Optional exploratory/Kaggle notebooks.
  models/         Local model checkpoints; ignored by Git.
```

`temp_encode/` is local transfer scratch space and is ignored by Git.

## Main Workflow

### 1. Inspect The Original Dataset

```powershell
.venv\Scripts\python.exe scripts\inspect_original_frecs_dataset.py
```

This inspects the raw FReCS CSV and writes dataset inventory tables under
`reports/tables/`.

### 2. Build The Clean Training Schema

```powershell
.venv\Scripts\python.exe scripts\generate_training_dataset.py
```

The schema pipeline reads `data/raw/frecs.csv` and writes the cleaned tweet-level
schema to `data/processed/`. The current schema keeps one row per unique tweet,
preserves multi-role labels, moves `Other` responder-role rows to the bottom,
and assigns deterministic sequential `tweet_id` values after final ordering.

### 3. Generate Synthetic Summaries

```powershell
.venv\Scripts\python.exe scripts\generate_summaries.py `
  --prompt-version v2 `
  --input-text-version v3 `
  --selection-mode dataset_order `
  --limit 4001 `
  --output data/generated/summaries_prompt_v2_input_v3_first_2000.jsonl
```

The summary generator is append-only and resume-aware. Re-running the same
command skips rows with prior `generation_status == "success"` and retries
failed or validation-failed rows.

### 4. Prepare T5 Baseline Splits

```powershell
.venv\Scripts\python.exe scripts\prepare_t5_baseline_data.py
```

This filters successful generated summaries, deduplicates by `tweet_id`, and
writes fixed stratified `80/10/10` train/validation/test splits under
`data/modeling/t5_baseline_v1/`.

### 5. Train The T5-Small Baseline

Full training should run on a GPU environment such as Kaggle or Colab.

```bash
python scripts/train_t5_baseline.py \
  --output-dir /kaggle/working/t5_small_baseline_v1 \
  --num-train-epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 1
```

Model checkpoints are local or external artifacts and are not committed to Git.

### 6. Evaluate The Baseline

```bash
python scripts/evaluate_t5_baseline.py \
  --model-dir /kaggle/working/t5_small_baseline_v1 \
  --metrics-csv /kaggle/working/t5_small_baseline_metrics.csv \
  --metrics-md /kaggle/working/t5_small_baseline_metrics.md \
  --predictions /kaggle/working/test_predictions.jsonl
```

The committed baseline evaluation artifacts are:

- `reports/tables/t5_small_baseline_metrics.csv`
- `reports/tables/t5_small_baseline_metrics.md`
- `data/modeling/t5_baseline_v1/test_predictions.jsonl`

### 7. Score Role-Aware Rewards

```powershell
.venv\Scripts\python.exe scripts\score_rewards.py
```

The reward scorer evaluates predictions with:

- relevance
- factuality proxy
- role coverage
- urgency

The composite reward is:

```text
0.35 relevance + 0.25 factuality + 0.20 role_coverage + 0.20 urgency
```

The reward criteria are defined in `docs/reward_specification.md`.

### 8. Build DPO Preference Pairs

```powershell
.venv\Scripts\python.exe scripts\build_dpo_preference_dataset.py --write-split-files
```

This joins GPT teacher summaries, T5 baseline predictions, and their reward
scores. It writes DPO-style `prompt`, `chosen`, and `rejected` JSONL records
under `data/preferences/`, skipping near-ties with a default reward margin of
`0.03`.

## Key Scripts

```text
scripts/inspect_original_frecs_dataset.py   Inspect raw FReCS columns and values.
scripts/generate_training_schema_lib.py     Reusable schema-building library.
scripts/generate_training_dataset.py        Write cleaned schema CSV/JSONL.
scripts/generate_summaries.py               Generate role-aware synthetic summaries.
scripts/prepare_t5_baseline_data.py         Build T5 train/validation/test splits.
scripts/train_t5_baseline.py                Fine-tune t5-small.
scripts/evaluate_t5_baseline.py             Compute ROUGE, BLEU, and BERTScore.
scripts/score_rewards.py                    Compute role-aware reward scores.
scripts/analyze_summary_reward_dataset.py   Analyze reward JSONL distributions.
scripts/build_prediction_reward_report.py   Join summaries with reward scores.
scripts/build_dpo_preference_dataset.py     Build DPO chosen/rejected pairs.
scripts/train_t5_dpo.py                     Preference-optimize T5 with DPO.
scripts/smoke_test_openai.py                Check OpenAI API connectivity.
```

`scripts/generate_base_summaries.py` is preserved as an earlier prototype for
history and comparison.

## Data And Artifact Notes

- The project currently commits the raw FReCS CSV, processed schema files,
  generated summary datasets, modeling splits, test predictions, and metrics
  tables.
- `models/` is ignored because checkpoint files are large and should be kept
  locally, downloaded from Kaggle, or stored with a dedicated model artifact
  system.
- `.env` is ignored. Use `.env.example` as the template for local secrets.

## Current Baseline Result

The first `t5-small` baseline was trained on 3,200 examples, validated on 400,
and evaluated on 401 held-out examples.

```text
ROUGE-1 F1:   0.492000
ROUGE-2 F1:   0.249925
ROUGE-L F1:   0.426849
BLEU:        17.859800
BERTScore F1: 0.920640
```
