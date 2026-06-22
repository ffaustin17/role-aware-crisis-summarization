# Project Timeline And Evolution Log

This document narrates the project as a sequence of research and engineering
steps. It is meant to preserve why each step happened, what was tried, and what
the project gained from it.

## 1. Project Setup And Initial Direction

**Motivation:** Establish a dedicated Python project for role-aware crisis tweet
summarization using the FReCS dataset. The guiding research idea was to move
beyond first-responder classification and toward summaries that are useful for
EMS, Police, Firefighters, or multi-role responder groups.

**What was done:** The repository was initialized with Python project metadata,
dependencies, a prompt folder, a scripts folder, and a basic README describing
the research goals.

**Result:** The project had a stable workspace for dataset inspection, schema
construction, prompt engineering, summary generation, baseline model training,
and later reward modeling.

## 2. Raw FReCS Dataset Inspection

**Motivation:** Before building a summarization pipeline, the original FReCS CSV
needed to be understood. The raw dataset had columns such as `Tweet ID`,
`Tweet_Text`, `Information_Source`, `Information_Type`, `Informativeness`,
`First_Responder`, `Disaster_Type`, `Secondary_Annotation`, and several unnamed
extra columns.

**What was tried:** A dataset inspection script was created to examine column
names, missing values, duplicate tweets, responder-role distributions, disaster
types, and secondary annotations. The inspection output was summarized into an
inventory spreadsheet.

**Result:** The project confirmed that the raw tweet ID was not useful as the
canonical project identifier, that duplicate tweet text needed handling, that
`Other` responder rows dominated the dataset, and that multi-role labels such as
`Police/EMS` should remain tweet-level rows rather than being expanded.

Primary artifact:
- `reports/tables/frecs_dataset_inventory.xlsx`

## 3. Clean Tweet-Level Training Schema

**Motivation:** The raw FReCS CSV was not ready for generation or model training.
The project needed a clean, deterministic schema where each row represented one
unique tweet with normalized fields and parsed arrays.

**What was tried:** A reusable schema-building library was created, along with a
small script for writing schema artifacts. The pipeline normalized fields,
removed duplicate tweets, parsed responder labels into `roles_array`, parsed
secondary annotations into `secondary_annotations_array`, and generated a
canonical sequential `tweet_id`.

**Result:** The project produced cleaned CSV and JSONL schema files. The schema
remained tweet-level and preserved multi-role rows instead of expanding them.

Primary artifacts:
- `data/processed/frecs_training_schema_v2.csv`
- `data/processed/frecs_training_schema_v2.jsonl`

## 4. Responder-Relevant Ordering

**Motivation:** The dataset contained many `Other` responder rows. Since early
generation batches needed to be responder-relevant, non-`Other` rows should
appear first while still preserving the `Other` rows for future use.

**What was tried:** The schema library was updated to move exact `Other` role
rows to the bottom, shuffle non-`Other` rows reproducibly with a fixed seed, and
regenerate sequential `tweet_id` values after final ordering. A `source_row_id`
field was preserved to trace cleaned rows back to the raw CSV row position.

**Result:** Early generation could proceed through varied responder-relevant
rows without dropping `Other` data. The canonical IDs stayed deterministic and
clean after reordering.

## 5. OpenAI Environment And Smoke Testing

**Motivation:** Summary generation depended on the OpenAI API, so the project
needed a minimal way to confirm the environment and restricted API key worked
before spending time or credits on full generation.

**What was tried:** Environment dependencies were added, `.env.example` was
documented, and a small OpenAI smoke-test script was created.

**Result:** The project gained a quick API connectivity check that could fail
early with a clear message instead of failing during a generation batch.

Primary artifact:
- `scripts/smoke_test_openai.py`

## 6. First Summary Generation Prototype

**Motivation:** The project needed synthetic supervised targets before training
a baseline summarization model. The first goal was to generate role-aware base
summaries from cleaned FReCS rows.

**What was tried:** An initial `generate_base_summaries.py` script and a v1
prompt were added. This prototype captured the original generation intent and
served as a historical reference.

**Result:** The project had a proof-of-concept generation path, but it still
needed more flexible input construction and prompt versioning.

Primary artifacts:
- `scripts/generate_base_summaries.py`
- `prompts/base_summaries_generation_v1.txt`

## 7. Runtime Input Builders And Prompt Versions

**Motivation:** The older training schema included a precomputed `input_text`
field, but experiments needed different input formats at runtime. Hardcoding one
input string into the schema would make prompt and model experiments harder.

**What was tried:** The schema was updated to remove hardcoded `input_text`.
A new summary generation utility was added with runtime input builders:

- `v1`: tweet text, disaster type, responder roles
- `v2`: v1 plus secondary annotations
- `v3`: v2 plus information type

Prompt versioning was added, and a v2 prompt was moved into an external prompt
file with FReCS secondary-annotation context.

**Result:** Summary generation became experiment-friendly. The training schema
kept normalized source fields, while the generation script owned construction of
runtime input text.

Primary artifacts:
- `scripts/generate_summaries.py`
- `prompts/summary_generation_prompt_v2.txt`

## 8. Scale-Ready Summary Generation And Resume

**Motivation:** Generating thousands of summaries takes hours and costs API
credits. The pipeline needed to survive interruptions without restarting from
scratch.

**What was tried:** The summary generator was updated with explicit row
selection modes and append-only resume behavior. In dataset-order mode,
`--limit N` means "complete the first N eligible rows." Prior records with
`generation_status == "success"` are skipped, while failed or validation-failed
rows are retried.

**Result:** Large generation became recoverable. The project generated a large
summary file incrementally, including successful retries after transient
validation failures.

Primary artifact:
- `data/generated/summaries_prompt_v2_input_v3_first_2000.jsonl`

## 9. Synthetic Summary Dataset Growth

**Motivation:** A first baseline could be trained with 2,000 examples, but a
larger dataset would improve the credibility and stability of the baseline.

**What was tried:** Generation proceeded in batches using the resume mechanism:
first 2,000 successful summaries, then additional batches of 600, 600, and 800.
The output file remained append-only, with a few historical validation-failed
records preserved for traceability.

**Result:** The project reached 4,001 unique successful generated summaries.
These became the synthetic supervised data foundation for the first T5 baseline.

## 10. T5-Small Baseline Data Preparation

**Motivation:** The append-only generation file contained both final successes
and historical failed attempts, so it needed to be converted into a clean
supervised modeling dataset.

**What was tried:** A preparation script was added to keep only successful
records, deduplicate by `tweet_id`, map `input_text` to `target_text =
final_base_summary_text`, and create fixed stratified train/validation/test
splits by exact role label.

**Result:** The project produced 4,001 clean supervised examples split into:

- 3,200 training examples
- 400 validation examples
- 401 held-out test examples

Primary artifacts:
- `data/modeling/t5_baseline_v1/train.jsonl`
- `data/modeling/t5_baseline_v1/validation.jsonl`
- `data/modeling/t5_baseline_v1/test.jsonl`
- `data/modeling/t5_baseline_v1/split_metadata.json`

## 11. T5-Small Baseline Training Pipeline

**Motivation:** The project needed a first model baseline that could be trained
and evaluated reproducibly. `t5-small` was chosen because it is lightweight,
well-known, and feasible on Kaggle/Colab GPU resources.

**What was tried:** Training and evaluation scripts were added with Hugging Face
Transformers. A local smoke test verified that tokenization, training, checkpoint
saving, and evaluation wiring worked. Full training was then run on Kaggle GPU
for three epochs.

**Result:** A baseline `t5-small` checkpoint was trained successfully on the
prepared synthetic dataset. The checkpoint was treated as a local/external model
artifact rather than committed to Git.

Primary artifacts:
- `scripts/train_t5_baseline.py`
- `scripts/evaluate_t5_baseline.py`
- Kaggle checkpoint: `t5_small_baseline_v1`

## 12. Baseline Evaluation

**Motivation:** The baseline needed objective metrics for the professor-facing
deliverable: ROUGE, BLEU, and BERTScore.

**What was tried:** The trained checkpoint generated summaries for the 401
held-out test examples. The predictions were compared against the synthetic
target summaries using ROUGE, SacreBLEU, and BERTScore.

**Result:** The first baseline achieved:

- ROUGE-1 F1: 0.492000
- ROUGE-2 F1: 0.249925
- ROUGE-L F1: 0.426849
- BLEU: 17.859800
- BERTScore F1: 0.920640

These results show that `t5-small` learned to approximate the generated
role-aware targets reasonably well. The major caveat is that the targets are
synthetic summaries rather than human gold summaries.

Primary artifacts:
- `reports/tables/t5_small_baseline_metrics.csv`
- `reports/tables/t5_small_baseline_metrics.md`
- `data/modeling/t5_baseline_v1/test_predictions.jsonl`

## 13. Role-Aware Reward Specification

**Motivation:** ROUGE, BLEU, and BERTScore compare model outputs against target
summaries, but they do not directly measure whether a summary is useful for EMS,
Firefighters, or Police. The project needed a task-specific reward function.

**What was tried:** The generated summaries, source tweets, roles, and T5
predictions were scanned to identify recurring role-specific terminology. The
reward spec was grounded in actual project language rather than only abstract
role definitions.

The first reward function combines:

```text
0.35 relevance + 0.25 factuality + 0.20 role_coverage + 0.20 urgency
```

**Result:** A transparent reward specification and scoring script were added.
The factuality score is currently a lightweight source-grounding proxy, with a
future path to MiniCheck/MiniFactScore-style factuality scoring.

Primary artifacts:
- `docs/reward_specification.md`
- `scripts/score_rewards.py`

## 14. Project Structure Cleanup

**Motivation:** As the project evolved, the repository contained unused package
scaffolding and an outdated README that no longer matched the real workflow.

**What was tried:** The empty `src/crisis_summarization/` package skeleton and
unused root `main.py` were removed. The project was made explicitly
script-first, `pyproject.toml` was updated accordingly, local transfer scratch
space was ignored, and the README was rewritten around the actual project
structure.

**Result:** The repository now has a clearer shape:

- `scripts/` contains runnable pipeline steps.
- `data/` contains raw, processed, generated, modeling, and future reward data.
- `prompts/` contains generation prompts.
- `docs/` contains research specifications and project notes.
- `reports/` contains metrics and inventory tables.
- `models/` remains local-only for checkpoints.

## Current State

The project currently has:

- a cleaned FReCS training schema
- 4,001 successful generated role-aware synthetic summaries
- fixed T5 baseline train/validation/test splits
- a trained `t5-small` baseline checkpoint stored externally/local to Kaggle
- committed baseline prediction and metrics artifacts
- a role-aware reward specification and first reward scoring script

## Near-Term Next Steps

1. Run the reward scorer over the full baseline test predictions.
2. Inspect reward-score distributions by role and disaster type.
3. Improve factuality scoring with a learned grounding model if feasible.
4. Compare T5 predictions, OpenAI targets, and reward scores qualitatively.
5. Decide whether to generate more summaries or move into preference/reward
   optimization experiments.
