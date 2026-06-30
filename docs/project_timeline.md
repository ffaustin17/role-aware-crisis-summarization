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
- `docs/reward_specification_v1.md`
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

## 15. Reward Scoring For Baseline Predictions

**Motivation:** The first baseline metrics showed how closely T5 matched the
synthetic GPT targets, but they did not explain whether predictions were useful
under the project's role-aware criteria.

**What was tried:** The baseline test predictions were scored with the first
reward function. The initial scoring pass used SentenceTransformer relevance,
a transparent factuality proxy, role coverage, and urgency. A report-ready CSV
was also created so predictions, targets, source tweets, and component rewards
could be inspected side by side.

**Result:** The first reward pass showed that the T5 baseline mostly produced
middle-range summaries. It also revealed examples where summaries were
semantically related to the disaster but missed the tweet's specific operational
content.

Primary artifacts:
- `data/rewards/t5_baseline_v1_reward_scores.jsonl`
- `reports/tables/t5_baseline_reward_summary.csv`
- `reports/tables/baseline_predictions_with_rewards_report.csv`

## 16. MiniCheck Factuality And Tweet-Dominant Relevance

**Motivation:** Qualitative inspection showed that generic role-aware summaries
could receive too much relevance credit when they matched role or disaster
metadata but missed the actual tweet. The factuality proxy was also useful but
too shallow for final reward experiments.

**What was tried:** The reward scorer was extended with optional MiniCheck
factuality. Relevance was changed from one combined source/context embedding to
a tweet-dominant blend:

```text
relevance = 0.70 tweet_relevance + 0.30 context_relevance
```

The new scorer was run on Kaggle for the 401 baseline test predictions, and a
new prediction/reward report was generated with tweet relevance, context
relevance, MiniCheck support probability, and the hard MiniCheck label for
traceability.

**Result:** The newer scorer became stricter. Mean reward decreased from about
0.601 to about 0.563, mainly because tweet-dominant relevance reduced credit for
generic summaries. MiniCheck factuality was useful as a grounding signal, though
its probabilities were tightly clustered near the decision boundary.

Primary artifacts:
- `data/rewards/t5_baseline_v1_reward_scores_tweet_relevance_minicheck.jsonl`
- `reports/tables/t5_baseline_reward_summary_tweet_relevance_minicheck.csv`
- `reports/tables/baseline_predictions_with_tweet_relevance_minicheck_rewards_report.csv`

## 17. Larger GPT Teacher Summary Dataset

**Motivation:** The first T5 baseline used 4,001 synthetic examples. To support
a stronger second baseline and later preference optimization, the project needed
more non-`Other` GPT teacher summaries.

**What was tried:** The resume-aware summary generator continued from the
existing append-only JSONL. It generated 1,200 additional successful summaries,
then another 800, using dataset-order mode with prompt version 2 and input text
version 3. A small number of malformed-JSON responses were preserved as
historical validation failures, then retried successfully.

**Result:** The append-only generation file now contains 6,001 unique
successful non-`Other` summaries. A definitive successful-only JSONL was created
so future modeling steps do not need to filter historical failed records.

Primary artifacts:
- `data/generated/summaries_prompt_v2_input_v3_first_2000.jsonl`
- `data/generated/gpt4o_initial_summaries_v0203.jsonl`

## 18. Summary Reward Analysis Reports

**Motivation:** As the project moves toward comparing GPT teacher summaries,
T5 baseline summaries, and future DPO summaries, reward outputs need a standard
analysis layer. Simple averages are not enough; the project needs distributions,
length metrics, and breakdowns by exact role label, disaster type, and
information type.

**What was tried:** A new JSONL-based analyzer was added for reward datasets.
It computes descriptive statistics for reward, relevance, factuality, role
coverage, urgency, summary word count, character length, and sentence count.
Roles are analyzed by exact raw label only, so multi-role labels such as
`Police/EMS` remain single categories.

**Result:** The project now has reusable reward-analysis CSVs for both existing
baseline reward runs. These reports showed that the current T5 baseline is
clustered in the middle reward range, that multi-role rows are harder, that
Firefighter rows currently score highest on average, and that MiniCheck
factuality is compressed but still useful as a continuous support signal.

Primary artifacts:
- `scripts/analyze_summary_reward_dataset.py`
- `reports/tables/summary_reward_analysis/`

## 19. Emerging Three-System Research Story

**Motivation:** The project has evolved beyond training one baseline model. The
more coherent research question is whether a prompted LLM can bootstrap
role-aware crisis summaries, whether a smaller reusable model can learn that
behavior, and whether reward-guided preference optimization can improve the
student model.

**What was clarified:** The project now frames GPT-4o/GPT-4o-mini summaries as
silver-standard teacher outputs rather than human gold labels. T5-small is the
first frozen reusable student model. The reward function is both an evaluator
and the basis for future preference-pair construction.

**Result:** The likely comparison path is:

- Prompted GPT teacher summaries
- Supervised T5 baseline summaries
- Future DPO-optimized T5 summaries

These systems can be compared with traditional summarization metrics and with
the same role-aware reward function.

## 20. T5 Baseline V2 And Full-Dataset Prediction

**Motivation:** The first T5 baseline was trained on 4,001 teacher summaries.
After the GPT teacher dataset grew to 6,001 successful non-`Other` examples,
the project needed a stronger supervised baseline trained on the larger
distribution.

**What was tried:** A fresh Kaggle GPU notebook rebuilt stratified 80/10/10
splits from `gpt4o_initial_summaries_v0203.jsonl`, trained `t5-small` for three
epochs, evaluated on the held-out test set, and generated predictions for all
6,001 examples across train, validation, and test splits.

**Result:** T5 baseline v2 improved over the first baseline:

- ROUGE-1 F1: 0.511072
- ROUGE-2 F1: 0.271986
- ROUGE-L F1: 0.452822
- BLEU: 19.952732
- BERTScore F1: 0.925358

The full prediction file gives the project a reusable student-model summary
for every non-`Other` teacher-summary row.

Primary artifacts:
- `data/modeling/t5_baseline_v2/all_predictions.jsonl`
- `data/modeling/t5_baseline_v2/train.jsonl`
- `data/modeling/t5_baseline_v2/validation.jsonl`
- `data/modeling/t5_baseline_v2/test.jsonl`
- `reports/tables/t5_small_baseline_v2_metrics.csv`
- `notebooks/t5-training-and-eval-notebook.ipynb`

## 21. Reward Scorer Tightening

**Motivation:** Before scoring the full 6,001-summary datasets, the reward
components needed one more calibration pass. Relevance and MiniCheck factuality
were stable, but role coverage and urgency were still too brittle because they
relied heavily on exact keyword overlap.

**What was tried:** The reward scorer kept the same composite equation and the
same tweet-dominant relevance and MiniCheck factuality behavior. Role coverage
vocabularies were expanded for EMS, Firefighter, and Police. Phrase matching
was made more punctuation-aware. Urgency was changed from raw term overlap to
concept coverage over categories such as casualty/injury, rescue/evacuation,
active hazard, and severity/threat.

**Result:** The scorer became more interpretable and better aligned with the
project's role-aware criteria without changing reward weights. New diagnostic
fields now report applicable and covered role categories, urgency categories,
and score reasons.

Primary artifacts:
- `scripts/score_rewards.py`
- `docs/reward_specification_v1.md`

## 22. Full GPT Teacher Versus T5 Reward Comparison

**Motivation:** Once both GPT teacher summaries and T5 v2 predictions existed
for the same 6,001 non-`Other` rows, the project could compare the two systems
with the same role-aware reward function.

**What was tried:** A new Kaggle scoring notebook installed MiniCheck, scored
all T5 v2 predictions, scored all GPT teacher summaries, generated detailed
reward-analysis CSVs, built presentation-ready joined CSV reports, and created
overall and role-level comparison tables.

**Result:** GPT teacher summaries scored higher overall, but the gap was not
large:

- T5 v2 mean reward: 0.593620
- GPT teacher mean reward: 0.617690
- Difference: +0.024071 for GPT

The main differences were role coverage and urgency:

- GPT role coverage advantage: +0.036897
- GPT urgency advantage: +0.075868
- Relevance was close: +0.005485 for GPT
- MiniCheck factuality was essentially tied and slightly favored T5

Role-level analysis showed that GPT was especially stronger for EMS,
Firefighter, and complex multi-role labels, while T5 was competitive on plain
Police and Firefighter/EMS examples.

Primary artifacts:
- `data/rewards/t5_baseline_v2_all_predictions_reward_scores_tweet_relevance_minicheck.jsonl`
- `data/rewards/gpt4o_initial_summaries_v0203_reward_scores_tweet_relevance_minicheck.jsonl`
- `reports/tables/t5_baseline_v2_all_predictions_with_rewards_report.csv`
- `reports/tables/gpt4o_initial_summaries_v0203_with_rewards_report.csv`
- `reports/tables/t5_v2_vs_gpt4o_reward_comparison.csv`
- `reports/tables/t5_v2_vs_gpt4o_reward_comparison_by_role.csv`
- `notebooks/scoring-notebook.ipynb`

## 23. Factuality Signal Reflection

**Motivation:** After scoring more than 12,000 summaries across GPT teacher
outputs and T5 v2 predictions, the project needed to interpret whether the
MiniCheck factuality component was doing useful work at its current reward
weight.

**What was observed:** MiniCheck factuality was implemented as a sentence-level
support probability. Each generated summary sentence is treated as a claim and
checked against the source context, then sentence probabilities are averaged.
Because most project summaries are one sentence, factuality is usually a single
support probability per summary.

The full comparison showed that factuality barely separated GPT from T5:

- T5 v2 factuality mean: 0.480305
- GPT teacher factuality mean: 0.478697
- Difference: -0.001608

The distributions were also tightly compressed near the decision boundary. This
suggests that factuality currently behaves more like a weak grounding prior or
constant offset than a strong ranking signal.

**Interpretation:** The component is still conceptually important because the
reward should discourage unsupported summaries. However, its current 0.25
weight may overstate how much discriminative information MiniCheck is actually
contributing in this dataset. The compression may be caused by short tweet
contexts, role-aware operational inferences that are reasonable but not
explicitly stated, and stylistic similarity between GPT and T5 summaries.

**Result:** The project will keep the current factuality backend and weight for
the present exploration to preserve a stable reward function. Future reward
calibration should investigate whether factuality is better used as a lower
weight component, a rescaled support score, or a thresholded penalty/guardrail
against unsupported claims.

## 24. DPO Preference Pair Construction

**Motivation:** Before training a DPO-optimized model, the project needed to
know whether the existing GPT teacher summaries and T5 v2 predictions could
produce a healthy preference dataset. DPO requires chosen/rejected pairs, not
single supervised targets, and near-ties can introduce noisy preference labels.

**What was tried:** A new preference-building script joined four existing
artifacts:

- T5 v2 all-prediction summaries
- T5 v2 reward scores
- GPT teacher summaries
- GPT teacher reward scores

For each shared `tweet_id`, the script compared GPT and T5 reward scores. The
higher-reward summary became `chosen`, the lower-reward summary became
`rejected`, and pairs below a reward-margin threshold of `0.03` were skipped as
near-ties.

**Result:** The first DPO preference dataset retained a meaningful number of
clean preference pairs:

- Candidate comparisons: 6,001
- Retained preference pairs: 3,891
- Skipped near-ties: 2,110
- Train pairs: 3,120
- Validation pairs: 394
- Test pairs: 377
- GPT teacher chosen: 2,399
- T5 baseline v2 chosen: 1,492

This suggests the dataset is large enough for a first DPO experiment while
remaining less noisy than a zero-margin "use everything" setup. The chosen
distribution is also useful because it does not simply select GPT for every
row; T5 wins a substantial minority of comparisons.

Primary artifacts:
- `scripts/build_dpo_preference_dataset.py`
- `data/preferences/dpo_preferences_t5_v2_vs_gpt4o_reward_v1.jsonl`
- `data/preferences/dpo_preferences_t5_v2_vs_gpt4o_reward_v1_train.jsonl`
- `data/preferences/dpo_preferences_t5_v2_vs_gpt4o_reward_v1_validation.jsonl`
- `data/preferences/dpo_preferences_t5_v2_vs_gpt4o_reward_v1_test.jsonl`
- `reports/tables/dpo_preferences_t5_v2_vs_gpt4o_summary.csv`
- `reports/tables/dpo_preferences_t5_v2_vs_gpt4o_by_role.csv`
- `reports/tables/dpo_preferences_t5_v2_vs_gpt4o_margin_distribution.csv`

## 25. DPO Training Script Scaffold

**Motivation:** After constructing a preference dataset, the next step was to
prepare a training path that could preference-optimize the supervised T5 v2
checkpoint. Since DPO is a training method rather than a separate model
architecture, the intended model is still T5: a supervised T5 v2 checkpoint
continued with preference optimization.

**What was tried:** A custom seq2seq DPO training script was added instead of
depending immediately on a higher-level DPO library. The script treats the
supervised T5 v2 checkpoint as the trainable policy model and also loads a
frozen reference copy of that same checkpoint. It computes chosen and rejected
sequence log probabilities, applies the DPO loss, evaluates on the validation
preference split, and saves a new T5-DPO checkpoint.

**Result:** The project now has a Kaggle-ready DPO training entrypoint. It is
designed to start from the externally stored T5 v2 checkpoint and use the
committed preference JSONLs. The first expected use is a small smoke run before
full training.

Primary artifact:
- `scripts/train_t5_dpo.py`

## 26. First T5-DPO Training And Evaluation

**Motivation:** The project needed to test whether preference optimization could
improve the supervised T5 v2 model on the role-aware reward criteria, especially
role coverage and urgency.

**What was tried:** The custom seq2seq DPO trainer was run on Kaggle from the
supervised T5 v2 checkpoint. The first experiment used the `0.03` margin
preference dataset, beta `0.1`, one epoch, 3,120 train pairs, and 394 validation
pairs. The resulting checkpoint generated summaries for the 601-example held-
out test set and all 6,001 non-`Other` inputs.

The DPO objective trained successfully:

- Validation preference accuracy: 0.720812
- Validation DPO loss: 0.677679
- Validation implicit reward margin: 0.032062

Traditional target-similarity metrics decreased relative to supervised T5 v2:

- ROUGE-1 F1: 0.463752
- ROUGE-2 F1: 0.229735
- ROUGE-L F1: 0.406215
- BLEU: 16.312138
- BERTScore F1: 0.916531

**Result:** DPO moved the model away from exact GPT teacher imitation but
increased the project's composite reward:

```text
GPT teacher mean reward:  0.617690
T5 v2 mean reward:        0.593620
T5-DPO mean reward:       0.653345
```

The gain also appeared on the held-out test split:

```text
GPT teacher test reward:  0.615422
T5 v2 test reward:        0.595655
T5-DPO test reward:       0.646819
```

The overall improvement was driven primarily by role coverage, which increased
from 0.471233 for supervised T5 to 0.751469 for T5-DPO. Relevance declined from
0.719472 to 0.705166, while factuality remained nearly unchanged.

Primary artifacts:
- `notebooks/t5-dpo.ipynb`
- `data/modeling/t5_dpo_v1_beta_0_1/all_predictions.jsonl`
- `data/modeling/t5_dpo_v1_beta_0_1/test_predictions.jsonl`
- `data/rewards/t5_dpo_v1_beta_0_1_reward_scores_tweet_relevance_minicheck.jsonl`
- `reports/tables/t5_small_dpo_v1_beta_0_1_metrics.csv`
- `reports/tables/gpt4o_vs_t5_v2_vs_t5_dpo_beta_0_1_reward_comparison.csv`

## 27. Qualitative DPO Analysis And Reward Shortcuts

**Motivation:** A higher automated reward does not necessarily mean that humans
would judge every summary as better. The large role-coverage increase and lower
relevance made qualitative inspection necessary.

**What was tried:** A balanced 42-example inspection set was created with six
diagnostic samples for each exact role label: largest DPO gain/loss versus T5,
largest gain/loss versus GPT, largest relevance drop, and a median-reward case.
The samples were read alongside their source tweets, all three candidate
summaries, component scores, and reward deltas. Template frequency was also
measured across all 6,001 outputs.

**Result:** DPO produced some genuine improvements. It sometimes preserved more
tweet-specific evidence, corrected irrelevant T5 role behavior, and expressed
appropriate responder needs. However, it also learned a strong reward shortcut:

- `scene security` appeared in 93.7% of DPO summaries
- it appeared at least twice in 38.5% of DPO summaries
- only 54.7% of DPO summaries were unique
- some exact summaries were repeated dozens or hundreds of times

The reward scorer correctly penalized several obvious role failures, but other
awkward or repetitive summaries received high rewards by mentioning configured
role terms, urgency concepts, and secondary-annotation abbreviations. Police,
the dominant role label, accounted for most of the overall reward improvement.

The first DPO run should therefore be interpreted as both a successful
preference-optimization proof of concept and evidence of reward
misspecification. It improved the measured objective, but part of that gain came
from template collapse and keyword-oriented behavior rather than uniformly
better human-readable summaries.

Primary artifacts:
- `reports/tables/t5_dpo_v1_beta_0_1_qualitative_samples_by_role.jsonl`
- `reports/tables/t5_dpo_v1_beta_0_1_with_rewards_report.csv`
- `reports/tables/gpt4o_vs_t5_v2_vs_t5_dpo_beta_0_1_reward_comparison_by_role.csv`

## 28. Reward V1 Freeze And Future Reward Design

**Motivation:** The first DPO run demonstrated that reward v1 was useful enough
to train a model, but also exploitable enough to encourage repetition,
role-keyword stuffing, and Police-dominant templates. The completed experiments
needed to remain reproducible while future reward revisions were explored
separately.

**What was tried:** The current specification was renamed and explicitly frozen
as `reward_specification_v1.md`. A separate analysis document was created to
record the reward system's journey, empirical results, qualitative examples,
component-level challenges, and possible future architecture.

The future design space includes:

- stronger tweet salience and key-information coverage
- factuality calibration or guardrail behavior
- role leakage and unsupported-urgency penalties
- repetition and degeneration penalties
- optional metadata-aware role, disaster, information-type, and secondary-
  annotation adherence
- observed-context and oracle-context modes for future ablation studies

**Result:** Reward v1 remains stable for reproducing the completed GPT, T5, and
DPO results, while future reward versions can evolve without rewriting the
meaning of historical scores.

Primary artifacts:
- `docs/reward_specification_v1.md`
- `docs/reward_system_evolution_and_future_design.md`

## 29. Final Three-System Human Review Report

**Motivation:** Existing reward JSONLs and analysis tables were useful for
programmatic analysis but too dense for direct human review. A final report
needed to foreground the actual tweet, the three competing summaries, and their
metric differences without IDs or unrelated metadata.

**What was tried:** A 6,001-row CSV was created in canonical tweet order. Each
row contains the runtime input, original tweet, GPT teacher summary, supervised
T5 summary, and T5-DPO summary. Reward, relevance, tweet relevance, context
relevance, factuality probability, role coverage, and urgency are consolidated
into newline-separated GPT/T5/T5-DPO breakdown cells.

**Result:** The project now has a report-ready artifact for qualitative reading
and professor-facing comparison. Factuality includes only the continuous
MiniCheck support probability, and identifiers or irrelevant metadata are
intentionally omitted.

Primary artifact:
- `reports/tables/final_gpt_t5_dpo_summary_reward_report.csv`

## Current State

The project currently has:

- a cleaned FReCS training schema
- 6,001 successful non-`Other` GPT teacher summaries
- a definitive successful-only teacher summary JSONL
- fixed T5 baseline v1 splits, predictions, metrics, and reward reports
- fixed T5 baseline v2 splits, predictions, and metrics from the 6,001-summary dataset
- trained `t5-small` baseline checkpoints stored externally/local to Kaggle
- a frozen reward v1 specification and scoring script with MiniCheck support
- a separate reward evolution and future-design analysis document
- full-dataset reward outputs for both GPT teacher summaries and T5 v2 predictions
- reusable reward dataset analysis reports and presentation CSVs
- an overall and role-level reward comparison between GPT teacher summaries and T5 v2
- a first DPO-style preference dataset built from GPT-versus-T5 reward comparisons
- a custom seq2seq DPO training script
- a trained beta `0.1` T5-DPO checkpoint stored locally and as a Kaggle output
- full T5-DPO predictions, metrics, reward scores, and comparison reports
- a qualitative analysis showing both genuine DPO gains and reward-shortcut behavior
- a finalized 6,001-row human-review CSV comparing GPT, T5, and T5-DPO
- Kaggle notebooks for T5 training/evaluation, reward scoring, and DPO training

## Near-Term Next Steps

1. Interpret the full GPT-versus-T5 reward comparison, especially multi-role
   behavior and the role coverage/urgency gaps.
2. Use the finalized human-review report for broader qualitative inspection and
   professor-facing discussion.
3. Add automated degeneration diagnostics such as exact-output duplication,
   repeated phrase frequency, and repeated n-grams.
4. Treat the beta `0.1` model as the stable first DPO checkpoint before deciding
   whether additional beta or reward-margin experiments are justified.
5. Explore future reward calibration, including redundancy, fluency, salience,
   and more discriminative factuality behavior.
