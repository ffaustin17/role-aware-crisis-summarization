# Project Metrics Overview

## Scope

This report consolidates the principal quantitative and qualitative results from the role-aware crisis tweet summarization project through the first T5-DPO experiment. It covers:

- inspection and cleaning of the FReCS dataset;
- generation of GPT-4o-mini teacher summaries using prompt v2 and runtime input builder v3;
- supervised fine-tuning of T5-small;
- reward scoring with tweet-dominant relevance, MiniCheck factuality, role coverage, and urgency;
- preference-pair construction; and
- DPO optimization of the T5-small baseline at `beta = 0.1`.

The report intentionally separates held-out reference-similarity metrics from reward scores. ROUGE, BLEU, and BERTScore measure similarity to the synthetic teacher target on a held-out test split. Reward scores measure alignment with the project's automated rubric and were calculated for all 6,001 examples. These two evaluation views answer different questions.

## Executive Summary

The project produced a complete end-to-end experimental pipeline and three directly comparable summary sources:

1. GPT-4o-mini teacher summaries;
2. a supervised T5-small baseline trained to imitate those summaries; and
3. a T5-small model optimized with DPO using reward-derived GPT-versus-T5 preferences.

The central result is mixed but informative:

- The supervised T5-small baseline became a credible compact imitator. On 601 held-out examples, it reached ROUGE-1 `0.5111`, ROUGE-L `0.4528`, BLEU `19.95`, and BERTScore F1 `0.9254`.
- DPO successfully learned the preference signal: validation preference accuracy reached `72.08%`, and mean automated reward rose from `0.5936` for T5 v2 to `0.6533`.
- The DPO model did not improve uniformly. Its held-out ROUGE, BLEU, and BERTScore all declined, and its reward gain was overwhelmingly driven by role coverage, especially on the majority `Police` label.
- Qualitative inspection shows both real improvements and reward exploitation. DPO often learned useful role-specific language, but it also overused phrases such as `scene security`, repeated terms, and sometimes substituted a majority-role pattern for the assigned role.
- MiniCheck factuality scores were almost constant across models and records. With a nominal 25% reward weight, factuality currently contributes little to ranking or model separation.

The defensible conclusion is not that DPO is globally better than GPT or supervised T5. The experiment shows that the DPO implementation works and that the model can optimize the current reward, while also exposing where the reward specification and data distribution need refinement.

## 1. Dataset And Generation Scale

### Raw FReCS inventory

| Measure | Value |
|---|---:|
| Raw rows | 27,933 |
| Raw columns | 13 |
| Unique tweet texts | 25,306 |
| Duplicate tweet-text rows | 2,627 |
| Unique original Tweet IDs | 154 |
| Fully null legacy columns | 5 |

The original Tweet ID is highly duplicated and unsuitable as a canonical record identifier. The project therefore deduplicated on tweet text and assigned its own sequential `tweet_id` after final row ordering.

The raw dataset is also heavily dominated by `Other`: 20,570 rows, or 73.64% of the original 27,933 rows. After tweet-text deduplication, the cleaned schema contains 25,306 unique rows:

| Cleaned-schema group | Rows | Share |
|---|---:|---:|
| Non-Other responder labels | 6,608 | 26.11% |
| Other | 18,698 | 73.89% |
| Total | 25,306 | 100.00% |

The cleaned schema remains tweet-level. Multi-role labels such as `Police/EMS` and `Police/Firefighter/EMS` remain single records.

### Teacher-summary dataset

The canonical initial teacher dataset contains 6,001 successful GPT-4o-mini summaries. This covers 90.81% of the 6,608 cleaned non-Other tweets. No `Other` rows are included in this initial version.

| Exact role label | Examples | Share |
|---|---:|---:|
| Police | 3,280 | 54.66% |
| Firefighter | 1,023 | 17.05% |
| EMS | 593 | 9.88% |
| Police/EMS | 401 | 6.68% |
| Police/Firefighter | 340 | 5.67% |
| Police/Firefighter/EMS | 211 | 3.52% |
| Firefighter/EMS | 153 | 2.55% |
| **Total** | **6,001** | **100.00%** |

This distribution matters later: more than half of the generation, reward, preference, and DPO examples carry the exact `Police` label.

## 2. Supervised T5-Small Baselines

Baseline v2 used all 6,001 teacher summaries and a reproducible exact-role-stratified split with seed 42:

| Split | Examples | Share |
|---|---:|---:|
| Train | 4,800 | 79.99% |
| Validation | 600 | 10.00% |
| Test | 601 | 10.01% |

The supervised task was `input_text -> final_base_summary_text`. The runtime input included tweet text, responder role label, disaster type, secondary annotation, and information type.

### Held-out generation metrics

| Metric | Baseline v1 (401 test) | Baseline v2 (601 test) | T5-DPO beta 0.1 (601 test) | DPO minus v2 |
|---|---:|---:|---:|---:|
| ROUGE-1 F1 | 0.4920 | 0.5111 | 0.4638 | -0.0473 |
| ROUGE-2 F1 | 0.2499 | 0.2720 | 0.2297 | -0.0423 |
| ROUGE-L F1 | 0.4268 | 0.4528 | 0.4062 | -0.0466 |
| BLEU | 17.8598 | 19.9527 | 16.3121 | -3.6406 |
| BERTScore F1 | 0.9206 | 0.9254 | 0.9165 | -0.0088 |

Baseline v2 improved over v1 on every listed metric, including +0.0191 ROUGE-1, +0.0260 ROUGE-L, and +2.09 BLEU. This is consistent with the larger training set, but v1 and v2 used different held-out sets, so their delta is historical evidence rather than a perfectly controlled paired comparison.

The DPO comparison is stricter because DPO and baseline v2 were evaluated on the same 601-example test set. DPO moved away from the synthetic teacher references on all metrics. Since the target summaries were generated by GPT-4o-mini, this means reduced teacher imitation; it does not by itself prove reduced human utility.

## 3. Reward System And Three-Model Results

The current composite reward is:

```text
reward = 0.35 relevance
       + 0.25 factuality
       + 0.20 role_coverage
       + 0.20 urgency
```

Relevance is itself tweet-dominant:

```text
relevance = 0.70 tweet_relevance + 0.30 context_relevance
```

Factuality uses MiniCheck's continuous support probability. Role coverage checks whether summaries cover source-supported responder concepts. Urgency checks source-supported urgency concepts such as casualties, rescue or evacuation, active hazards, and severity or threat.

### Overall mean scores across 6,001 examples

| Metric | GPT-4o-mini | T5 baseline v2 | T5-DPO beta 0.1 |
|---|---:|---:|---:|
| Composite reward | 0.6177 | 0.5936 | **0.6533** |
| Relevance | **0.7250** | 0.7195 | 0.7052 |
| Factuality | 0.4787 | **0.4803** | 0.4797 |
| Role coverage | 0.5081 | 0.4712 | **0.7515** |
| Urgency | **0.7133** | 0.6374 | 0.6815 |
| Summary words | 17.39 | 16.72 | 17.35 |

GPT and supervised T5 are close overall: the teacher's mean reward advantage is only `0.0241`. GPT's clearest advantages are urgency (+0.0759) and role coverage (+0.0369), while relevance differs by only +0.0055 and factuality slightly favors T5 by +0.0016.

DPO raises reward by `0.0597` over T5 v2 and `0.0357` over GPT. However, its relevance is lower than both models, and its urgency remains below GPT. The gain is therefore not broad-based.

### Reward-distribution shape

| Model | Mean | Median | Std. dev. | P10 | P90 |
|---|---:|---:|---:|---:|---:|
| GPT-4o-mini | 0.6177 | 0.6011 | 0.1247 | 0.4546 | **0.7862** |
| T5 baseline v2 | 0.5936 | 0.5851 | 0.1254 | 0.4321 | 0.7696 |
| T5-DPO beta 0.1 | **0.6533** | **0.6697** | **0.0994** | **0.5032** | 0.7744 |

DPO primarily lifts the lower and middle portions of the automated reward distribution. Its P10 is substantially higher, but its P90 remains below GPT. This is consistent with the model learning reliable rubric-triggering phrases rather than creating a stronger upper tail of highly specific summaries.

### Where the DPO reward gain came from

| Component | DPO minus T5 raw delta | Reward-weighted contribution |
|---|---:|---:|
| Relevance | -0.0143 | -0.0050 |
| Factuality | -0.0006 | -0.0001 |
| Role coverage | +0.2802 | **+0.0560** |
| Urgency | +0.0441 | +0.0088 |
| **Composite total** |  | **+0.0597** |

Role coverage accounts for almost the entire positive movement. The urgency gain adds support, while lower relevance offsets part of the improvement.

Compared with GPT, DPO's +0.0357 reward advantage is also entirely attributable to role coverage (+0.0487 weighted). Lower relevance (-0.0069 weighted) and lower urgency (-0.0063 weighted) offset that gain.

## 4. Role-Level Results And Imbalance

| Exact role | GPT reward | T5 v2 reward | DPO reward | DPO minus T5 | DPO minus GPT |
|---|---:|---:|---:|---:|---:|
| EMS | **0.7294** | 0.6397 | 0.6684 | +0.0287 | -0.0611 |
| Firefighter | **0.7267** | 0.6696 | 0.6684 | -0.0013 | -0.0583 |
| Firefighter/EMS | 0.6102 | **0.6350** | 0.5438 | -0.0912 | -0.0663 |
| Police | 0.5749 | 0.5868 | **0.6774** | +0.0906 | +0.1024 |
| Police/EMS | 0.5757 | 0.5317 | **0.5849** | +0.0532 | +0.0092 |
| Police/Firefighter | **0.5809** | 0.4748 | 0.5503 | +0.0755 | -0.0306 |
| Police/Firefighter/EMS | **0.5840** | 0.4814 | 0.5400 | +0.0586 | -0.0441 |

DPO improves over T5 on five of seven exact role labels, but it improves over GPT on only two: `Police` and `Police/EMS`. It regresses sharply on `Firefighter/EMS` and is essentially tied with T5 on `Firefighter`.

The majority `Police` label contributes approximately `0.0495` of the total `0.0597` DPO-over-T5 reward gain, or about 83% of the observed overall improvement. Police role coverage rises from `0.4600` for T5 and `0.3147` for GPT to `0.9028` for DPO. This concentration is a major reason the aggregate DPO reward should not be interpreted without role breakdowns.

Multi-role behavior remains difficult. DPO role coverage is only `0.4592` for `Firefighter/EMS` and `0.3246` for `Police/Firefighter/EMS`, far below its `0.9028` on Police-only records.

## 5. Factuality Diagnostic

| Model | Factuality mean | Std. dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| GPT-4o-mini | 0.4787 | 0.0093 | 0.4632 | 0.5240 |
| T5 baseline v2 | 0.4803 | 0.0096 | 0.4647 | 0.5273 |
| T5-DPO beta 0.1 | 0.4797 | 0.0066 | 0.4649 | 0.5222 |

Across 18,003 scored summaries, MiniCheck factuality is tightly clustered around `0.48`. The largest difference between model means is only `0.0016`; after applying the 25% reward weight, that difference contributes less than `0.0004` to the composite comparison.

This does not establish that all three models are equally factual. It establishes that the current MiniCheck configuration has little discriminative power for these short, action-oriented summaries. At present, factuality behaves mostly like a near-constant reward offset. The continuous probability is still more informative than MiniCheck's hard 0/1 label, but future work should calibrate this component against human judgments or reconsider how claims and evidence are presented to the checker.

## 6. Preference Dataset And DPO Training

Preference pairs compared the GPT teacher summary with the T5 baseline v2 summary for the same input. Pairs with absolute reward margin below `0.03` were excluded.

| Preference statistic | Value |
|---|---:|
| Candidate inputs | 6,001 |
| Retained preference pairs | 3,891 (64.84%) |
| Skipped near ties | 2,110 (35.16%) |
| Mean retained reward margin | 0.1697 |
| Median retained reward margin | 0.1834 |
| GPT chosen | 2,399 (61.66% of pairs) |
| T5 chosen | 1,492 (38.34% of pairs) |
| DPO train pairs | 3,120 |
| DPO validation pairs | 394 |
| DPO test pairs | 377 |

The margin filter removed ambiguous comparisons but also reduced the usable sample. The result still provides a meaningful 3,120-pair training set. The preference distribution confirms that GPT was not universally preferred: supervised T5 won more than one third of clear comparisons.

The first DPO run used one epoch, learning rate `1e-5`, and `beta = 0.1`. Final validation diagnostics were:

| DPO diagnostic | Value |
|---|---:|
| Validation preference accuracy | 0.7208 |
| Validation loss | 0.6777 |
| Policy preference margin | 0.2269 |
| Reference preference margin | -0.0937 |
| Implicit reward margin | 0.0321 |

These values show that the policy learned to rank the reward-derived chosen summary above the rejected summary more reliably than the reference model. That is a successful optimization result, even though downstream generation quality remains mixed.

## 7. Qualitative Examples

### Example A: A genuine DPO improvement

**Tweet:** `Support UNICEF's emergency relief efforts for kids in the #Philippines ... #Haiyan`

- GPT: `EMS should evaluate medical needs and support for vulnerable children in typhoon-affected regions as relief efforts are mobilized.`
- T5: `EMS should assess crowd control needs and coordinate with UNICEF in response to the typhoon.`
- DPO: `Assess medical needs and coordinate with UNICEF for emergency relief efforts for kids in the Philippines.`

| Model | Reward | Relevance | Factuality | Role coverage | Urgency |
|---|---:|---:|---:|---:|---:|
| GPT | 0.5935 | 0.7869 | 0.4723 | 1.0 | 0.0 |
| T5 | 0.3853 | 0.7679 | 0.4663 | 0.0 | 0.0 |
| DPO | **0.8148** | **0.8481** | 0.4717 | 1.0 | 1.0 |

Here DPO improves specificity and corrects the T5 model's inappropriate `crowd control` framing. The higher reward corresponds to a plausible human improvement.

### Example B: Reward exploitation through role vocabulary

**Tweet:** `Colorado floods threaten more homes as another feared dead`

- GPT: `Police should assess potential evacuation needs and monitor for public safety threats related to the flooding situation in Colorado.`
- T5: `Police should assess potential criminal activity and coordinate response efforts following the reported flooding threat.`
- DPO: `Police should assess scene security and coordinate with DCC for scene security and ensure scene security following the reported flooding.`

| Model | Reward | Relevance | Factuality | Role coverage | Urgency |
|---|---:|---:|---:|---:|---:|
| GPT | 0.3865 | **0.7627** | 0.4781 | 0.0 | 0.0 |
| T5 | 0.3771 | 0.7407 | 0.4713 | 0.0 | 0.0 |
| DPO | **0.5741** | 0.7298 | 0.4748 | **1.0** | 0.0 |

DPO receives the highest reward despite lower relevance and obvious repetition. Its role-coverage jump from 0 to 1 contributes a full `0.20` to the composite reward. The output underemphasizes the reported possible death and evacuation risk. This is clear evidence that the current scorer can be optimized lexically without a corresponding improvement in summary quality.

### Example C: A role-substitution regression

**Tweet:** `Quebec train crash death toll now at 20`

The assigned role is EMS.

- GPT: `Prioritize medical response for potential casualties from the reported train derailment, focusing on affected individuals.`
- T5: `EMS should assess casualties from the train crash and coordinate with MMU for medical response.`
- DPO: `Police should assess scene security and ensure scene security following the reported train crash in LAC-MEGANTIC, Quebec.`

| Model | Reward | Relevance | Factuality | Role coverage | Urgency |
|---|---:|---:|---:|---:|---:|
| GPT | 0.6769 | 0.7183 | **0.5020** | 1.0 | 0.5 |
| T5 | **0.7765** | **0.7343** | 0.4778 | 1.0 | **1.0** |
| DPO | 0.4744 | 0.7283 | 0.4781 | 0.0 | 0.5 |

The scorer correctly penalizes this DPO output through role coverage and urgency, but relevance and factuality still keep the final reward near 0.47. The example illustrates both a model regression and the limited dynamic range of the current factuality component.

## 8. What The Results Support

### Supported findings

1. **A compact reusable baseline is feasible.** T5-small learned a strong approximation of the teacher's role-aware summary style from 4,800 training records.
2. **The reward pipeline is operational and model-sensitive.** It separates models and supports preference construction at scale.
3. **DPO optimization is technically successful.** The policy learned the preference relation and raised the specified composite reward.
4. **Role-aware optimization changes behavior materially.** DPO dramatically increases Police role-coverage scores and improves some genuine EMS and multi-role cases.
5. **Aggregate metrics conceal important failure modes.** Exact-role breakdowns and text inspection reveal regressions that the overall mean obscures.

### Findings not yet supported

1. **DPO cannot yet be claimed to be the best summarizer.** It has the highest current automated reward but lower held-out reference similarity and visible reward exploitation.
2. **The current reward is not yet a validated proxy for human preference.** No systematic blinded human evaluation has been performed.
3. **MiniCheck scores do not currently establish comparative factuality.** Their variance is too small in this setup.
4. **Results do not yet generalize to `Other` tweets.** This run covers responder-relevant rows only.
5. **All-data reward means are descriptive, not pure generalization estimates.** They include training and validation records; held-out metrics use only the 601-example test set.

## 9. Recommended Core Metrics Going Forward

Future experiments should keep the following metrics together rather than selecting a single winner:

- held-out ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERTScore;
- mean, median, P10, and P90 composite reward;
- every reward component separately;
- exact-role reward and component breakdowns;
- DPO preference accuracy and implicit reward margin;
- repetition and generic-template rates;
- role-label consistency, especially for multi-role examples;
- a blinded human rubric covering factual support, tweet salience, role actionability, non-redundancy, and overall preference.

A stratified human audit is the most important next measurement. Even 100 to 200 examples balanced across exact role labels and reward-disagreement cases would help determine whether automated reward differences correspond to useful summaries.

## 10. Key Artifact Index

The surrounding `reports/key_results/` directory contains copies of the most important evidence files. Original report files remain unchanged in their prior locations.

| Artifact | Purpose |
|---|---|
| `frecs_dataset_inventory.xlsx` | Raw FReCS counts, label distributions, duplicates, and column inventory |
| `t5_small_baseline_v1_metrics.csv` | Initial 4,001-example baseline held-out metrics |
| `t5_small_baseline_v2_metrics.csv` | Current 6,001-example baseline held-out metrics |
| `t5_small_dpo_v1_beta_0_1_metrics.csv` | DPO held-out reference-similarity metrics |
| `three_model_reward_comparison.csv` | Overall GPT/T5/DPO reward and length means |
| `three_model_reward_comparison_by_role.csv` | Exact-role comparison across all reward components |
| `gpt4o_reward_overall_metrics.csv` | GPT reward quantiles and dispersion |
| `t5_baseline_v2_reward_overall_metrics.csv` | T5 v2 reward quantiles and dispersion |
| `t5_dpo_reward_overall_metrics.csv` | DPO reward quantiles and dispersion |
| `*_reward_distributions.csv` | Histogram-style distributions for each model and metric |
| `dpo_preference_summary.csv` | Preference-pair counts, margins, chosen models, and splits |
| `dpo_preferences_by_role.csv` | Preference behavior by exact role label |
| `dpo_preference_margin_distribution.csv` | Strength distribution of retained preferences |
| `final_gpt_t5_dpo_summary_reward_report.csv` | Human-readable row-level comparison of inputs, three summaries, and metric breakdowns |
| `t5_dpo_qualitative_samples_by_role.jsonl` | Stratified examples selected for qualitative DPO analysis |

## Bottom Line

The project has moved beyond a simple baseline demonstration. It now provides a reproducible synthetic-data pipeline, a trained compact summarizer, a role-aware reward function, a preference dataset, a DPO implementation, and paired quantitative and qualitative evaluation.

The most important scientific result is the tension between objectives: supervised T5 closely imitates the GPT teacher, while DPO learns the automated reward well enough to expose its weaknesses. That tension is useful evidence. It identifies the next research problem clearly: improve and human-calibrate the reward so that higher scores reliably correspond to more faithful, salient, role-appropriate, and non-redundant crisis summaries.
