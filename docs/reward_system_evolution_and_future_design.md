# Reward System Evolution And Future Design

## Purpose

This document explains how the project's reward system evolved, what the first
large-scale experiments revealed, where the current scorer succeeds and fails,
and how a future reward design could be improved.

It is intentionally separate from `reward_specification_v1.md`:

- `reward_specification_v1.md` is the frozen, reproducible definition used for
  the completed GPT teacher, supervised T5 v2, and T5-DPO beta 0.1 runs.
- This document is an analytical record and design space for future reward
  versions. Its proposals are not yet implemented or validated.

## 1. Why The Project Needed A Reward Function

ROUGE, BLEU, and BERTScore measure similarity to a reference summary. They are
useful for determining whether T5 learned the GPT teacher's output style, but
they do not directly answer the project's operational questions:

- Is the summary grounded in the actual tweet?
- Does it preserve urgent information?
- Is it useful for the assigned responder role or roles?
- Does it avoid unsupported claims?

The reward function was introduced to evaluate these task-specific properties
and later provide preference labels for Direct Preference Optimization (DPO).

## 2. Reward System Journey

### 2.1 Initial Transparent Scorer

The first scorer combined semantic relevance, a local factuality proxy,
role-specific keyword coverage, and urgency terms:

```text
reward = 0.35 relevance
       + 0.25 factuality
       + 0.20 role_coverage
       + 0.20 urgency
```

This version was intentionally transparent. It made component behavior easy to
inspect and gave the project a reproducible baseline.

### 2.2 Tweet-Dominant Relevance

Early qualitative inspection found that generic role/disaster summaries could
receive too much relevance credit. Relevance was therefore split into tweet and
metadata/context similarity:

```text
relevance = 0.70 * tweet_relevance + 0.30 * context_relevance
```

This made the scorer more sensitive to the source tweet while retaining some
credit for role and disaster alignment.

### 2.3 MiniCheck Factuality

The local token-grounding proxy was retained for lightweight smoke tests, but
MiniCheck became the preferred factuality backend for final reward runs.
MiniCheck treats each summary sentence as a claim, checks support against the
source context, and returns a support probability. Since nearly all project
summaries contain one sentence, factuality is usually one probability per
summary.

### 2.4 Role And Urgency Tightening

Role vocabularies were expanded using actual teacher summaries, FReCS labels,
and secondary annotations. Phrase matching was made punctuation-aware. Urgency
was changed from exact-term overlap to concept coverage:

- casualty/injury
- rescue/evacuation
- active hazard
- severity/threat

These changes improved interpretability and reduced simple synonym failures.

### 2.5 Full-Scale Reward Comparison

The stabilized v1 scorer evaluated 6,001 GPT teacher summaries and 6,001
supervised T5 v2 summaries:

| System | Reward | Relevance | Factuality | Role coverage | Urgency |
|---|---:|---:|---:|---:|---:|
| GPT teacher | 0.617690 | 0.724957 | 0.478697 | 0.508129 | 0.713276 |
| T5 v2 | 0.593620 | 0.719472 | 0.480305 | 0.471233 | 0.637408 |

GPT scored modestly higher overall. The difference was driven mainly by role
coverage and urgency, while relevance was close and factuality was nearly
identical.

### 2.6 DPO Preference Optimization

GPT and T5 summaries were paired by `tweet_id`. The higher-reward summary
became `chosen`, the lower-reward summary became `rejected`, and reward margins
below 0.03 were skipped. This produced 3,891 preference pairs.

The first T5-DPO run improved the v1 reward:

| System | Reward | Relevance | Factuality | Role coverage | Urgency |
|---|---:|---:|---:|---:|---:|
| GPT teacher | 0.617690 | 0.724957 | 0.478697 | 0.508129 | 0.713276 |
| T5 v2 | 0.593620 | 0.719472 | 0.480305 | 0.471233 | 0.637408 |
| T5-DPO beta 0.1 | 0.653345 | 0.705166 | 0.479749 | 0.751469 | 0.681531 |

The DPO gain generalized to held-out test inputs:

```text
GPT teacher test reward: 0.615422
T5 v2 test reward:       0.595655
T5-DPO test reward:      0.646819
```

However, traditional similarity metrics decreased. DPO moved away from teacher
imitation while optimizing the automated reward.

## 3. What DPO Revealed About Reward V1

Compared with supervised T5, DPO gained approximately 0.0597 reward. The
weighted component contributions were approximately:

```text
role coverage: +0.0560
urgency:       +0.0088
relevance:     -0.0050
factuality:    -0.0001
```

Role coverage alone explains almost the entire net increase. DPO successfully
learned the objective, but it also identified the easiest path through it.

Across 6,001 outputs:

- `scene security` appeared in 93.7% of DPO summaries
- `scene security` appeared at least twice in 38.5% of DPO summaries
- only 54.7% of DPO summaries were unique
- one exact Police summary appeared 100 times

For comparison:

| Behavior | GPT | T5 v2 | T5-DPO |
|---|---:|---:|---:|
| Contains `scene security` | 12.6% | 21.1% | 93.7% |
| Repeats it at least twice | 0% | approximately 0% | 38.5% |
| Unique summaries | 94.5% | 65.9% | 54.7% |

Police accounts for 3,280 of 6,001 records. Its role coverage increased from
0.459959 for T5 v2 to 0.902795 for DPO. This dominant class contributed most
of the overall reward gain, and Police-style language leaked into other roles.

## 4. Qualitative Examples

The examples below come from
`t5_dpo_v1_beta_0_1_qualitative_samples_by_role.jsonl`.

### 4.1 Genuine Improvement: EMS Relief Tweet

Tweet `5217` asks readers to support UNICEF emergency relief for children in
the Philippines after Typhoon Haiyan.

```text
GPT:
EMS should evaluate medical needs and support for vulnerable children in
typhoon-affected regions as relief efforts are mobilized.

T5:
EMS should assess crowd control needs and coordinate with UNICEF in response
to the typhoon.

DPO:
Assess medical needs and coordinate with UNICEF for emergency relief efforts
for kids in the Philippines.
```

DPO preserved UNICEF, children, Philippines, relief efforts, and medical needs.
Its reward of 0.814753 agrees with human reading: this is a legitimate
improvement over T5's irrelevant crowd-control framing.

### 4.2 Correctly Penalized Role Failure

Tweet `103` reports a train crash death toll of 20. The assigned role is EMS.

```text
T5:
EMS should assess casualties from the train crash and coordinate with MMU for
medical response.

DPO:
Police should assess scene security and ensure scene security following the
reported train crash in LAC-MEGANTIC, Quebec.
```

DPO switched to the wrong responder role and repeated scene-security language.
The scorer assigned zero role coverage and lowered the DPO reward to 0.474410,
well below T5's 0.776471. This is evidence that reward v1 can detect clear role
misalignment.

### 4.3 Reward-Inflated Firefighter Summary

Tweet `4074` concerns an illegally constructed collapsed factory.

```text
GPT:
Firefighters should evaluate the collapsed factory's stability and be ready
for search and rescue efforts.

DPO:
Firefighters should assess fire hazards and ensure scene security following
the collapse and ensure scene security.
```

GPT is more coherent and operationally appropriate. DPO repeats scene security
and introduces generic fire hazards, yet its reward is 0.749 compared with
GPT's 0.568. The DPO summary matched configured hazard and structural concepts
without preserving the most useful action: structural assessment and rescue.

### 4.4 Genuine Police Improvement With Specific Evidence

Tweet `1059` reports that more than 1,750 civilians and hundreds of pets were
evacuated from a flood.

```text
T5:
Police should assess crowd control needs and coordinate response efforts
following the reported flooding.

DPO:
Police should assess crowd control and ensure scene security following the
evacuation of more than 1750 civilians and hundreds of pets from the flood.
```

DPO retained the number, affected populations, evacuation, and flood. Although
it uses the learned Police template, the output is more informative and its
higher reward is defensible.

### 4.5 Multi-Role Keyword And Annotation Stuffing

Tweet `3981` reports at least 20 deaths after an earthquake that collapsed
buildings and damaged roads. Roles are Police, Firefighter, and EMS.

```text
GPT:
Responders should secure the scene, prepare for search and rescue, and mobilize
medical units for potential casualties after the earthquake.

DPO:
Police should ensure scene security, firefighters, and EMS should ensure scene
security, while EMS prepares for casualties and coordinates with DCC, USAR,
and MMU.
```

DPO received a higher reward despite malformed coordination and mechanical use
of all three annotation abbreviations. This is the clearest example of a
summary satisfying the scorer's categories without matching human preferences
for fluency, concision, and integrated multi-role reasoning.

## 5. Component-By-Component Assessment

### 5.1 Relevance

Tweet-dominant relevance is directionally sound. It decreased when DPO became
more generic, which is the correct response. Its weakness is magnitude: generic
templates still retain moderate semantic similarity because they repeat the
role, disaster, and operational vocabulary.

Future improvements could:

- increase tweet weight from 0.70 to 0.80
- measure entity, number, location, and event preservation separately
- distinguish broad semantic similarity from salient-information coverage
- report whether similarity is driven primarily by metadata terms

### 5.2 Factuality

MiniCheck factuality barely varied across systems:

```text
GPT: 0.478697
T5:  0.480305
DPO: 0.479749
```

It therefore behaved as a weak grounding prior or near-constant offset. It did
not counter repetitive, malformed, or generic summaries because those are not
necessarily factuality failures.

Future options include:

- reducing its additive weight
- using it as a thresholded guardrail or penalty
- separating tweet-supported facts from metadata-supported role framing
- calibrating its probabilities against human support judgments
- testing a more discriminative entailment/factuality backend

### 5.3 Role Coverage

Role coverage is both useful and highly exploitable. It correctly identifies
missing medical, rescue, security, and traffic concepts, but category coverage
can often be satisfied by one configured phrase.

Future role scoring should:

- score every assigned role separately and macro-average multi-role results
- penalize actions belonging to unassigned roles
- use semantic action descriptions rather than keyword presence alone
- give no extra credit for repeated terms
- require source/tweet evidence for inferred operational actions
- distinguish mentioning a role concept from expressing a coherent action

### 5.4 Urgency

Category-based urgency handles synonyms better than exact-term overlap, but can
still be optimized by inserting one casualty, evacuation, hazard, or threat
term.

Future urgency should separate:

```text
urgency recall:
Did the summary preserve urgent source evidence?

urgency precision:
Did the summary introduce unsupported urgency?
```

### 5.5 Missing Salience Coverage

A summary may be relevant but miss the tweet's central information. A future
salience component could measure preservation of:

- casualty/injury counts
- locations and organizations
- affected populations
- evacuations and requests for assistance
- infrastructure damage
- active hazards and operational constraints

This component should reward semantic preservation, not verbatim copying.

### 5.6 Missing Degeneration Penalty

Reward v1 has no direct cost for repetition, low diversity, malformed role
coordination, or template collapse. A future degeneration score could include:

- repeated clauses and n-grams
- repeated role/action phrases
- exact-output duplication frequency
- malformed or incomplete coordination
- excessive similarity to a small set of templates

A multiplicative penalty may be safer than another additive component:

```text
final_reward = quality_reward * (1 - penalty_strength * degeneration)
```

This prevents high keyword coverage from fully compensating for severe output
degeneration.

## 6. Optional Specialized Context Adherence

The general reward should remain usable when only a tweet is available.
However, FReCS provides additional structured context that can support a
specialized evaluation mode:

- responder role label
- disaster type
- information type
- secondary responder annotation

When those fields are available to the model, it is reasonable to evaluate how
well the summary respects them. This should be implemented as a separate score,
not silently folded into general quality.

### 6.1 Proposed Context Components

```text
context_adherence =
    role_alignment
  + disaster_alignment
  + information_type_alignment
  + secondary_annotation_alignment
  - role_leakage
```

Weights should be normalized across only the enabled fields.

### 6.2 Role Alignment

Role alignment should evaluate coherent actions for each assigned role and
penalize unassigned-role leakage. For multi-role labels, each role should be
scored separately before macro-averaging so Police cannot dominate EMS or
Firefighter behavior.

### 6.3 Disaster Alignment

Disaster alignment should evaluate operational compatibility, not simple label
mentioning. Examples include:

- wildfire: smoke, spread, containment, exposure
- collapse: stability, trapped people, structural rescue
- flood: access, rising water, evacuation, displacement
- shooting: active threat, scene safety, injuries
- typhoon: evacuation, shelter, damage, relief

The disaster label alone must not justify unsupported claims. A Firefighter row
for a typhoon should not automatically receive credit for mentioning fire
spread.

### 6.4 Information-Type Alignment

Information types can provide soft expectations:

- `Affected individuals`: people, casualties, injuries, displacement, needs
- `Infrastructure and utilities`: roads, buildings, power, communications
- `Donations and volunteering`: aid, supplies, volunteers, organizations

FReCS labels may be noisy, so this component should remain low-weight and
diagnostic until validated.

### 6.5 Secondary-Annotation Alignment

Secondary annotations should map to semantic action descriptions:

```text
DCC    dispatch and coordination
TEU    traffic and access enforcement
CPU    crime prevention or investigation
MMU    mobile medical response
CERT   community emergency assistance
FC     fire control
HAZMAT hazardous-material response
USAR   urban search and rescue
```

Literal abbreviation copying should not be required or independently rewarded.
The summary should express the associated action coherently. Mechanical
insertion of `DCC`, `USAR`, and `MMU` should be penalized as annotation stuffing.

## 7. Flags And Ablation Compatibility

A future scorer could accept explicit context flags:

```text
--context-fields role,disaster_type,information_type,secondary_annotations
```

or separate switches:

```text
--context-role
--context-disaster
--context-information-type
--context-secondary-annotation
```

Two evaluation modes are important:

### Observed Context

Score only metadata actually supplied to the model. This is the fair comparison
for ablation runs.

### Oracle Context

Score against all dataset metadata even when hidden from the model. This asks
whether a tweet-only model can infer the latent role/disaster context.

Examples:

```text
Tweet-only + observed mode:
Evaluate general summary quality without metadata requirements.

Tweet-only + oracle mode:
Measure whether the model implicitly recovers hidden FReCS context.
```

## 8. Candidate Future Architecture

Versioning should keep general quality, optional specialization, and penalties
separate:

```text
reward_v1
    Frozen scorer used for completed experiments.

general_reward_v2
    Tweet relevance, factuality guardrail, salience, and urgency precision/recall.

context_adherence_v1
    Optional metadata-aware role/disaster/information/annotation alignment.

degeneration_penalty_v1
    Repetition, templating, malformed coordination, and role leakage.

specialized_reward_v1
    General quality + enabled context adherence - degeneration penalty.
```

An initial experimental combination could be:

```text
combined_quality = 0.75 * general_reward_v2
                 + 0.25 * context_adherence_v1

specialized_reward = combined_quality
                   * (1 - penalty_strength * degeneration_penalty_v1)
```

These values are design hypotheses, not finalized weights.

## 9. Preference Construction Safeguards

The first DPO run demonstrated that scalar reward ranking alone can propagate
reward weaknesses into the model. Future preference construction should use
constraints in addition to a total score.

A candidate should not become `chosen` if it has:

- severe repetition
- malformed or incomplete output
- clear role leakage
- unsupported casualty or hazard claims
- substantially lower tweet relevance without compensating salient coverage

A possible decision sequence is:

1. Reject candidates that fail degeneration or role-consistency checks.
2. Require a minimum factual-support threshold.
3. Compare salience and tweet relevance.
4. Compare role/urgency/context adherence.
5. Require a meaningful preference margin.

This is more robust than allowing one large role-coverage gain to compensate for
every other weakness.

## 10. Human Calibration

Automated components should be calibrated against human reading before reward
v2 is used for another optimization run. A manageable validation study could
sample 10 summaries per exact role label and collect ratings for:

- tweet faithfulness
- salient-information preservation
- role appropriateness
- urgency preservation without exaggeration
- fluency and concision
- redundancy/template behavior
- overall preference between candidate summaries

The project can then measure component correlations with human judgments and
adjust thresholds or weights accordingly.

## 11. Recommended Next Steps

1. Keep reward v1 frozen and retain all completed results unchanged.
2. Add automated output-degeneration analysis before changing the reward.
3. Create a small blinded human-rating sample from GPT, T5, and DPO outputs.
4. Prototype salience coverage and role-leakage detection as diagnostics.
5. Prototype optional metadata-aware context adherence with observed/oracle
   modes.
6. Calibrate components against human ratings.
7. Define and version reward v2 only after the diagnostics are understood.
8. Rebuild preference pairs under the revised scorer before another DPO run.

## Conclusion

Reward v1 was successful as a transparent first evaluator and as a mechanism
for producing a working DPO pipeline. Its weaknesses became visible precisely
because optimization was strong enough to exploit them. The DPO experiment
therefore provides more than a model result: it supplies empirical requirements
for the next reward version.

The future objective is not simply to add more reward terms. It is to create a
scoring system where improved reward more consistently corresponds to improved
human judgment, while preserving the transparency and role-aware focus that
made reward v1 useful.
