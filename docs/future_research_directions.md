# Future Research Directions

This document preserves possible research branches that emerged after the first
large-scale GPT teacher versus T5 baseline comparison. These are not immediate
implementation commitments. They are future considerations for extending the
project after the current reward-modeling and DPO exploration is complete.

## Current Baseline Insight

The current project state has a surprisingly strong supervised `t5-small`
baseline. After training on 6,001 GPT-generated teacher summaries, T5 v2 scored
close to the GPT teacher summaries under the role-aware reward function:

```text
T5 v2 mean reward:     0.593620
GPT teacher reward:    0.617690
GPT minus T5:         +0.024071
```

This result changes the research story. The smaller student model is not simply
a weak baseline waiting to be rescued by DPO. Instead, it appears capable of
absorbing much of the prompted teacher's behavior for a narrow, structured,
one-sentence crisis summarization task.

The useful interpretation is:

```text
GPT teacher summaries provide a scalable synthetic labeling policy.
T5-small can distill much of that policy into a reusable local model.
Reward analysis identifies the residual role-aware weaknesses.
```

This makes future work more precise. The question is not only whether a model
can summarize crisis tweets, but what context it needs, what the teacher prompt
contributes, and whether preference optimization can improve the remaining
role-specific weaknesses.

## 1. Better Synthetic Teacher Prompting

The current GPT teacher summaries were generated with prompt version 2 and
input-text version 3. The prompt produced consistent one-sentence operational
summaries, which made the data learnable by T5. That consistency is useful, but
it may also mean the teacher policy is somewhat formulaic.

Future prompt versions could test whether GPT teacher quality improves when the
prompt more strongly emphasizes:

- concrete tweet-specific evidence
- avoiding generic responder phrasing
- distinguishing observed facts from inferred responder actions
- explicit multi-role balancing
- role-specific actionability without over-inference
- short but less template-like wording

A future prompt could require the model to identify the source evidence before
writing the summary, while still only saving the final concise summary. Another
variant could ask the model to explicitly state why each role is relevant before
synthesizing a final integrated summary.

Potential research question:

```text
Does a more evidence-focused teacher prompt produce summaries that improve
student model training and role-aware reward scores?
```

## 2. Stronger Teacher Models

The current teacher was GPT-4o-mini through the project generation pipeline.
This was practical for cost and scale, but it may not represent the best
possible prompted teacher. A stronger model could generate more nuanced,
better-grounded, or more role-complete summaries.

Future teacher comparisons could include:

- GPT-4o-mini as the current practical teacher
- a stronger GPT model for a smaller but higher-quality subset
- another strong commercial or open model as an alternate teacher

The key comparison should not only be raw quality. It should ask whether the
student model benefits from the stronger teacher enough to justify the extra
generation cost.

Potential research question:

```text
Does a stronger teacher model produce synthetic summaries that train a better
small reusable student model, or does the constrained task saturate quickly?
```

## 3. Factuality Calibration

The current reward function uses MiniCheck factuality as a sentence-level
support probability. This is conceptually valuable because the reward should
discourage unsupported summaries. However, the first full comparison showed
that factuality barely separated GPT teacher summaries from T5 predictions:

```text
T5 v2 factuality mean:  0.480305
GPT factuality mean:    0.478697
Difference:            -0.001608
```

The distribution was tightly compressed near the decision boundary. This means
that factuality currently behaves more like a weak grounding prior or near-
constant offset than a strong ranking signal.

Future factuality experiments could test:

- lowering factuality weight from `0.25` to a smaller value
- rescaling MiniCheck probabilities across the observed support range
- using factuality as a thresholded penalty rather than a normal reward term
- separating tweet-grounded factuality from metadata-grounded appropriateness
- comparing MiniCheck with another factuality or entailment backend

The present project should keep the current MiniCheck setup for the current
exploration to preserve comparability. Future work can treat factuality
calibration as a separate reward-ablation study.

Potential research question:

```text
Is MiniCheck factuality more useful as a continuous reward component, a
rescaled score, or a guardrail penalty for unsupported claims?
```

## 4. Salience Or Key-Information Coverage

The current reward function has four components:

```text
0.35 relevance + 0.25 factuality + 0.20 role_coverage + 0.20 urgency
```

Relevance captures semantic relatedness, role coverage captures responder-
specific criteria, and urgency captures operational severity. One missing
dimension is salience: whether the summary preserves the most important
information in the tweet.

This matters because a summary can be relevant and role-flavored but still miss
the core point. For example, a flood tweet about fatalities may receive a
generic flood-response summary that is relevant but misses the casualty signal.

Possible salience approaches:

- keyphrase overlap between tweet and summary
- named entity and location preservation
- numeric/casualty/event-count preservation
- dependency- or attention-inspired importance scoring
- comparing summary coverage of high-information tweet tokens
- human-interpretable salience flags such as casualty, location, hazard, need,
  and action

The challenge is to avoid rewarding parroting. A salience component should not
force the model to copy the tweet. It should reward preserving critical
information while still allowing concise responder-oriented abstraction.

Potential research question:

```text
Can a salience coverage term reduce generic summaries without forcing lexical
copying from the tweet?
```

## 5. Input-Context Ablation Studies

The current T5 v2 baseline uses a rich input format:

```text
tweet + responder roles + disaster type + secondary annotations + information type
```

This makes the task learnable, but it raises an important question:

```text
Did the model learn crisis summarization, or did it learn to translate
structured labels into a summary template?
```

Input ablation runs could train matched T5 baselines with progressively less
context:

1. Full context:
   `tweet + roles + disaster type + secondary annotations + information type`
2. No information type:
   `tweet + roles + disaster type + secondary annotations`
3. No secondary annotations:
   `tweet + roles + disaster type + information type`
4. No disaster type:
   `tweet + roles + secondary annotations + information type`
5. Tweet plus roles:
   `tweet + roles`
6. Tweet only:
   `tweet`
7. Metadata only:
   `roles + disaster type + secondary annotations + information type`

The most important compact set would be:

- full context
- tweet plus roles
- tweet only
- metadata only

This would reveal whether full structured context is necessary, whether roles
alone carry most of the useful signal, whether tweet-only summarization is
viable, and whether the teacher summaries are too template-driven.

Potential research questions:

```text
What is the minimum input context required for useful role-aware crisis
summarization?

Can a reusable model produce competent summaries from tweet-only input, or does
it require upstream metadata such as roles and disaster type?
```

## 6. DPO As Targeted Optimization

The strong T5 v2 baseline changes how DPO should be motivated. DPO is not
needed simply because the supervised model is weak. Instead, it should be framed
as a targeted optimization method for residual failure modes.

The reward comparison suggests those residual weaknesses include:

- multi-role summaries
- role coverage gaps
- urgency coverage gaps
- generic but plausible summaries
- summaries that match the disaster but miss the tweet-specific operational cue

DPO could use preference pairs where one candidate has a higher reward than
another for the same input. Candidate pairs might come from:

- GPT teacher summary versus T5 prediction
- multiple T5 checkpoints or decoding settings
- future prompt variants
- future input-ablation models
- generated alternatives from the same model

The most interesting DPO setup may not be the full-context model, which is
already strong. DPO may be more revealing on weaker or more realistic settings,
such as tweet-only or tweet-plus-role inputs.

Potential research question:

```text
Can reward-guided preference optimization improve role coverage and urgency
without sacrificing tweet relevance or factual grounding?
```

## 7. Model Comparison As A Broader Research Frame

The project is evolving toward a three-system comparison:

1. Prompted GPT teacher summaries
2. Supervised T5 student summaries
3. Future DPO-optimized T5 summaries

This can be expanded into a broader model-comparison framework. Each system can
be evaluated with:

- ROUGE, BLEU, and BERTScore against synthetic teacher targets
- role-aware reward
- component reward breakdowns
- exact role-label breakdowns
- disaster-type and information-type breakdowns
- qualitative examples of high- and low-reward outputs

The central insight is that the reward function becomes the common evaluator.
Traditional metrics measure similarity to the teacher. The reward function
measures whether summaries satisfy the project's role-aware operational
criteria.

Potential research question:

```text
How do prompted teacher models, supervised student models, and preference-
optimized student models differ across role-aware reward dimensions?
```

## 8. Practical Deployment Question

The project has a practical motivation beyond model comparison. A prompted GPT
teacher requires repeated API calls and external service access. A trained T5
student is frozen, reusable, cheaper to run, and easier to deploy locally.

Future work should keep asking:

```text
What model and input setup gives the best tradeoff between quality,
deployability, cost, and required metadata?
```

Possible outcomes:

- Full-context T5 is best but requires upstream classifiers or annotations.
- Tweet-plus-role T5 is slightly weaker but much more practical.
- Tweet-only T5 is viable for rough summaries but weaker on role specificity.
- DPO improves a reduced-context model enough to make it practically useful.
- A stronger GPT teacher improves the student only marginally, suggesting the
  task saturates under the current schema.

These outcomes would all be publishable or professor-discussion-worthy because
they explain the cost and practicality of role-aware crisis summarization, not
just raw metric performance.

## Suggested Future Experiment Order

A realistic future sequence would be:

1. Finish the current DPO-preparation path using the existing reward function.
2. Run a compact input-context ablation study with full context, tweet plus
   roles, tweet only, and metadata only.
3. Re-score each ablation model with the same reward function.
4. Identify where supervised learning fails most clearly.
5. Apply DPO to the most informative failure setting.
6. Separately test a stronger or revised GPT teacher prompt on a smaller sample.
7. Revisit factuality calibration and possible salience coverage as reward v2.

This keeps the current project stable while preserving a clear map of future
extensions.
