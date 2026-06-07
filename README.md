# Role-Aware Crisis Summarization

This project is part of an independent study on role-aware crisis communication using the FReCS dataset. The goal is to build a role-conditioned summarization pipeline that converts disaster-related tweets into concise operational summaries tailored to different first responder roles such as EMS, Police, and Firefighters.

The project extends the idea of first responder classification into a summarization setting. Instead of only predicting which responder type is relevant to a tweet, the system generates a short role-specific summary describing what that responder group should pay attention to.

## Project Goals

The main goals are:

1. Prepare and inspect the FReCS crisis tweet dataset.
2. Construct a role-aware summarization schema.
3. Generate an initial supervised summarization dataset.
4. Fine-tune a baseline T5-style model.
5. Evaluate generated summaries using ROUGE, BLEU, BERTScore, and role-specific checks.
6. Develop automated reward scoring for relevance, factuality, role coverage, and urgency.
7. Experiment with preference optimization, such as DPO, to improve role-aware summaries.

## Project Structure

```text
role-aware-crisis-summarization/
  data/
    raw/          # Original datasets kept locally and not committed
    interim/      # Intermediate cleaned files
    processed/    # Final processed datasets
    generated/    # Generated summaries and synthetic targets
    splits/       # Train/dev/test splits

  notebooks/      # Exploratory notebooks

  reports/
    tables/       # Dataset inventories and metric tables
    figures/      # Figures for reports and paper drafts

  src/
    crisis_summarization/
      data/        # Data loading, cleaning, schema construction, splitting
      generation/  # Prompting and target summary generation
      training/    # Model training and prediction scripts
      evaluation/  # Metrics and role-aware evaluation
      utils/       # Shared helpers

  models/          # Local model checkpoints, not committed