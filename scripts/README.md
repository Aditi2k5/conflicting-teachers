# Experiment scripts

Run these scripts from the repository root. The maintained workflow is the
pretrained-model experiment; older scenario-generation, chat-model, and
one-off repair scripts have been removed.

## Data

- `data/build_mixtures_pt.py` builds pretrained-model training mixtures from
  `data/processed_v2d/`.

## Hard evaluation set

These scripts reproduce the hard evaluation set. Candidate review is manual:

1. `benchmark/generate_hard_candidates.py`
2. `benchmark/validate_hard_candidates.py`
3. `benchmark/create_hard_candidate_review.py` — creates the review CSV.
4. Complete the review CSV, then run `benchmark/freeze_hard_base.py`.
5. `benchmark/expand_hard_paraphrases.py`
6. `benchmark/validate_hard_eval.py`

The canonical frozen evaluation set is
`data/eval_v2/conflict_ood_hard_200_FROZEN.jsonl`.

## Evaluation

`evaluation/evaluate_logprob_pt.py` scores the pretrained model (or an adapter)
against an evaluation JSONL file. Pass `--eval` and `--output`; optionally pass
`--adapter` to evaluate a trained adapter.

## Analysis

- `analysis/summarize_hard_results.py` summarizes the hard-evaluation runs.
- `analysis/prevalence_regression.py` analyzes the prevalence trend.
- `analysis/hierarchical_bootstrap.py` compares conditions with a hierarchical
  bootstrap.
- `analysis/instability_analysis.py` measures paraphrase instability.
- `analysis/plot_final_results.py` creates the final figures.

The analysis scripts read prediction files from `runs/predictions_pt/` and
write reports and figures under `results/pt/`.
