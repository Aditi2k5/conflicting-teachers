# Conflicting Teachers

This repository contains the pretrained-model experiment on how mixtures of
good and bad teacher demonstrations affect privacy and oversight decisions.
The maintained pipeline uses MLX on Apple Silicon and the
`mlx-community/gemma-3-text-4b-pt-4bit` base model.


## Environment

The training and scoring commands require macOS on Apple Silicon because MLX
is the model runtime. Use Python 3.11 and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the base model from Hugging Face. Ensure the machine
has enough memory and disk space for the model and adapter outputs.

## Reproduce the experiment

Run commands from the repository root.

### 1. Build a training mixture

The processed teacher data and generated mixtures are committed. To recreate
one mixture, use the same mixture seed and condition. For example:

```bash
python scripts/data/build_mixtures_pt.py \
  --name A_100GS \
  --mixture GS=1.0 \
  --total 200 \
  --seed 42
```

The main prevalence conditions are `A_100GS` (0% bad), `B_80GS_20BS` (20%),
`C_60GS_40BS` (40%), `D_50GS_50BS` (50%), `J_40GS_60BS` (60%),
`K_20GS_80BS` (80%), and `H_100BS` (100%). The additional teacher-strength
conditions are `E_80GW_20BS`, `F_80GS_20BW`, `G_100GW`, `I_100BW`, and
`L_80GW_20BW`. Their exact sampled data is in `data/mixtures_pt/`.

### 2. Train an adapter

This example reproduces the `A_100GS`, seed-42 adapter:

```bash
mlx_lm.lora \
  --model mlx-community/gemma-3-text-4b-pt-4bit \
  --train \
  --data data/mixtures_pt/A_100GS \
  --adapter-path runs/adapters_pt/A_100GS_seed42 \
  --fine-tune-type lora \
  --optimizer adam \
  --num-layers 4 \
  --batch-size 1 \
  --iters 200 \
  --learning-rate 0.0001 \
  --max-seq-length 2048 \
  --save-every 100 \
  --grad-checkpoint \
  --seed 42
```

The adapter configuration records LoRA rank 8, scale 20, dropout 0, and the
remaining training settings. The prevalence analysis uses mixture seeds
`42, 123, 2026, 7, 999` for A, B, C, D, J, K, and H. The other five conditions
use mixture seeds `42, 123, 2026`. The seed in each run name identifies the
mixture sampling seed; it is not always the MLX optimizer seed. The recorded
training configs use MLX seed `0` for mixture seeds `42`, `123`, and `2026`,
and MLX seeds `7` and `999` for the corresponding runs with those mixture
seeds. Preserve this distinction when reproducing the recorded adapters.
For non-default mixture seeds, build the corresponding
`--name <CONDITION>_seed<SEED>` mixture and use its directory for `--data`.
Set the MLX `--seed` according to the training seed described above.

### 3. Score the frozen benchmark

The canonical dataset is
`data/eval_v2/conflict_ood_hard_200_FROZEN.jsonl` (200 scenarios, 50 dilemma
groups, four variants per group). To score the base model:

```bash
python scripts/evaluation/evaluate_logprob_pt.py \
  --eval data/eval_v2/conflict_ood_hard_200_FROZEN.jsonl \
  --output runs/predictions_pt/PT_BASE_hard200.jsonl
```

To score an adapter, pass its directory with `--adapter` and save the output
using the pattern
`runs/predictions_pt/<CONDITION>_seed<SEED>_hard200.jsonl`. For example:

```bash
python scripts/evaluation/evaluate_logprob_pt.py \
  --adapter runs/adapters_pt/A_100GS_seed42 \
  --eval data/eval_v2/conflict_ood_hard_200_FROZEN.jsonl \
  --output runs/predictions_pt/A_100GS_seed42_hard200.jsonl
```

The scorer compares token-normalized log probabilities for each action label
and writes aligned margins and predictions used by the analyses.

### 4. Recreate reports and figures

The analysis scripts read the prediction files in `runs/predictions_pt/`:

```bash
python scripts/analysis/summarize_hard_results.py
python scripts/analysis/prevalence_regression.py
python scripts/analysis/hierarchical_bootstrap.py
python scripts/analysis/instability_analysis.py
python scripts/analysis/plot_final_results.py
```

The final reports and figures are written to or stored in `results/pt/`.
`scripts/README.md` describes the benchmark-construction steps, including
manual candidate review.

## Repository layout

- `data/processed_v2d/`: processed teacher demonstrations
- `data/mixtures_pt/`: prepared pretrained-model training splits
- `data/eval_v2/`: hard evaluation benchmark and its construction inputs
- `runs/predictions_pt/`: base and adapter predictions on the frozen benchmark
- `runs/adapters_pt/`: locally generated LoRA adapters (ignored by Git)
- `results/pt/`: saved analysis, annotations, and figures
- `scripts/`: mixture, benchmark, evaluation, and analysis programs
