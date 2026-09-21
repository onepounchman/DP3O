# DP3O

This repository contains the code for the TMLR paper [Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation](https://openreview.net/forum?id=X6r1bU1m6x).

## Getting Started

Edit `.env` to set `DP3O_STORAGE` to an absolute directory outside this repository, select your GPUs with `CUDA_VISIBLE_DEVICES`, and provide `HF_TOKEN` if needed. Obtain access to the required models on Hugging Face before downloading them. Datasets, model weights, and generated outputs are stored externally and are not included in this repository.

Install the environment:

```bash
bash setup/install.sh
```

The scripts load `.env` and use the configured Python environment automatically.

## Experiments

Run the following stages in order, using the same run name throughout.

### Data Preparation

Download the models and prepare the data:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages download prepare
```

Fit the annotation model and annotate the preference data:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages golden_rm audit_golden_rm golden_score_train golden_score_val annotate
```

### Reward Model Fitting

Fit the reward model and score the preference pairs:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages gemma_rm audit_gemma_rm proxy_score_train proxy_score_val
```

### Training

Train the SFT model:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages sft audit_sft
```

Train DP3O from the SFT checkpoint using the experiment configuration:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages dp3o audit_dp3o
```

### Evaluation

After training finishes, generate responses for SFT and DP3O:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages generate_sft generate_dp3o
```

Score the generated responses with the evaluation reward model:

```bash
bash scripts/run.sh --run-name experiment --resume \
  --stages score_sft score_dp3o
```

Generation and reward scoring are implemented in [scripts/infer.py](scripts/infer.py). Outputs are written to `evaluation/test/` inside the configured run directory.
