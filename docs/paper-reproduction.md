# Reproducing the paper experiments

[Repository home](../README.md) · [Protocol spec](protocol.md) · [Paper](https://openreview.net/forum?id=TiZQnKDIHq) · [PDF](https://openreview.net/pdf?id=TiZQnKDIHq)

This guide documents the historical experiment scripts and released artifacts. Run commands from the repository root. These scripts use fixed model choices and historical defaults; the new `evaluate.py` is a separate, reusable entry point.

## Reproduction boundaries

The release contains source code, small result summaries, and annotation materials, not an exact replay bundle. The full generation records, model feature arrays, probe checkpoints, and returned human-label files are absent. Dependency versions are lower bounds rather than a frozen environment. Re-running generation or API judging can change results. In particular, inspect model/judge arguments and dataset selection before interpreting a run as a reproduction of a table.

The historical BBH downloader lists 23 tasks, while the released task mapping covers additional tasks; a default historical run should not be assumed to reconstruct the paper sample. The new benchmark explicitly enumerates available BBH configurations and records them. 

## Navigate by research question

The [experiment map](../experiments/README.md) links every research theme to its scripts, paper sections and available artifacts. `experiments/pipeline/` generates and labels the main continuations; the other directories test the paper's scientific claims. `results/` contains released snapshots, and `data/annotations/` contains both human studies.

Use the commands below from the repository root. Intermediate records are stored in `data/processed/`; released summaries are grouped by research topic under `results/`.

## Setup

**Hardware.** All model inference was run on a single NVIDIA H100 80 GB. Other GPU configurations need validation with appropriate batch and memory settings. The judge stage and all analysis/plotting scripts run on CPU.

**Software.** Python 3.10+, CUDA 12.x.

```bash
git clone https://github.com/r2m-ai/load-bearing-cot.git
cd load-bearing-cot
pip install -r requirements.txt
```

**Model access.** Gemma-2-9B-IT and Llama-3.1-8B-Instruct are gated on Hugging Face; accept the licenses and run `huggingface-cli login` once. DeepSeek-R1-Distill-Qwen-7B is ungated.

**LLM judge.** Borderline A/B cases are classified by Claude (`claude-haiku-4-5-20251001` in production, `claude-sonnet-4-20250514` for validation). Pass `--model claude-haiku-4-5-20251001` explicitly to `experiments/pipeline/judge_behaviors.py` to select the production model described in the paper; its code default is Sonnet. The judge scripts take the key as `--api-key <key>` (or `--api-key "$ANTHROPIC_API_KEY"`). This is the only external API call in the pipeline; everything else runs locally.

---

## Reproducing the main pipeline (Gemma-2-9B-IT)

Run from the repository root. Times are single-H100 estimates.

| Step | Command | Purpose | ≈ time |
|---|---|---|---|
| 0 | `python experiments/pipeline/download_datasets.py` | Download GSM8K and MMLU to `data/raw/` | 1 min |
| 1 | `python experiments/pipeline/generate_baselines.py` | Zero-shot CoT with Gemma-2-9B-IT (vLLM); keeps correct, ≥3-step baselines | 20 min |
| 1b | `python experiments/pipeline/generate_bbh_baselines.py --step all` | Same for BIG-Bench Hard (download + CoT) | 15 min |
| 2 | `python experiments/pipeline/perturb_and_continue.py` | Perturb one step (7 strategies × 3 positions), truncate, force-continue; writes `data/processed/all_continuation_pairs.json` | 45 min |
| 2b | `python experiments/pipeline/classify_behaviors.py` | Rule-based A/B/C labels; ambiguous rows are marked `NEEDS_JUDGE` | 1 min |
| 3 | `python experiments/validation/validate_judge.py --api-key "$ANTHROPIC_API_KEY"` then `python experiments/pipeline/judge_behaviors.py --api-key "$ANTHROPIC_API_KEY" --model claude-haiku-4-5-20251001` | Validate the LLM judge on known cases, then label every `NEEDS_JUDGE` row | 3 min |
| 3b | `python experiments/validation/fix_bbh_labels.py` then `python experiments/validation/propagate_labels.py` | BBH answer-format fix (see below) and propagation of corrected labels to downstream files | 5 min |
| 4 | `python experiments/probes/layer_sweep.py` | Extract hidden states at three token positions across all 42 layers and train linear + MLP probes (3-class, bypass, error-propagation) with grouped 5-fold CV | 25 min |
| 4b | `python experiments/probes/retrain_probes.py`, `python experiments/probes/compute_auroc.py` | Re-fit probes from saved features (no GPU) and compute AUROCs | 2 min |
| 5 | `python experiments/steering/single_direction.py` | Single-direction activation steering at layer 21 with dose–response, layer, cross-dataset and random-direction controls (11,556 generations) | 50 min |
| 6 | `python experiments/steering/prompt_steering.py` | Prompt-stage steering control | 10 min |
| 7 | `python experiments/behavior/analyze_behaviors.py`, `python experiments/probes/leakage_test.py` | Position gradient, strategy breakdown, feature-leakage check | 2 min |
| 8 | `python experiments/models/llama.py` | Full pipeline on Llama-3.1-8B-Instruct | 105 min |

The judge stage writes `expanded_pairs.json`. The BBH label-fix script requires `ANTHROPIC_API_KEY` in the environment.

**The BBH label fix (step 3b).** The original answer check compared BBH answers against an empty `correct_letter` field, which mislabeled 3,229 correct BBH continuations as Type C. `experiments/validation/fix_bbh_labels.py` re-scores them and sends the ambiguous rows to the judge; all numbers in the paper are post-fix.

**Classification prompts.** The rule is in `experiments/pipeline/classify_behaviors.py`; the production judge prompt and its validation prompt are in `experiments/pipeline/judge_behaviors.py` / `experiments/validation/validate_judge.py`; the four prompt variants used in the sensitivity analysis are in `experiments/validation/judge_prompt_variants.py`. All prompts are also printed in the paper's appendix.

---

## Controls, validation and extensions

These experiments test the difficulty interpretation, label reliability, model generality and controllability. Consult the named sections in the [experiment map](../experiments/README.md); `--stage` selects generate / judge / analyze where the script supports it.

| Script | Paper | What it does |
|---|---|---|
| `experiments/controls/gsm8k_text_perturbations.py` | Results: type × difficulty | Text-based perturbations on GSM8K (off-diagonal cell of the type × difficulty 2×2) |
| `experiments/controls/hard_task_numerical_perturbations.py` | Results: type × difficulty | Numerical perturbations on BBH multistep arithmetic and other hard tasks (the other off-diagonal cell) |
| `experiments/controls/perturbation_strength.py` | Results: type × difficulty; appendix | Perturbation-magnitude sweep (n+1, n×2, n×2+3) |
| `experiments/models/llama_matched_controls.py` | Results: cross-model validation; appendix | Llama replication of the two new arms |
| `experiments/controls/variance_decomposition.py` | Results: type × difficulty; appendix | Logistic GLM and sequential deviance partition on the combined n = 28,584 frame (reads the outputs of the runs above) |
| `experiments/validation/judge_prompt_variants.py` | Method: behavioral classification | Re-judge every non-C continuation under four prompt variants; per-row results in `results/validation/judge_prompt_variants/stratification.json` |
| `experiments/probes/extract_human_annotation_features.py`, `experiments/probes/probe_vs_human.py` | App. | Probe-vs-human agreement on the n = 200 pilot rows |
| `experiments/models/deepseek.py`, `experiments/models/judge_deepseek.py`, `experiments/models/deepseek_prompt_diagnostic.py` | Results: cross-model validation; appendix | DeepSeek-R1-Distill-Qwen-7B run (`</think>` parsing, ≥4 post-think steps) and prompt diagnostics |
| `experiments/models/base_gemma.py`, `experiments/models/base_gemma_fewshot.py` | App. | Base Gemma-2-9B viability gate, zero- and few-shot |
| `experiments/steering/multi_direction.py`, `experiments/figures/plot_multi_direction.py` | Results: causal intervention | k-direction steering (k ∈ {2,4,8}) on two basis constructions |
| `experiments/figures/plot_human_agreement.py` | App. | Human-agreement figure |

Plotting scripts for the main-text figures are `experiments/figures/plot_difficulty_gradient.py`, `experiments/figures/plot_accuracy_vs_errorprop.py`, `experiments/figures/plot_judge_sensitivity_full.py` and `experiments/figures/plot_deepseek_layer_sweep.py`. [`figures/README.md`](../figures/README.md) maps every shipped figure and table to its producing script.

---

## Human validation

Two blind human-annotation studies validate the behavioral labels (paper §3.3 and Appendix "Human Annotation Protocol").

| | n = 200 pilot | n = 500 study |
|---|---|---|
| Annotators | 1 | 2, independent |
| Files | `data/annotations/pilot_n200/` | `data/annotations/study_n500/` |
| Instructions given to annotators | `INSTRUCTIONS.md` (EN), `INSTRUCTIONS_zh.md` (ZH) | `for_annotators/annotator{1,2}/INSTRUCTIONS*.md` |
| Blinded labeling file (as sent, unlabeled) | `data_to_label.csv` | `for_annotators/annotator{1,2}/data_to_label_annotator{1,2}.csv` |
| Hidden metadata (stratum, rule / judge labels; never shown to annotators) | `_analysis_metadata.csv` | `analysis/_analysis_metadata.csv`, `analysis/sample_manifest.json` |
| Sampler / analysis | `analyze_when_done.py` | `analysis/build_sample_n500.py`, `analysis/analyze_when_done_n500.py` |
| Aggregate results | `results_n200.json` | `analysis/results_n500.json` |

The labeling files are released exactly as they were sent to the annotators (labels blank), together with the hidden metadata and the analysis scripts that join the two. The annotators' returned label files are not part of this repository; the aggregate numbers they produced are in the `results_*.json` files and in the paper. Given a returned file, the analysis is

```bash
python data/annotations/study_n500/analysis/analyze_when_done_n500.py \
    --annotator1 <returned annotator1 csv> --annotator2 <returned annotator2 csv>
```

Headline: inter-annotator κ = 0.93 on the four-way label and κ = 1.00 on the C-vs-non-C split; the human consensus agrees with the paper's labels on 96.2% of rows for C-vs-non-C and 76.0% for the full three-way label. `analysis/build_sample_n500.py` is the deterministic sampler (seed 20260829); it needs the regenerated `data/processed/expanded_pairs.json`.

---

## Data

The source datasets (GSM8K, MMLU, BIG-Bench Hard) are public and have download scripts. Exact numerical reproduction also depends on the historical sample, generation and judge configuration, and unavailable artifacts described above. Human agreement cannot be recomputed from the blank annotation sheets without the returned labels.

Included in this repository (small):

- `data/samples/gemma_sample_50.json`, `data/samples/llama_sample_50.json` — 50 labeled continuations each, showing labeled record examples from the paper pipeline (`prompt`, `original_step`, `perturbed_step`, `continuation`, `strategy`, `perturbation_point`, rule and judge labels).
- `data/processed/judge_validation.json` — the hand-labeled cases used to validate the judge.
- `data/processed/bbh_task_mapping.json`, `data/processed/llama_results_summary.json`.
- `results/**` — JSON/CSV summaries of the revision experiments, including the per-row four-variant judge labels.
- Both human-annotation studies: instructions, blinded labeling files, hidden metadata, scripts, and aggregate results. The annotators' returned label files are not released.

Not in this repository, because Git ignores the pipeline's generated output: the ~21k labeled continuations (`data/processed/expanded_pairs.json`, ~100 MB), the matched controls, the steering generations and the per-model raw CoT files. Rather than regenerate them, take the exact rows the paper used from the [Hugging Face dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness), which ships all of them as parquet. Hidden-state feature arrays and probe checkpoints are released nowhere and must be regenerated; contact the authors if you need the originals.

---

## Paper

```bash
cd paper
tectonic main.tex        # or: latexmk -pdf main.tex
```

`paper/figures` is a symlink to `../figures`; on systems without symlink support, copy `figures/` into `paper/`.

---

For the main findings and citation, see the [repository README](../README.md).
