# Paper figures and table data

Rendered figures for `paper/main.tex` (which reaches them through the `paper/figures` symlink) and the CSV tables behind the reported numbers. These are paper artifacts; runs of `evaluate.py` write to `runs/` instead.

## Figures

| File | Paper | Produced by |
|---|---|---|
| `method_flow.png`, `method_detail.png` | Method: continuation-based causal testing | Hand-drawn diagrams; no script |
| `judge_sensitivity.png` | Method: three-mode behavioral classification | [`experiments/figures/plot_judge_sensitivity_full.py`](../experiments/figures/plot_judge_sensitivity_full.py) |
| `layer_sweep.png` | Results: probe results | [`experiments/probes/layer_sweep.py`](../experiments/probes/layer_sweep.py) |
| `steering.png` | Results: causal intervention | [`experiments/steering/single_direction.py`](../experiments/steering/single_direction.py) |
| `multidir_steering.png` | Results: causal intervention | [`experiments/figures/plot_multi_direction.py`](../experiments/figures/plot_multi_direction.py) |
| `difficulty_gradient_unified.png` | Results: cross-model validation | [`experiments/figures/plot_difficulty_gradient.py`](../experiments/figures/plot_difficulty_gradient.py) |
| `deepseek_layer_sweep.png` | Appendix: DeepSeek probes | [`experiments/figures/plot_deepseek_layer_sweep.py`](../experiments/figures/plot_deepseek_layer_sweep.py) |
| `human_agreement.png` | Appendix: human annotation | [`experiments/figures/plot_human_agreement.py`](../experiments/figures/plot_human_agreement.py) |
| `label_quality.png`, `robustness_panel.png` | Appendices: label quality, robustness | Assembled during the revision from the released `results/` snapshots; no single shipped script |

The README figures are separate and live in [`docs/assets/`](../docs/assets/).

## Tables

| File | Contents | Produced by |
|---|---|---|
| `table4_behavioral.csv` | A/B/C counts by position, strategy and dataset | [`experiments/behavior/analyze_behaviors.py`](../experiments/behavior/analyze_behaviors.py) |
| `table5_7_probe_with_ci.csv` | Probe accuracy with CIs at the selected layers | [`experiments/probes/retrain_probes.py`](../experiments/probes/retrain_probes.py) |
| `table6_probe_judged.csv` | Full 42-layer probe sweep on judge labels | [`experiments/probes/retrain_probes.py`](../experiments/probes/retrain_probes.py) |
| `table_auroc.csv` | AUROC at the best layer per probe task | [`experiments/probes/compute_auroc.py`](../experiments/probes/compute_auroc.py) |
| `table_llama_probes.csv` | Llama probe transfer | [`experiments/models/llama.py`](../experiments/models/llama.py) |
| `table7_intervention.csv` | Single-direction steering dose–response | [`experiments/steering/single_direction.py`](../experiments/steering/single_direction.py) |

Most producing scripts need the `data/processed/` records, which Git ignores; regenerate them, or take the paper's own rows from the [Hugging Face dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness). See the [reproduction guide](../docs/paper-reproduction.md#reproduction-boundaries). Some scripts also emit additional development figures that are not part of the release.
