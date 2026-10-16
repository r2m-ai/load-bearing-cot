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
| `behavioral_taxonomy.csv` | A/B/C counts by position, strategy and dataset | [`experiments/behavior/analyze_behaviors.py`](../experiments/behavior/analyze_behaviors.py) |
| `probe_with_ci.csv` | Probe accuracy with CIs at the selected layers | [`experiments/probes/retrain_probes.py`](../experiments/probes/retrain_probes.py) |
| `probe_layer_sweep.csv` | Full 42-layer probe sweep on judge labels | [`experiments/probes/retrain_probes.py`](../experiments/probes/retrain_probes.py) |
| `probe_auroc.csv` | AUROC at the best layer per probe task | [`experiments/probes/compute_auroc.py`](../experiments/probes/compute_auroc.py) |
| `llama_probe_transfer.csv` | Llama probe transfer | [`experiments/models/llama.py`](../experiments/models/llama.py) |
| `steering_dose_response.csv` | Single-direction steering dose–response | [`experiments/steering/single_direction.py`](../experiments/steering/single_direction.py) |

### Probe CSVs: label set and fit

`probe_with_ci.csv`, `probe_layer_sweep.csv` and `probe_auroc.csv` are all fitted on the same LLM-judge labels (`label_3class` / `subtype` from `data/processed/subclassified_pairs.json`, grouped 5-fold CV over question IDs, n = 2,000); no rule-label or human-label fit is shipped. They are three separate fits, so cells differ by under 1pp where fold assignment and MLP initialization differ (3-class MLP at layer 23: 74.6% in the selected-layer refit, 75.3% in the sweep). The README and the paper quote the sweep (`probe_layer_sweep.csv`) and the AUROC file; the two CSVs now carry `label_set` and `fit` columns, added to the shipped snapshots, so each file says this on its face. `tests/test_release.py::NumberAuditTests` pins the quoted README numbers to these files.

Most producing scripts need the `data/processed/` records, which Git ignores; regenerate them, or take the paper's own rows from the [Hugging Face dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness). See the [reproduction guide](../docs/paper-reproduction.md#reproduction-boundaries). Some scripts also emit additional development figures that are not part of the release.
