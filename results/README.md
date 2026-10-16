# Released research results

These files are paper result snapshots, organized by the scientific question they support. They are not results from the new `evaluate.py` entry point.

Each directory here matches the [`experiments/`](../experiments/) directory that produced it.

| Directory | Evidence |
|---|---|
| [`controls/`](controls/) | Task-difficulty × perturbation-type variance analysis |
| [`models/`](models/) | DeepSeek probing and prompt diagnostics, base-Gemma viability |
| [`probes/`](probes/) | Agreement of the bypass probe with pilot human labels |
| [`validation/`](validation/) | Judge-prompt variants, BBH answer-label correction, and the [`robustness/`](validation/robustness/) checks: binary replication, judge sensitivity, answer-change sensitivity, perturbation uptake |

File contents are the original snapshots, so some JSON keys still carry the identifiers the pipeline used when it wrote them. Hidden-state arrays are not shipped. Full continuation records and steering generations are not in this repository either, but they are released row-by-row in the [Hugging Face dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness). Generated steering outputs can be placed in `results/steering/` by the relevant workflow; no empty directory is presented as a released result.

See the [experiment map](../experiments/README.md) for producing scripts and paper references. A number audit requires the relevant underlying records; a summary JSON alone is not a full numerical reproduction.
