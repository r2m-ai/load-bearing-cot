# Contributing

The benchmark is organized around a question: when does changing a written reasoning step change what a model answers?

Useful contributions include new model adapters, dataset/answer adapters, parser regression cases, and complete reproducible runs. Start with the [protocol](docs/protocol.md). Keep new benchmark code in `load_bearing/`; place paper experiments under the relevant research theme in `experiments/` and summaries under the corresponding theme in `results/`. Update the experiment map and any dependent paths when moving code.

For a model result, include the model and dataset revisions, protocol version, full arguments, hardware/software environment, eligibility and parsing counts, and the generated manifest and summary. Inspect example interventions before interpreting a score. Distinguish paper results, reproduced results, and newly measured results. Do not add leaderboard numbers from a smoke test or a run with unresolved parsing failures.

For code changes, run the offline suite:

```bash
python -m unittest discover -s tests -v
```

Tests require only the Python standard library. Add regression cases for answer formatting, skipped interventions and continuation boundaries when changing the protocol. Changes to sampling, prompting, step selection, perturbation or scoring require a protocol-version change and documentation of comparability.

Agent trajectories are an extension direction, not an implemented dataset. Proposals should define the intervention unit, unedited control, success metric and replay semantics before adding an adapter.
