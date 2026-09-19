# Data and annotation materials

- `samples/`: small Gemma and Llama labeled-continuation examples for inspecting the paper record schema.
- `processed/`: released task mapping, judge-validation cases and Llama summary. Large generated records and features are ignored by Git.
- `raw/`: downloaded datasets; generated locally by the paper pipeline.
- [`annotations/`](annotations/): blinded human-validation materials, analysis scripts and aggregate results.

The benchmark's own per-run baselines, interventions and manifest go to `runs/` by default. They are distinct from the historical paper data format. See [reproduction boundaries](../docs/paper-reproduction.md#reproduction-boundaries).
