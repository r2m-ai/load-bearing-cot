# Human validation of behavioral labels

| Study | Purpose | Materials |
|---|---|---|
| [`pilot_n200/`](pilot_n200/) | Single-annotator pilot; also used for probe-vs-human evaluation | Instructions, blinded 200-row sheet, hidden metadata, analysis script and aggregate results |
| [`study_n500/`](study_n500/) | Main stratified study with two independent annotators | Sampling manifest, separate blinded sheets, metadata, sampler, agreement analysis and aggregate results |

The blinded sheets are the original blank materials sent to annotators. The annotators' returned label files are not released; the aggregate numbers they produced are in the `results_*.json` files and in the paper. If you rerun the exercise, keep the hidden metadata hidden while you do.

The larger study validates Gemma labels only. Its stratified sampling and production-label-favoring adjudication rule matter when interpreting agreement; see the paper's Human Annotation Protocol appendix. Code for extracting pilot hidden states and comparing probe predictions with human labels is in [`experiments/probes/`](../../experiments/probes/).
