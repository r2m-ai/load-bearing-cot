# Paper experiment map

[Paper](../paper/main.pdf) · [Repository home](../README.md) · [Reproduction commands](../docs/paper-reproduction.md)

The experiments are organized around the paper's argument. Start with the intervention protocol, establish the behavioral gradient and its controls, validate the labels, then examine generalization, representation and control. The reusable `evaluate.py` is a separate, simplified protocol; these scripts retain the paper's experimental logic.

| Research question | Code | Paper section / evidence | Released artifacts |
|---|---|---|---|
| Does an edited reasoning step change the answer? | [`pipeline/`](pipeline/): baseline generation → perturbation → classification → judging | Method: Continuation-Based Causal Testing; Three-Mode Behavioral Classification | [Example records](../data/samples/), [judge validation cases](../data/processed/judge_validation.json) |
| Does propagation track task difficulty and perturbation position? | [`behavior/`](behavior/), [`validation/binary_replication.py`](validation/binary_replication.py) | Results: Behavioral Taxonomy; Position Gradient | [Binary analysis](../results/robustness/exp1c_binary_replication.json), [behavior tables](../figures/table4_behavioral.csv) |
| Could perturbation type or strength explain the gradient? | [`controls/`](controls/): text-on-GSM8K, numerical-on-hard, magnitude sweep, variance decomposition | Results: Disentangling Perturbation Type from Task Difficulty; strength and variance appendices | [Variance decomposition](../results/controls/variance_decomposition/) |
| How reliable are the labels and the propagation interpretation? | [`validation/`](validation/): judge variants, binary replication, uptake, BBH label correction | Method: classification; Judge Validation and Human Annotation appendices | [Judge variants](../results/validation/judge_prompt_variants/), [uptake](../results/robustness/exp3b_perturbation_uptake.json), [BBH correction](../results/validation/bbh_label_correction/), [human studies](../data/annotations/) |
| Does the pattern generalize across models and prompting regimes? | [`models/`](models/): Llama, matched Llama controls, DeepSeek and prompt diagnostic, base Gemma and few-shot viability | Results: Cross-Model Validation; model-specific appendices | [Model snapshots](../results/models/), [Llama summary](../data/processed/llama_results_summary.json) |
| Can internal states predict the behavioral outcome? | [`probes/`](probes/): extraction, layer sweep, refitting, AUROC, leakage and human-label evaluation | Method: Hidden-State Probes; Results: Probe Results; robustness appendix | [Probe tables](../figures/table5_7_probe_with_ci.csv), [human-label comparison](../results/probes/human_agreement/) |
| Can those predictive directions change behavior? | [`steering/`](steering/): single direction, multiple directions, prompt control | Results: Causal Intervention: Detection ≠ Control | [Intervention table](../figures/table7_intervention.csv), [multi-direction figure](../figures/multidir_steering.png); full steering generations are not shipped |
| How are the paper figures assembled? | [`figures/`](figures/) | Figures in the main text and appendices | [`../figures/`](../figures/) stores the paper's rendered figures and CSV tables, mapped to their producing scripts in [`../figures/README.md`](../figures/README.md) |
| How is the public data release assembled? | [`release/`](release/): `build_hf_dataset.py` | — | [Hugging Face dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness), reshaped from the files under `data/` and `results/` |

## Main experimental dependencies

```text
pipeline/download_datasets.py + pipeline/generate_baselines.py
                           + pipeline/generate_bbh_baselines.py
                                      ↓
                       pipeline/perturb_and_continue.py
                                      ↓
            pipeline/classify_behaviors.py + pipeline/judge_behaviors.py
                                      ↓
       validation/fix_bbh_labels.py + validation/propagate_labels.py
                                      ↓
                      data/processed/expanded_pairs.json
                         ↙              ↓              ↘
             behavior + controls    validation       probes
                                        ↓               ↓
                              human annotation       steering

models/ runs the corresponding cross-model and format-viability studies.
```

The shared paper utilities are in `pipeline/perturbation_helpers.py`. Imports, repository-root resolution and plot inputs have been updated to the thematic layout. Generated data schemas and historical intermediate filenames are retained, so existing generated records remain usable.

## Reading the evidence

The paper's most controlled difficulty comparisons hold task format or perturbation type fixed. Raw dataset averages also reflect baseline selection and label quality. DeepSeek is a reasoning-trained contrast with post-thinking selection, while base Gemma is a format-viability study; neither should be described as an interchangeable replication of the main Gemma experiment.

Human annotation is part of label validation, not a separate revision phase. The 200-example pilot and 500-example two-annotator study live together under `data/annotations/`, with original forms and metadata. The returned labels are not included. Probe accuracies against the judge and probe agreement with humans are different evaluations.

`results/` contains released snapshots. Some experiment runners write working artifacts to `data/processed/`; a snapshot in `results/` is not necessarily the runner's output destination. Inspect script arguments and the reproduction guide before regenerating a figure. 

The README findings figure is generated by `figures/plot_readme_findings.py`, which reads the released variance-analysis snapshot for the MMLU subject spread and the matched cells, and checks the paper's reported C rates against it. [`../figures/README.md`](../figures/README.md) maps every shipped paper figure and table to its producing script. The final two-annotator figure uses `figures/plot_human_agreement.py` and requires returned-label data not included in the release.
