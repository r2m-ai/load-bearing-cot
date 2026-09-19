# Human validation, n = 500, two annotators

Blind two-annotator validation of the behavioral labels (paper §3.3 and the
"Human Annotation Protocol" appendix). Companion to the n = 200 single-annotator
pilot in `../pilot_n200/`.

```
for_annotators/annotator1/   INSTRUCTIONS.md / INSTRUCTIONS_zh.md   labeling guide given to annotator 1
                             data_to_label_annotator1.csv           the 500 rows exactly as sent (labels blank)
for_annotators/annotator2/   same rows, independently shuffled, as sent to annotator 2
analysis/build_sample_n500.py           deterministic stratified sampler (seed 20260829)
analysis/sample_manifest.json           stratum counts of the drawn sample
analysis/_analysis_metadata.csv         hidden per-row metadata: stratum, paper label, rule/judge label,
                                        four-variant judge labels (never shown to annotators)
analysis/analyze_when_done_n500.py      Cohen's kappa + consensus-vs-paper agreement report
analysis/results_n500.json              aggregate output of the analysis script on the returned files
```

Annotators saw only `prompt`, `original_step`, `perturbed_step`, `continuation`,
the reference answer, and the parser's `extracted_answer`; the stratum, the
rule label, and the judge labels were hidden. The two files contain the same
500 rows in different orders and were labeled independently.

The annotators' returned files are not included here; `results_n500.json`
holds the aggregate numbers reported in the paper. Given returned files, run
from the repository root:

```bash
python data/annotations/study_n500/analysis/analyze_when_done_n500.py \
    --annotator1 <returned annotator1 csv> --annotator2 <returned annotator2 csv> \
    [--adjudicated <csv with annotation_id,consensus_label for the disagreed rows>]
```
