# Variance decomposition (item 1.5)

n = 28,584

## Cell summary (C rate by task × type)

| Task | Type | n | C rate | 95% Wilson CI |
|---|---|---|---|---|
| bbh_multistep_arith | numerical | 939 | 64.5% | [61.4%, 67.5%] |
| bbh_other | numerical | 542 | 18.1% | [15.1%, 21.5%] |
| bbh_other | text | 8460 | 40.9% | [39.9%, 41.9%] |
| gsm8k | numerical | 2189 | 3.9% | [3.2%, 4.8%] |
| gsm8k | text | 5400 | 6.5% | [5.9%, 7.2%] |
| mmlu | numerical | 465 | 20.2% | [16.8%, 24.1%] |
| mmlu | text | 10589 | 22.3% | [21.5%, 23.1%] |

## Sequential deviance partition

Null deviance: 31939.4
Full-model deviance (with interaction): 28329.2
Total deviance explained: 3610.2

| Component | Δ deviance | df | p-value | fraction of total |
|---|---|---|---|---|
| type | 29.40 | 1 | 5.90e-08 | 0.8% |
| difficulty | 3567.02 | 1 | 0.00e+00 | 98.8% |
| interaction | 13.78 | 1 | 2.05e-04 | 0.4% |

## Question-clustered sensitivity (per-base_question aggregated rates, weighted GLM)

Aggregated to 7,104 unique base questions (8,092 (question × type × difficulty) cells).

| Term | Coef | SE | p-value | 95% CI |
|---|---|---|---|---|
| Intercept | -2.400 | 0.036 | 0.00e+00 | [-2.470, -2.329] |
| pt_num | -0.797 | 0.101 | 2.68e-15 | [-0.995, -0.600] |
| difficulty | 1.039 | 0.023 | 0.00e+00 | [0.994, 1.085] |
| pt_num:difficulty | 0.171 | 0.047 | 2.64e-04 | [0.079, 0.263] |

## Within-MMLU subject-difficulty contrast

Logistic slope on subject rank: β = 0.069 [0.063, 0.076], p = 1.94e-102

Per-subject C rates:

| Subject | n | C rate |
|---|---|---|
| high_school_psychology | 321 | 7.5% |
| high_school_government_and_politics | 543 | 8.8% |
| high_school_geography | 519 | 9.8% |
| high_school_european_history | 402 | 12.9% |
| high_school_biology | 840 | 13.9% |
| college_biology | 378 | 14.8% |
| high_school_computer_science | 240 | 16.7% |
| computer_security | 234 | 17.5% |
| clinical_knowledge | 603 | 18.4% |
| elementary_mathematics | 732 | 21.2% |
| college_medicine | 378 | 21.4% |
| astronomy | 357 | 22.1% |
| business_ethics | 198 | 22.2% |
| high_school_microeconomics | 582 | 22.7% |
| high_school_macroeconomics | 942 | 23.4% |
| anatomy | 288 | 25.0% |
| college_chemistry | 156 | 25.6% |
| electrical_engineering | 264 | 25.8% |
| conceptual_physics | 492 | 27.0% |
| high_school_mathematics | 729 | 28.0% |
| high_school_chemistry | 438 | 30.6% |
| econometrics | 200 | 34.0% |
| high_school_physics | 267 | 35.2% |
| formal_logic | 216 | 36.6% |
| college_physics | 204 | 39.2% |
| abstract_algebra | 161 | 39.8% |
| college_computer_science | 183 | 42.1% |
| college_mathematics | 133 | 43.6% |
| global_facts | 54 | 53.7% |
