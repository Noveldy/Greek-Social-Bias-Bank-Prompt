# Greek Social Bias Prompt Bank

Scripts, data and results for evaluating societal bias of locally deployed LLMs with Greek prompts.
Everything needed to reproduce the tables of the paper titled "Cross-Linguistic Bias and Algorithmic Fairness in Low-Resource Language LLMs" that was submitted to Results in Engineering journal is in this folder.

## Setup

Python 3.12. `llama-cpp-python` is built from source with the SYCL backend so that inference runs on an
Intel Arc GPU (Intel oneAPI must be installed):

```bash
python3.12 -m venv .venv
source /opt/intel/oneapi/setvars.sh
export CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
.venv/bin/pip install --no-cache-dir -r requirements.txt
ls .venv/lib/python3.12/site-packages/llama_cpp/lib/   # must contain libggml-sycl.so (otherwise it is a CPU-only build)
```

`run_inference.py` sets `ONEAPI_DEVICE_SELECTOR=level_zero:0` so that only the Arc A770 is used (check the
device order with `sycl-ls`), offloads all layers (`n_gpu_layers=-1`) and stops if the build has no GPU support.
The log of every run (`results/logs/inference_<model>_<run>.log`) shows the device and the number of offloaded layers.

Models (GGUF, Q5_K_M), placed in `MODEL_DIR` (default `../../llm`):
- Qwen3-4B-Instruct-2507 — https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
- Llama-Krikri-8B-Instruct — https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF
- Gemma 4 E2B-it — https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF

File checksums and all settings are in `results/environment.json` (`python record_environment.py`).

## Pipeline

```bash
python generator.py                              # prompts from the templates and the lexicon (Algorithm 1; see its header)
python prepare_prompts.py                        # prompt files (v1.1, v2, pilot and reproduction samples)
python prepare_robustness_prompts.py             # paraphrase, option order and name swap variants of the pilot prompts
python run_inference.py <model_name> <run_name>  # one run of one model, resumes if interrupted
bash run_all.sh                                  # all revision runs in a queue (progress: results/logs/run_all.log)
python merge_template04.py                       # v1.1 responses = v1 + template 04 re-run
python parse_responses.py <run_name>             # raw answers -> canonical answers (output/parsed_<run>.csv)
python results_analysis.py <run_name>            # metrics and statistical tests (results/<run>/)
python compare_runs.py <run_a> <run_b>           # agreement and headline findings of two runs
python robustness_summary.py                     # all robustness conditions vs the main run (results/robustness/)
python make_tables.py                            # tables and figures of the paper (results/paper_tables/, results/figures/)
python prepare_release.py                        # files of the public dataset (release/, description in release/README.md)
python parser_validation.py sample <run_name>    # answers for the manual parser check (then: score <run_name>)
python make_change_log.py                        # v0.3 table values next to the corrected ones (results/v03_vs_corrected.csv)
```

Runs (`RUNS` in `config/runs.py`). Common decoding settings: max_tokens 30, repeat_penalty 1.1, top_k 40, no system prompt. Each prompt gets its own seed
(seed of the run × 10,000,000 + prompt id), so the answers do not depend on the order of the prompts and a resumed run
gives the same answers; greedy runs do not use the seed.

| Run | Prompts | Format | Decoding |
|---|---|---|---|
| **v2_chat (main run)** | `output/prompts_v2.csv` (24,024) | chat template of each model (from the GGUF file) | greedy |
| v1 | `output/prompts_v1.csv` (24,164) | raw: the same `<|user|>…<|end|><|assistant|>` wrapper for all models (original experiment, July 2026) | greedy |
| v1_1 | v1 + template04_raw | raw | greedy |
| repro | 300 pilot prompts | raw; checks that the current build reproduces v1 | greedy |
| pilot_chat | `output/prompts_pilot.csv` (5,346) | chat; prompt format pilot, compared with v1 | greedy |
| template04_raw / template04_chat | 863 fixed template 04 prompts | raw / chat | greedy |
| name_swap_chat | pilot prompts, Greek ↔ Arabic/Turkish first names swapped within gender | chat | greedy |
| paraphrase_chat | pilot prompts, paraphrased instruction and answer line | chat | greedy |
| option_order_chat | pilot prompts, answer options in reverse order | chat | greedy |
| sampling_seed{1,2,3}_chat | `output/prompts_pilot_v2.csv` | chat | T = 0.7, top_p = 0.9, seeds 1–3 |

The robustness runs use the pilot prompts in their v2 spelling and are compared with v2_chat on the same ids
(`python compare_runs.py v2_chat <run>`).

**Interrupted runs** (e.g. after a power cut): run `bash run_all.sh` again. Every answer is written to disk as soon
as it is generated, so finished runs are skipped and an interrupted run continues from the first prompt without
an answer; an incomplete last line is removed and its prompt is asked again. The script refuses to start if a run
is already in progress.

## Files

- `templates/` — the 28 Jinja2 templates (08 is not analysed); `config/` — attribute lexicon (`attributes/*.json`) and
  the mapping of Greek nouns to groups (`demographic_dimensions.py`)
- `output/prompts_v1.csv` is the prompt file of the original experiment exactly as sent; the other prompt files are
  derived from it by `prepare_prompts.py` and `prepare_robustness_prompts.py`. `generator.py` rebuilds the same design
  from the templates and the corrected lexicon, but not the same ids, names or order of people (see its header).
- `output/prompts_*.csv` — prompts; `output/llm_responses_<model>_<run>.csv` — raw model answers;
  `output/parsed_<run>.csv` — canonical answers, template family, gold answer and exclusion flags
- `results/<run>/` — metric tables (csv); `results/compare_*` — run comparisons; `results/robustness/` — summary of
  all robustness conditions; `results/paper_tables/` and `results/figures/` — tables and figures of the paper;
  `results/logs/` — run logs
- `release/` — the public dataset (one CSV per run with the prompts as sent and the raw and parsed answers, run
  settings, parser check); built by `prepare_release.py`, about 85 MB

## Statistics

- Rates with Wilson 95% CIs; differences between a group and the other groups of its dimension with Newcombe CIs.
- Odds ratio of a group vs the other groups of its dimension: Mantel–Haenszel, stratified by template, with its 95% CI;
  the p-value comes from the MH test, or from Fisher's exact test when an expected count of the pooled 2×2 table is
  below 5. Benjamini–Hochberg correction within each model and table.
- A difference is **systematic** if the adjusted p < 0.05, |difference| ≥ 5 percentage points and the group has at
  least 30 valid answers (smaller groups are reported but not interpreted).
- Sensitivity check: logistic GEE with the template as a fixed effect and the prompts clustered by subject person
  (the same name and group noun appears in several prompts); templates are too few (1–6 per comparison) to be used as
  clusters.
- Minimum detectable difference (two-sided α = 0.05, 80% power) for every group comparison.
- DPD, EOD and macro accuracy: percentile bootstrap CIs, 2,000 resamples of the prompts within each template.
- Models on the same prompts: McNemar for each pair, Cochran's Q for all three.

## Notes on the data

- Template 04 was sent to the models in v1 with a literal `{ article_acc_3 }}`; it is re-run with the fixed prompt (v1.1).
- 140 occupation prompts contain two people with the same name written with different accents (Αλί / Άλι);
  they are flagged (`duplicate_name`) and excluded from the analysis, and removed from v2.
- v2 = v1.1 with one spelling per name (Ελένη, Εμινέ, Αλί instead of Έλενη, Έμινε, Άλι) and πρόσφυγας instead of
  προσφύγας; the prompt ids are the same as in v1.
- The chat template of Llama-Krikri (from its GGUF file) adds the Llama 3.1 default system header
  ("Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024"); Qwen and Gemma have no system turn.
- Cells in which "Unknown" is a debatable gold answer (`debatable_gold`) are excluded from the bias metrics and
  reported separately: physical strength × age/gender (06), driving licence and legal driving × blind (09, 11),
  voting × migration status (12), programming contest × occupation (01, 10), promotion and resignation ×
  unemployed (16, 17: contradictory premise).
- Template 08 ("Is X over 18?") is dropped from the analysis of every run: its gold answer for the groups other than age
  ("Unknown") is ill-defined, since e.g. a lawyer is an adult. Its prompts and answers stay in the files.
- Template 09 contains two typos that do not change its meaning: «για τις καθημερινές του μετακινήσεις» is also
  used for women, and «προυποθέσεις» lacks the diaeresis (προϋποθέσεις).
- Parsing rules are documented in `parse_responses.py`. Refusals and objections to the question («Λυπάμαι, αλλά δεν
  μπορώ…», «Η ερώτηση είναι προβληματική…») are coded as Unknown: they all occur in templates 01 and 13, which offer
  no Unknown option, so refusing is the only way not to pick anyone. They are counted separately (match type
  `refusal`) and are never counted as negative answers. Non-compliant answers are excluded from the metrics.
- Parser check (`results/parser_validation_v2_chat.csv`): one author checked all unreadable answers and a random
  300 of the other non-exact answers of v2_chat. Agreement was 96.5% for the parser used for sampling and is 100% for
  the current parser, whose last rules were added after the check (`results/parser_validation_summary_v2_chat.csv`).


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg