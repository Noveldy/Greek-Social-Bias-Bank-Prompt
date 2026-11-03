# Greek Social Bias Prompt Bank — version 2

A Greek-language benchmark for evaluating social bias in large language models (LLMs), with the answers of three
locally run, quantized models: Qwen3-4B-Instruct-2507, Llama-Krikri-8B-Instruct and Gemma 4 E2B-it (GGUF Q5_K_M,
llama.cpp). It accompanies the paper Cross-Linguistic Bias and Algorithmic Fairness in Low-Resource Language LLMs that was submitted on the Results in Engineering journal. The code that generates the prompts, runs the models and
produces every table of the paper is at https://github.com/Noveldy/Greek-Social-Bias-Bank-Prompt.

## The benchmark

- 27 templates in five task families: stereotype attribution (02–07), eligibility (09, 11, 12), decisions with a known
  answer (16 promotion, 17 resignation), ambiguous selection (01, 10, 13) and disambiguated selection (14, 15, 18–28).
- 40 social groups in nine dimensions: age, disability, ethnicity, gender, migration status, occupation, religion,
  sexual orientation and socioeconomic status. Every prompt describes three people of the same dimension, each with a
  first name and a group noun inflected for gender and case.
- 858 prompts per template, 23,166 prompts in the benchmark. The files also contain the 858 prompts of template 08,
  which was removed from the benchmark (see the changelog); they are marked as not analysed.

## Files

| File | Run | Prompts |
|---|---|---|
| `main_run.csv` | main run: each model's own chat template, greedy decoding | 24,024 |
| `original_run_v1.csv` | original experiment (July 2026): one generic `<\|user\|>…<\|end\|><\|assistant\|>` wrapper for all models, greedy decoding | 24,164 |
| `prompt_format_pilot.csv` | the original prompts of a stratified subset in the chat format (used to decide the main run's format) | 5,346 |
| `name_swap.csv` | subset with Greek and Arabic/Turkish first names swapped within gender | 5,346 |
| `paraphrase.csv` | subset with a paraphrased instruction and answer line | 5,346 |
| `option_order.csv` | subset with the answer options in reverse order | 5,346 |
| `sampling_seed1.csv`, `sampling_seed2.csv`, `sampling_seed3.csv` | subset, temperature 0.7, top-p 0.9, top-k 40, three seeds | 5,346 each |
| `runs.json` | settings of every run, model files with SHA-256 checksums, hardware and software | |
| `parser_validation.csv` | manual check of 344 parsed answers of the main run | |

The subset is 20% of the prompts of every template and dimension (all prompts of age, gender and migration status),
drawn from the original prompts; template 04 is not in it.

## Columns of the run files

| Column | Content |
|---|---|
| `id` | prompt identifier (the same prompt has the same id in every file) |
| `template`, `family`, `attribute` | template number, task family, dimension |
| `prompt` | the prompt exactly as it was sent to the models (before the chat template) |
| `firstname_k`, `noun_nom_k`, `group_k` | first name, Greek group noun and group (English) of person k = 1, 2, 3 |
| `ground_truth` | expected answer: «Ναι», «Όχι», «Άγνωστο» or the first name of the person described; empty for templates 01 and 13, which ask for a choice without an "Unknown" option and have no correct person |
| `debatable_gold` | the template × dimension cell has a debatable expected answer (e.g. a blind person and a driving licence) |
| `analysed`, `exclusion_reason` | whether the prompt is used in the bias metrics of the paper, and why not |
| `answer_raw_<model>` | the model's answer as generated (new lines removed) |
| `answer_<model>` | canonical answer: «Ναι», «Όχι», «Άγνωστο», one of the three first names, or NONCOMPLIANT |
| `parse_<model>` | how the answer was read: exact, extracted (from a longer answer), phrase ("cannot be determined"), refusal (coded «Άγνωστο»), none (non-compliant) |
| `finish_reason_<model>` | stop (the model ended its answer) or length (30-token limit) |
| `inference_time_<model>` | seconds |

`<model>` is `qwen`, `krikri` or `gemma`.

## Changes from version 1

1. **Raw answers.** The answer columns of version 1 (`final_answer_*`) had been processed by a cleaning step that
   removed punctuation and normalized the spelling of names; for Gemma, a character replacement turned 1,456 correct
   «Αλί» answers into «Άγνωστο». Version 2 gives the answers as generated and, separately, the canonical answer of a
   documented parser.
2. **Template 04.** In the original experiment, template 04 was sent with an unrendered placeholder
   (`{ article_acc_3 }}`). The version 1 file showed the corrected prompt text next to answers given to the broken
   prompt. In version 2, `original_run_v1.csv` shows the prompts exactly as sent and marks template 04 as not analysed;
   the main run used the corrected template.
3. **Duplicate names.** In 140 occupation prompts, two people had the same name written with different accents
   (e.g. Αλί and Άλι). They are marked in `original_run_v1.csv` and removed from the main run.
4. **Spelling.** The main run uses one spelling for every word (Ελένη, Εμινέ, Αλί, πρόσφυγας).
5. **Template 08 removed.** Template 08 ("Is X over 18?") expected «Άγνωστο» for every group other than age, although
   e.g. a lawyer is an adult. Its prompts remain in the files, marked as not analysed.
6. **Expected answers.** For templates 01 and 13 the expected answer was «Άγνωστο», although these prompts do not offer
   it; it is now empty. Seven template × dimension cells with a debatable expected answer are flagged.
7. **New runs.** The main run uses each model's own chat template instead of the generic wrapper of the original
   experiment, and six robustness runs were added.

Version 1 remains available as the record of the submitted manuscript.