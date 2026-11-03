# Build the files of the public dataset release (Kaggle version 2) from the run outputs.
# Usage: python prepare_release.py      (writes release/; the dataset description is release/README.md)
# One CSV per run: the prompts exactly as sent, the raw and the parsed answer of every model, and the columns that say
# whether a prompt is analysed in the paper and, if not, why.
import json
import shutil
from pathlib import Path
import pandas as pd
from config.runs import RUNS

OUT = Path("release")
MODEL_FILES = {"qwen": "Qwen3-4B-Instruct-2507-Q5_K_M", "krikri": "llama-krikri-8b-instruct-q5_k_m",
               "gemma": "gemma-4-E2B-it-Q5_K_M"}
# released runs: file name -> run name (template04_* and repro are checks of the original run, kept in the code repository)
RELEASED_RUNS = {"main_run": "v2_chat", "original_run_v1": "v1", "prompt_format_pilot": "pilot_chat",
                 "name_swap": "name_swap_chat", "paraphrase": "paraphrase_chat", "option_order": "option_order_chat",
                 "sampling_seed1": "sampling_seed1_chat", "sampling_seed2": "sampling_seed2_chat",
                 "sampling_seed3": "sampling_seed3_chat"}
PERSON_COLUMNS = [f"{c}_{k}" for c in ["firstname", "noun_nom", "group"] for k in [1, 2, 3]]


def exclusion_reason(row, run_name):
    if row["template"] == 8:
        return "template 08 removed (ill-defined gold answer)"
    if run_name == "v1" and row["template"] == 4:
        return "template 04 sent with an unrendered placeholder"
    if row["duplicate_name"]:
        return "two people with the same name in different spellings"
    if row["debatable_gold"]:
        return "debatable gold answer (reported separately)"
    return ""


def run_table(run_name):
    parsed = pd.read_csv(Path(f"output/parsed_{run_name}.csv"))
    table = parsed[["id", "template", "family", "attribute"]].copy()
    # the prompt exactly as it was sent to the models (the same for the three models)
    sent = pd.read_csv(Path(f"output/llm_responses_{MODEL_FILES['qwen']}_{run_name}.csv")).set_index("id")["prompt"]
    table["prompt"] = table["id"].map(sent)
    table[PERSON_COLUMNS] = parsed[PERSON_COLUMNS]
    table["ground_truth"] = parsed["ground_truth"]  # empty for templates 01 and 13: no correct person
    table["debatable_gold"] = parsed["debatable_gold"]
    table["exclusion_reason"] = [exclusion_reason(row, run_name) for _, row in parsed.iterrows()]
    table["analysed"] = table["exclusion_reason"] == ""
    for model, file in MODEL_FILES.items():
        responses = pd.read_csv(Path(f"output/llm_responses_{file}_{run_name}.csv")).set_index("id")
        assert (responses["prompt"].reindex(table["id"]).values == table["prompt"].values).all(), (run_name, model)
        table[f"answer_raw_{model}"] = parsed[f"raw_answer_{model}"]
        table[f"answer_{model}"] = parsed[f"answer_{model}"]
        table[f"parse_{model}"] = parsed[f"match_{model}"]
        table[f"finish_reason_{model}"] = parsed[f"finish_reason_{model}"]
        table[f"inference_time_{model}"] = table["id"].map(responses["inference_time"]).round(3)
    return table


OUT.mkdir(exist_ok=True)
for file_name, run_name in RELEASED_RUNS.items():
    table = run_table(run_name)
    table.to_csv(OUT / f"{file_name}.csv", index=False, encoding="utf-8")
    print(f"{file_name}.csv: {len(table)} prompts, {int(table['analysed'].sum())} analysed")

# settings of every run, models and hardware
environment = json.load(open("results/environment.json", encoding="utf-8"))
runs = {file_name: {"run": run_name, **{k: v for k, v in RUNS[run_name].items() if k != "prompts"}}
        for file_name, run_name in RELEASED_RUNS.items()}
for settings in runs.values():
    settings["seed_per_prompt"] = "seed of the run x 10,000,000 + prompt id (used only when sampling)"
json.dump({"runs": runs, "common": {"max_tokens": 30, "repeat_penalty": 1.1, "top_k": 40, "system_prompt": None},
           "models": environment["models"], "hardware": {k: environment[k] for k in ["cpu", "ram_gb", "gpu", "gpu_selection"]},
           "software": environment["packages"]},
          open(OUT / "runs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
shutil.copy("results/parser_validation_v2_chat.csv", OUT / "parser_validation.csv")
print(f"Release files written to {OUT}/")
