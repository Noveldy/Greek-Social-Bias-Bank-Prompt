# Robustness of the main run (v2_chat) to the prompt format (v1, raw), the first names (name swap), the wording
# (paraphrase), the order of the answer options and sampling (T = 0.7, three seeds).
# Usage: python robustness_summary.py      (after compare_runs.py for every condition; writes results/robustness/)
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from results_analysis import MODELS, load_parsed

# condition -> (folder written by compare_runs.py, run name of the condition)
CONDITIONS = {
    "prompt format (v1, raw)": ("compare_v1_v2_chat", "v1"),
    "name swap": ("compare_v2_chat_name_swap_chat", "name_swap_chat"),
    "paraphrase": ("compare_v2_chat_paraphrase_chat", "paraphrase_chat"),
    "option order": ("compare_v2_chat_option_order_chat", "option_order_chat"),
    "sampling seed 1": ("compare_v2_chat_sampling_seed1_chat", "sampling_seed1_chat"),
    "sampling seed 2": ("compare_v2_chat_sampling_seed2_chat", "sampling_seed2_chat"),
    "sampling seed 3": ("compare_v2_chat_sampling_seed3_chat", "sampling_seed3_chat"),
}
# runs answered on the same prompts, for the agreement of all of them at once
RUN_SETS = {"prompt variants": ["v2_chat", "name_swap_chat", "paraphrase_chat", "option_order_chat"],
            "sampling seeds": ["sampling_seed1_chat", "sampling_seed2_chat", "sampling_seed3_chat"]}

out = Path("results/robustness")
out.mkdir(parents=True, exist_ok=True)

# 1. item-level agreement of each condition with the main run (Cohen's kappa)
agreement = pd.concat([pd.read_csv(f"results/{folder}/agreement.csv").query("family == 'all'").assign(condition=name)
                       for name, (folder, _) in CONDITIONS.items()])
agreement = agreement[["condition", "model", "n", "agreement", "kappa"]]
agreement.to_csv(out / "agreement.csv", index=False)
print(agreement.pivot(index="condition", columns="model", values="agreement")[MODELS].round(3).to_string())

# 2. headline findings: does each one hold (same value) in every condition?
verdicts = pd.DataFrame({name: pd.read_csv(f"results/{folder}/headline_findings.csv").set_index("finding")["holds"]
                         for name, (folder, _) in CONDITIONS.items()})
verdicts.insert(0, "v2_chat", pd.read_csv(f"results/{CONDITIONS['name swap'][0]}/headline_findings.csv").set_index("finding")["v2_chat"])
verdicts["verdict"] = [("holds in all conditions" if row.all() else "changes in: " + ", ".join(row.index[~row]))
                       for _, row in verdicts[list(CONDITIONS)].iterrows()]
verdicts.to_csv(out / "headline_findings.csv")
print(verdicts[["v2_chat", "verdict"]].to_string())

# 3. group-level systematic differences of the main run (on the prompts of each comparison):
#    same direction, and still systematic in the same direction, under each condition?
#    (the robustness runs use the 5,346 pilot prompts, so most groups have fewer than 30 answers there)
rows = []
for name, (folder, run) in CONDITIONS.items():
    groups = pd.read_csv(f"results/{folder}/group_odds_ratios.csv")
    groups = groups[groups["systematic_v2_chat"]]
    for (table, model), g in groups.groupby(["table", "model"]):
        kept = g[f"systematic_{run}"] & g["same_direction"]
        rows.append({"condition": name, "table": table, "model": model, "systematic_in_v2_chat": len(g),
                     "same_direction": int(g["same_direction"].sum()), "kept": int(kept.sum()), "share_kept": kept.mean()})
kept = pd.DataFrame(rows)
kept.to_csv(out / "systematic_differences_kept.csv", index=False)
print(kept.groupby(["condition", "model"])[["systematic_in_v2_chat", "same_direction", "kept"]].sum().unstack().to_string())


# 4. agreement of all the runs of a set at once: share of prompts with the same answer in every run, and Fleiss' kappa
def answer_label(data, model):
    # names are replaced by the position of the person (1, 2, 3) so that labels mean the same in every prompt
    labels = data[f"answer_{model}"].copy()
    for k in [1, 2, 3]:
        labels[data[f"answer_{model}"] == data[f"firstname_{k}"]] = f"person_{k}"
    return labels


rows = []
for set_name, runs in RUN_SETS.items():
    data = {run: load_parsed(run) for run in runs}
    ids = set.intersection(*[set(d.loc[~d["debatable_gold"], "id"]) for d in data.values()])
    data = {run: d[d["id"].isin(ids)].sort_values("id").reset_index(drop=True) for run, d in data.items()}
    for model in MODELS:
        labels = np.column_stack([answer_label(d, model).values for d in data.values()])
        rows.append({"set": set_name, "runs": " ".join(runs), "model": model, "n": len(labels),
                     "same_answer_in_all": (labels == labels[:, [0]]).all(axis=1).mean(),
                     "fleiss_kappa": fleiss_kappa(aggregate_raters(labels)[0])})
fleiss = pd.DataFrame(rows)
fleiss.to_csv(out / "fleiss_kappa.csv", index=False)
print(fleiss.round(3).to_string(index=False))
