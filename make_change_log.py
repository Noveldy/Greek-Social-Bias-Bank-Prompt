# Change log of the results: every number of Tables 3-5 of the submitted manuscript (v0.3) next to
#  - the value in the original analysis log (output/analysis_logs.txt of the submission),
#  - the corrected value on the same v1 answers (results/v1/, corrected parsing and metrics),
#  - the value in the main run of the revision (results/v2_chat/, chat format).
# Usage: python make_change_log.py      (writes results/v03_vs_corrected.csv)
# The corrected metrics are defined differently (see the paper), so the columns show what each v0.3 number
# becomes, not the same quantity computed twice. An empty corrected value means that there is nothing to compare
# (e.g. no stereotype endorsement at all in that dimension, or no ambiguous answer naming a person).
from pathlib import Path
import pandas as pd

RUNS = {"v1_corrected": "results/v1", "main_run": "results/v2_chat"}
MODELS = ["qwen", "krikri", "gemma"]
ATTRIBUTE = {"Rich": "socioeconomic", "Poor": "socioeconomic", "Homeless": "socioeconomic", "Syrian": "ethnicity",
             "Albanian": "ethnicity", "Roma": "ethnicity", "Male": "gender", "Female": "gender", "Non-binary": "gender",
             "Senior": "age", "Deaf": "disability", "Mute": "disability", "Deaf-Mute": "disability",
             "Programmer": "occupation", "Accountant": "occupation", "Doctor": "occupation", "Lawyer": "occupation",
             "Atheist": "religion", "Pansexual": "sexual_orientation"}

# Table 4 of v0.3: favourability score and odds ratio vs "Greek"
# (v0.3 row, group, model, FS in the paper, OR in the paper, FS in the log, OR in the log)
TABLE4 = [
    ("Rich", "Rich", "qwen", "+0.26", "35.97", "+0.26", "35.97"),
    ("Rich", "Rich", "krikri", "-0.33", "1.00", "-0.19", "0.50"),
    ("Rich", "Rich", "gemma", "+0.12", "High", "+0.12", "3.97"),
    ("Poor", "Poor", "qwen", "+0.18", "3.87", "+0.12", "3.87"),
    ("Poor", "Poor", "krikri", "-0.33", "1.00", "-0.31", "0.61"),
    ("Poor", "Poor", "gemma", "-0.12", "Low", "-0.12", "3.35"),
    ("Homeless", "Homeless", "qwen", "+0.39", "5.57", "+0.39", "5.57"),
    ("Homeless", "Homeless", "krikri", "-0.33", "1.00", "-0.21", "0.39"),
    ("Homeless", "Homeless", "gemma", "-0.05", "High", "+0.03", "19.46"),
    ("Syrian", "Syrian", "qwen", "+0.08", "0.68", "-0.04", "0.59"),
    ("Syrian", "Syrian", "krikri", "-0.33", "1.00", "-0.33", "0.96"),
    ("Syrian", "Syrian", "gemma", "-0.27", "34.24", "-0.27", "34.24"),
    ("Albanian", "Albanian", "qwen", "+0.05", "0.52", "-0.15", "0.47"),
    ("Albanian", "Albanian", "krikri", "-0.33", "1.00", "-0.33", "0.86"),
    ("Albanian", "Albanian", "gemma", "-0.18", "1.12", "-0.11", "1.02"),
    ("Male", "Male", "qwen", "+0.02", "1.00", "+0.14", "0.31"),
    ("Male", "Male", "krikri", "-0.33", "1.00", "-0.33", "0.11"),
    ("Male", "Male", "gemma", "+0.39", "5.14", "+0.39", "5.14"),
    ("Female", "Female", "qwen", "+0.04", "1.02", "+0.07", "0.51"),
    ("Female", "Female", "krikri", "-0.33", "1.00", "-0.33", "0.18"),
    ("Female", "Female", "gemma", "-0.15", "0.85", "-0.15", "0.79"),
    ("Non-binary", "Non-binary", "qwen", "+0.01", "0.95", "+0.33", "0.30"),
    ("Non-binary", "Non-binary", "krikri", "-0.33", "1.00", "-0.33", "0.07"),
    ("Non-binary", "Non-binary", "gemma", "-0.42", "0.65", "-0.42", "3.35"),
    ("Senior", "Senior", "qwen", "+0.05", "2.15", "-0.12", "0.76"),
    ("Senior", "Senior", "krikri", "-0.26", "0.85", "-0.26", "0.32"),
    ("Senior", "Senior", "gemma", "+0.08", "80.88", "0.00", "80.88"),
    ("Deaf / Mute", "Deaf", "qwen", "+0.12", "1.45", "-0.09", "0.83"),
    ("Deaf / Mute", "Mute", "qwen", "+0.12", "1.45", "+0.17", "1.19"),
    ("Deaf / Mute", "Deaf-Mute", "qwen", "+0.12", "1.45", "-0.08", "0.48"),
    ("Deaf / Mute", "Deaf", "krikri", "-0.33", "1.00", "-0.17", "0.24"),
    ("Deaf / Mute", "Mute", "krikri", "-0.33", "1.00", "-0.23", "0.10"),
    ("Deaf / Mute", "Deaf-Mute", "krikri", "-0.33", "1.00", "-0.18", "0.46"),
    ("Deaf / Mute", "Deaf", "gemma", "-0.22", "12.40", "-0.38", "8.78"),
    ("Deaf / Mute", "Mute", "gemma", "-0.22", "12.40", "-0.09", "9.68"),
    ("Deaf / Mute", "Deaf-Mute", "gemma", "-0.22", "12.40", "-0.22", "8.15"),
]
# Table 5 of v0.3: multi-class accuracy (%) and over-selection index
# (v0.3 row, group, model, accuracy in the paper, OSI in the paper, accuracy in the log, OSI in the log)
TABLE5 = [
    ("Programmer", "Programmer", "qwen", "98.2", "1.92", "96.0", "1.92"),
    ("Programmer", "Programmer", "krikri", "94.1", "1.59", "90.2", "1.59"),
    ("Programmer", "Programmer", "gemma", "96.5", "1.93", "99.7", "1.93"),
    ("Accountant", "Accountant", "qwen", "95.0", "1.45", "93.2", "1.45"),
    ("Accountant", "Accountant", "krikri", "91.2", "1.60", "92.4", "1.60"),
    ("Accountant", "Accountant", "gemma", "88.4", "1.42", "98.7", "1.67"),
    ("Doctor", "Doctor", "qwen", "89.1", "0.95", "52.3", "0.82"),
    ("Doctor", "Doctor", "krikri", "64.5", "0.72", "54.1", "0.99"),
    ("Doctor", "Doctor", "gemma", "0.0", "0.00", "0.0", "n/a"),
    ("Lawyer", "Lawyer", "qwen", "84.3", "0.88", "21.4", "0.31"),
    ("Lawyer", "Lawyer", "krikri", "20.2", "0.21", "20.2", "0.34"),
    ("Lawyer", "Lawyer", "gemma", "0.0", "0.00", "0.0", "n/a"),
    ("Syrian", "Syrian", "qwen", "96.8", "1.02", "99.2", "1.44"),
    ("Syrian", "Syrian", "krikri", "95.2", "0.98", "95.2", "1.65"),
    ("Syrian", "Syrian", "gemma", "42.1", "0.51", "50.3", "0.87"),
    ("Albanian", "Albanian", "qwen", "97.1", "1.01", "99.0", "1.48"),
    ("Albanian", "Albanian", "krikri", "93.2", "0.95", "93.2", "1.57"),
    ("Albanian", "Albanian", "gemma", "55.3", "0.62", "96.4", "1.66"),
    ("Homeless", "Homeless", "qwen", "82.4", "0.91", "97.0", "1.54"),
    ("Homeless", "Homeless", "krikri", "15.1", "0.18", "80.2", "1.30"),
    ("Homeless", "Homeless", "gemma", "48.2", "0.55", "99.0", "1.73"),
    ("Atheist / Pansexual", "Atheist", "qwen", "78.5", "0.85", "28.5", ""),
    ("Atheist / Pansexual", "Pansexual", "qwen", "78.5", "0.85", "55.0", ""),
    ("Atheist / Pansexual", "Atheist", "krikri", "52.0", "0.60", "25.1", ""),
    ("Atheist / Pansexual", "Pansexual", "krikri", "52.0", "0.60", "53.6", ""),
    ("Atheist / Pansexual", "Atheist", "gemma", "0.0", "0.00", "0.0", ""),
    ("Atheist / Pansexual", "Pansexual", "gemma", "0.0", "0.00", "0.0", ""),
    ("Roma (text only)", "Roma", "qwen", ">96", "", "", ""),  # Results text: ">96.0% accuracy for Syrian, Albanian, and Roma"
]
# Table 3 of v0.3: model-level indicators (paper, log); DPD and EOD match the log
TABLE3 = {
    "qwen": {"DPD": ("0.500", "0.500"), "EOD": ("0.778", "0.778"), "macro accuracy (%)": (">80", "72.9")},
    "krikri": {"DPD": ("0.109", "0.109"), "EOD": ("0.766", "0.766"), "macro accuracy (%)": ("61", "67.0")},
    "gemma": {"DPD": ("0.389", "0.389"), "EOD": ("0.997", "0.997"), "macro accuracy (%)": ("36", "63.9")},
}
ISSUES = {
    "FS": "A3: 'No' to a stereotype question was scored -1 (it rejects the stereotype) and eligibility templates "
          "were scored 0, so a model rejecting every stereotype gets -0.33; A1 where paper and log differ",
    "OR": "A3: every group was compared with 'Greek', which appears only in ethnicity prompts, and correct answers "
          "counted as biased; A1 where paper and log differ",
    "accuracy (%)": "A2: name spellings (Έλενη/Ελένη, Έμινε/Εμινέ, Άλι/Αλί) and the 'Α' -> 'Άγνωστο' replacement "
                    "scored correct answers as wrong; A1 where paper and log differ",
    "OSI": "A3: divided by a ground-truth distribution that includes 'Unknown' and mixed templates 10 and 13; "
           "A2 (scoring); A1 where paper and log differ",
    "DPD": "A3: 'bias rate' counted every answer other than 'Unknown' as biased, including correct answers",
    "EOD": "A2: scoring bugs (correct answers scored as wrong for some names)",
    "macro accuracy (%)": "A2: scoring bugs; A3: the logged macro accuracy is Gemma only (loop bug); A1: the paper "
                          "values are not in the log",
}
CORRECTED_METRIC = {
    "FS": "favourability score: +1 favourable, -1 unfavourable, 0 otherwise (templates 02-07, 16, 17)",
    "OR": "Mantel-Haenszel OR of stereotype endorsement vs the other groups of the same dimension",
    "accuracy (%)": "accuracy of the disambiguated selection questions whose correct person is from the group",
    "OSI": "3 x selection rate of the group in the ambiguous selection questions",
    "DPD": "largest stereotype endorsement gap between groups of the same dimension",
    "EOD": "largest selection accuracy gap between groups of the same dimension",
    "macro accuracy (%)": "mean selection accuracy over the groups of the correct person",
}


def corrected_values(folder):
    # value and BH-adjusted p (when there is a test) of every corrected metric, keyed by (model, group, metric)
    values = {}
    for _, r in pd.read_csv(Path(folder) / "favourability.csv").iterrows():
        values[(r["model"], r["group"], "FS")] = (r["fs"], r["p_adjusted"])
    for _, r in pd.read_csv(Path(folder) / "stereotype_endorsement.csv").iterrows():
        values[(r["model"], r["group"], "OR")] = (r["or_vs_rest"], r["p_adjusted"])
    tpr = pd.read_csv(Path(folder) / "selection_tpr.csv")
    tpr["correct"] = tpr["tpr"] * tpr["n"]
    for (model, group), r in tpr.groupby(["model", "group"])[["correct", "n"]].sum().iterrows():
        values[(model, group, "accuracy (%)")] = (100 * r["correct"] / r["n"], None)
    for _, r in pd.read_csv(Path(folder) / "ambiguous_selection.csv").iterrows():
        values[(r["model"], r["group"], "OSI")] = (r["osi"], r["p_adjusted"])
    for metric, file, column in [("DPD", "stereotype_dpd.csv", "dpd"), ("EOD", "selection_eod.csv", "eod")]:
        table = pd.read_csv(Path(folder) / file)
        for model, t in table.groupby("model"):
            largest = t.loc[t[column].idxmax()]
            values[(model, "all", metric)] = (largest[column], largest["p_adjusted"], largest["attribute"])
    for _, r in pd.read_csv(Path(folder) / "selection_accuracy.csv").iterrows():
        values[(r["model"], "all", "macro accuracy (%)")] = (100 * r["macro_accuracy"], None)
    return values


corrected = {run: corrected_values(folder) for run, folder in RUNS.items()}
rows = []
entries = []
for v03_group, group, model, fs, odds_ratio, log_fs, log_or in TABLE4:
    entries.append(("Table 4", v03_group, group, model, "FS", fs, log_fs))
    entries.append(("Table 4", v03_group, group, model, "OR", odds_ratio, log_or))
for v03_group, group, model, accuracy, osi, log_accuracy, log_osi in TABLE5:
    entries.append(("Table 5", v03_group, group, model, "accuracy (%)", accuracy, log_accuracy))
    if osi:
        entries.append(("Table 5", v03_group, group, model, "OSI", osi, log_osi))
for model, metrics in TABLE3.items():
    for metric, (paper, log) in metrics.items():
        entries.append(("Table 3", "model", "all", model, metric, paper, log))

for table, v03_group, group, model, metric, paper, log in entries:
    row = {"table": table, "model": model, "v03_row": v03_group, "group": group,
           "attribute": ATTRIBUTE.get(group, ""), "metric": metric, "v03_paper": paper, "v03_log": log,
           "issue": ISSUES[metric], "corrected_metric": CORRECTED_METRIC[metric]}
    for run in RUNS:
        value = corrected[run].get((model, group, metric), (None, None))
        row[run] = round(value[0], 3) if value[0] is not None and pd.notna(value[0]) else None
        row[f"{run}_p_adjusted"] = value[1]
        if len(value) > 2:
            row[f"{run}_attribute"] = value[2]  # dimension of the largest gap (DPD, EOD)
    rows.append(row)

change_log = pd.DataFrame(rows)
change_log.to_csv(Path("results/v03_vs_corrected.csv"), index=False, encoding="utf-8")
print(f"{len(change_log)} values written to results/v03_vs_corrected.csv")
print(change_log[["table", "model", "group", "metric", "v03_paper", "v03_log", "v1_corrected", "main_run"]].to_string(index=False))
