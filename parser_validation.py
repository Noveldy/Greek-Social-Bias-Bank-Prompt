# Manual check of the parser on the answers that are not a single exact answer word.
# Usage: python parser_validation.py sample <run_name>
#            writes results/parser_validation_<run_name>.csv with all answers the parser could not read (none, refusal)
#            and a random sample of the other non-exact answers (extracted, phrase). Fill in the column "correct"
#            (y if the parsed answer is what the model answered, n otherwise) and "correct_answer" when it is n.
#        python parser_validation.py score <run_name>
#            agreement with the manual check, per match type, of the parser used for sampling and of the current
#            parser (rules added after the check) (results/parser_validation_summary_<run_name>.csv)
import sys
from pathlib import Path
import pandas as pd
from config.runs import RUNS
from results_analysis import MODELS

SAMPLE_SIZE = 300
MODE, RUN_NAME = sys.argv[1], sys.argv[2]
CHECK_FILE = Path(f"results/parser_validation_{RUN_NAME}.csv")

parsed = pd.read_csv(Path(f"output/parsed_{RUN_NAME}.csv"))
answers = pd.concat([
    pd.DataFrame({"id": parsed["id"], "model": model, "template": parsed["template"], "match": parsed[f"match_{model}"],
                  "raw_answer": parsed[f"raw_answer_{model}"], "parsed_answer": parsed[f"answer_{model}"]})
    for model in MODELS
])
non_exact = answers[answers["match"] != "exact"]

if MODE == "sample":
    if CHECK_FILE.exists():
        sys.exit(f"{CHECK_FILE} already exists, not overwriting the manual check")
    unreadable = non_exact[non_exact["match"].isin(["none", "refusal"])]
    readable = non_exact[non_exact["match"].isin(["extracted", "phrase"])]
    sample = pd.concat([unreadable, readable.sample(n=min(SAMPLE_SIZE, len(readable)), random_state=42)])
    sample["correct"] = ""
    sample["correct_answer"] = ""
    prompts = pd.read_csv(Path(RUNS[RUN_NAME]["prompts"])).set_index("id")["prompt"]
    sample["prompt"] = sample["id"].map(prompts)
    sample.to_csv(CHECK_FILE, index=False, encoding="utf-8-sig")  # with BOM, so that spreadsheet programs show Greek
    print(f"{len(sample)} answers to check written to {CHECK_FILE}")
    print(sample.groupby(["model", "match"]).size().to_string())
else:
    checked = pd.read_csv(CHECK_FILE, encoding="utf-8-sig")
    checked["correct"] = checked["correct"].astype(str).str.strip().str.lower()
    checked = checked[checked["correct"].isin(["y", "n"])]
    # manual answer: the parsed answer at sampling time if it was correct, otherwise the corrected one
    # (correct_answer is only read when correct is n)
    checked["manual_answer"] = checked["parsed_answer"].where(checked["correct"] == "y", checked["correct_answer"])
    current = answers[["id", "model", "parsed_answer"]].rename(columns={"parsed_answer": "current_answer"})
    checked = checked.merge(current, on=["id", "model"])
    checked["agree_at_sampling"] = checked["correct"] == "y"
    checked["agree_current"] = checked["manual_answer"] == checked["current_answer"]
    # match = how the parser read the answer at sampling time
    summary = checked.groupby("match").agg(checked=("id", "size"), agreement_at_sampling=("agree_at_sampling", "mean"),
                                           agreement_current_parser=("agree_current", "mean"))
    summary["answers_in_run"] = non_exact.groupby("match").size()
    summary.loc["all checked"] = [len(checked), checked["agree_at_sampling"].mean(), checked["agree_current"].mean(),
                                  len(non_exact)]
    disagreements = checked[~checked["agree_current"]]
    print(f"Disagreements with the current parser: {len(disagreements)}")
    print(disagreements[["id", "model", "template", "current_answer", "manual_answer", "raw_answer"]].to_string(index=False))
    print(f"Answers in run {RUN_NAME}: {len(answers)}, exact: {len(answers) - len(non_exact)}")
    print(summary.round(3).to_string())
    summary.to_csv(Path(f"results/parser_validation_summary_{RUN_NAME}.csv"))
