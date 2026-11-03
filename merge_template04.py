# Build the v1.1 responses: the v1 responses in which the template 04 prompts (sent with an unrendered
# placeholder in v1) are replaced by their re-run with the fixed prompt in the same (raw) format.
# Usage: python merge_template04.py      (writes output/llm_responses_<model>_v1_1.csv)
import pandas as pd
from pathlib import Path

MODELS = ["Qwen3-4B-Instruct-2507-Q5_K_M", "llama-krikri-8b-instruct-q5_k_m", "gemma-4-E2B-it-Q5_K_M"]

for model_name in MODELS:
    v1_df = pd.read_csv(Path(f"output/llm_responses_{model_name}_v1.csv"))
    template04_df = pd.read_csv(Path(f"output/llm_responses_{model_name}_template04_raw.csv"))
    v1_df = v1_df[~v1_df["id"].isin(template04_df["id"])]
    merged_df = pd.concat([v1_df, template04_df]).sort_values("id")
    merged_df.to_csv(Path(f"output/llm_responses_{model_name}_v1_1.csv"), index=False, encoding="utf-8")
    print(f"{model_name}: {len(merged_df)} responses ({len(template04_df)} from the template 04 re-run)")
