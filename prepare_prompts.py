# Prepare the prompt files used in the revision runs.
# - output/prompts_v1_1.csv: the v1 prompts with template 04 fixed. In v1, the 863 template 04 prompts were sent
#   to the models with a literal "{ article_acc_3 }}" instead of the article of the third person.
# - output/prompts_template04.csv: only the fixed template 04 prompts (re-run in the format chosen as main).
# - output/prompts_pilot.csv: stratified sample of v1 prompts used to compare the raw and the chat prompt format.
# - output/prompts_repro.csv: random sample of the pilot prompts, re-run in the v1 (raw) format to check that the
#   current build reproduces the v1 answers.
# - output/prompts_v2.csv: the v1.1 prompts (same ids) with one spelling per name (Ελένη, Εμινέ, Αλί) and the noun
#   πρόσφυγας spelled correctly, without the prompts in which two people share a name. Used for the main run.
# - output/prompts_pilot_v2.csv: the pilot prompts in their v2 spelling (robustness runs).
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd
from config.demographic_dimensions import ATTRIBUTE_MAPS

PILOT_FRACTION = 0.2       # share of prompts per template and attribute in the pilot
SMALL_ATTRIBUTES = ["age", "gender", "migration"]  # few prompts: included completely in the pilot
REPRO_SIZE = 300
# spellings of the v1 prompts that are fixed in v2
SPELLING_V2 = {"Έλενη": "Ελένη", "Έμινε": "Εμινέ", "Άλι": "Αλί", "προσφύγας": "πρόσφυγας"}


def normalize(text):
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()


NOUN_TO_ATTRIBUTE = {}
for attribute, attribute_map in ATTRIBUTE_MAPS.items():
    for noun in attribute_map:
        NOUN_TO_ATTRIBUTE[noun] = attribute

# accusative article of every (noun, first name), taken from the attribute files
# nouns and first names are compared without accents because some spellings changed after v1 (e.g. Άλι / Αλί)
ARTICLE_ACC = {}
for attribute_file in Path("config/attributes/").glob("*.json"):
    for entry in json.load(open(attribute_file, encoding="utf-8")):
        ARTICLE_ACC[(normalize(entry["noun_nom"]), normalize(entry["firstname"]))] = entry["article_acc"]

prompts_df = pd.read_csv(Path("output/prompts_v1.csv"))
prompts_df["template"] = prompts_df["template"].astype(int)
prompts_df["attribute"] = prompts_df["noun_nom_1"].map(NOUN_TO_ATTRIBUTE)

# fix template 04
broken = prompts_df["prompt"].str.contains("{ article_acc_3 }}", regex=False)
print(f"Template 04 prompts with the placeholder: {broken.sum()}")
for index, row in prompts_df[broken].iterrows():
    article = ARTICLE_ACC[(normalize(row["noun_nom_3"]), normalize(row["firstname_3"]))]
    prompts_df.at[index, "prompt"] = row["prompt"].replace("{ article_acc_3 }}", article)
assert not prompts_df["prompt"].str.contains("{", regex=False).any()
prompts_df.drop(columns=["attribute"]).to_csv(Path("output/prompts_v1_1.csv"), index=False, encoding="utf-8")
prompts_df[broken].drop(columns=["attribute"]).to_csv(Path("output/prompts_template04.csv"), index=False, encoding="utf-8")

# pilot sample: without template 04 and without prompts in which two people share a name (e.g. Αλί / Άλι)
duplicate_name = prompts_df.apply(
    lambda x: len({normalize(x["firstname_1"]), normalize(x["firstname_2"]), normalize(x["firstname_3"])}) < 3, axis=1
)
candidates = prompts_df[(prompts_df["template"] != 4) & (~duplicate_name)]
small = candidates[candidates["attribute"].isin(SMALL_ATTRIBUTES)]
large = candidates[~candidates["attribute"].isin(SMALL_ATTRIBUTES)]
sampled = large.groupby(["template", "attribute"]).sample(frac=PILOT_FRACTION, random_state=42)
pilot_df = pd.concat([small, sampled]).sample(frac=1, random_state=42)
pilot_df.drop(columns=["attribute"]).to_csv(Path("output/prompts_pilot.csv"), index=False, encoding="utf-8")
print(f"Pilot prompts: {len(pilot_df)}")
print(pilot_df["attribute"].value_counts())

repro_df = pilot_df.sample(n=REPRO_SIZE, random_state=42)
repro_df.drop(columns=["attribute"]).to_csv(Path("output/prompts_repro.csv"), index=False, encoding="utf-8")
print(f"Reproduction check prompts: {len(repro_df)}")

# v2: fixed spellings, without the prompts in which two people share a name
v2_df = prompts_df[~duplicate_name].drop(columns=["attribute"])
for old, new in SPELLING_V2.items():
    for column in ["prompt", "firstname_1", "firstname_2", "firstname_3", "noun_nom_1", "noun_nom_2", "noun_nom_3",
                   "ground_truth"]:
        v2_df[column] = v2_df[column].str.replace(rf"(?<!\w){old}(?!\w)", new, regex=True)
for old in SPELLING_V2:
    assert not v2_df["prompt"].str.contains(rf"(?<!\w){old}(?!\w)", regex=True).any()
v2_df.to_csv(Path("output/prompts_v2.csv"), index=False, encoding="utf-8")
print(f"v2 prompts: {len(v2_df)} ({duplicate_name.sum()} prompts with a shared name removed)")

pilot_v2_df = v2_df.set_index("id").loc[pilot_df["id"]].reset_index()
pilot_v2_df.to_csv(Path("output/prompts_pilot_v2.csv"), index=False, encoding="utf-8")
print(f"Pilot prompts in v2 spelling: {len(pilot_v2_df)}")
