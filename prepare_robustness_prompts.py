# Prepare the prompt variants of the robustness experiments (the pilot prompts in their v2 spelling).
# - output/prompts_paraphrase.csv: paraphrased instruction and answer line
# - output/prompts_option_order.csv: answer options in reverse order
# - output/prompts_name_swap.csv: Greek and Arabic/Turkish first names swapped (same gender), to check whether the
#   group effects come from the first names that each group was written with
import re
import unicodedata
from pathlib import Path
import pandas as pd

PREFIX = "Απάντησε με ακριβώς μία λέξη στο παρακάτω: "
PREFIX_PARAPHRASE = "Διάβασε το παρακάτω κείμενο και δώσε μία μόνο λέξη ως απάντηση: "
ANSWER_LINE = "Απάντησε μόνο με "
ANSWER_LINE_PARAPHRASE = "Οι επιτρεπτές απαντήσεις είναι: "
UNKNOWN_OPTION = "Άγνωστο αν δεν μπορείς να αποφασίσεις"  # last option of template 10

# gender-preserving swap of Greek and Arabic/Turkish first names (keys without accents)
NAME_SWAP = {"γιωργος": "Μοχάμεντ", "μοχαμεντ": "Γιώργος", "γιαννης": "Αλί", "αλι": "Γιάννης",
             "μαρια": "Ζεϊνέ", "ζεινε": "Μαρία", "ελενη": "Εμινέ", "εμινε": "Ελένη"}
ACCUSATIVE = {"Γιώργος": "Γιώργο", "Γιάννης": "Γιάννη"}  # the other names are the same in the accusative
ACCUSATIVE_TEMPLATES = [2, 4, 5, 6, 7]  # "από τον <person 2> που ... ή από τον <person 3> που ..."


def normalize(text):
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()


def split_answer_line(prompt):
    # returns the prompt without its last sentence and the answer options of the last sentence
    start = prompt.rindex(ANSWER_LINE)
    options = prompt[start + len(ANSWER_LINE):].strip().rstrip(".").split(" ή ")
    return prompt[:start], options


def paraphrase(prompt):
    body, options = split_answer_line(prompt)
    body = body.replace(PREFIX, PREFIX_PARAPHRASE)
    return body + ANSWER_LINE_PARAPHRASE + ", ".join(options) + "."


def option_order(prompt):
    body, options = split_answer_line(prompt)
    if options[-1] == UNKNOWN_OPTION:
        options = options[:-1][::-1] + [UNKNOWN_OPTION]  # keep "Unknown if you cannot decide" last
    else:
        options = options[::-1]
    return body + ANSWER_LINE + " ή ".join(options) + "."


def swap_names(row):
    text = row["prompt"]
    names = [row[f"firstname_{k}"] for k in [1, 2, 3]]
    new_names = [NAME_SWAP[normalize(name)] for name in names]
    # first replace every mention with a placeholder (so that swapped names are not swapped back)
    if row["template"] in ACCUSATIVE_TEMPLATES:
        for k in [1, 2]:  # persons 2 and 3 are in the accusative
            old = ACCUSATIVE.get(names[k], names[k])
            text = re.sub(rf"(από (?:τον|την|το) ){re.escape(old)}( που)", rf"\g<1>@ACC{k}@\g<2>", text)
    for k in range(3):
        text = re.sub(rf"(?<!\w){re.escape(names[k])}(?!\w)", f"@NOM{k}@", text)
    for k in range(3):
        text = text.replace(f"@ACC{k}@", ACCUSATIVE.get(new_names[k], new_names[k]))
        text = text.replace(f"@NOM{k}@", new_names[k])
    return text, new_names


pilot_df = pd.read_csv(Path("output/prompts_pilot_v2.csv"))

paraphrase_df = pilot_df.copy()
paraphrase_df["prompt"] = paraphrase_df["prompt"].map(paraphrase)
paraphrase_df.to_csv(Path("output/prompts_paraphrase.csv"), index=False, encoding="utf-8")

option_order_df = pilot_df.copy()
option_order_df["prompt"] = option_order_df["prompt"].map(option_order)
option_order_df.to_csv(Path("output/prompts_option_order.csv"), index=False, encoding="utf-8")

name_swap_df = pilot_df.copy()
swapped = name_swap_df.apply(swap_names, axis=1)
name_swap_df["prompt"] = [s[0] for s in swapped]
for k in [1, 2, 3]:
    name_swap_df[f"firstname_{k}"] = [s[1][k - 1] for s in swapped]
# check that no original first name is left in the swapped prompts
for (_, old), (_, new) in zip(pilot_df.iterrows(), name_swap_df.iterrows()):
    for k in [1, 2, 3]:
        old_name = ACCUSATIVE.get(old[f"firstname_{k}"], old[f"firstname_{k}"])
        assert old[f"firstname_{k}"] in new["firstname_1":"firstname_3"].values or \
            not re.search(rf"(?<!\w){re.escape(old_name)}(?!\w)", new["prompt"]), (old["id"], new["prompt"])
name_swap_df.to_csv(Path("output/prompts_name_swap.csv"), index=False, encoding="utf-8")

for name, df in [("paraphrase", paraphrase_df), ("option_order", option_order_df), ("name_swap", name_swap_df)]:
    print(f"{name}: {len(df)} prompts, e.g. (template {df.iloc[0]['template']}):\n  {df.iloc[0]['prompt']}")
