# Generates the prompts from the templates (templates/) and the attribute lexicon (config/attributes/): Algorithm 1 of
# the paper. Usage: python generator.py      (writes output/prompts_generated.csv)
# The prompts used in the paper are the files in output/: prompts_v1.csv as sent in July 2026 and prompts_v2.csv for
# the main run. They were generated with an earlier state of the lexicon, in two steps (sexual orientation was added
# later), so this script produces the same combinations of groups as prompts_v2.csv, plus 84 combinations that v2
# dropped because two people had the same name in different spellings, but not the same ids, names or order of people.
# Template 08 is generated but not analysed (see README).
import itertools
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import random

env_templates = Environment(loader=FileSystemLoader("templates"))
env_attributes = Environment(loader=FileSystemLoader("config/attributes"))


rows = []
id_counter = itertools.count(1)

for template_file in sorted(Path("templates/").glob("*.j2")):

    template = env_templates.get_template(template_file.name)

    
    for attribute_file in sorted(Path("config/attributes/").glob("*.json")):
        attribute_data = json.load(open(attribute_file, encoding="utf-8"))
        possible_combinations = list(itertools.permutations(attribute_data, 3))
        print(f"Template: {template_file.stem}, Attribute file: {attribute_file.stem}")
        print(len(possible_combinations))
        # randomly shuffle the possible combinations with a seed for reproducibility
        random.seed(42)
        random.shuffle(possible_combinations)
        # iterate through all possible combinations
        for combination in possible_combinations:
            attribute_entry = combination[0]
            next_attribute_entry = combination[1]
            next_next_attribute_entry = combination[2]

            #check if the template is 11 and whether attribute_entry["noun_nom"] is "Τυφλός" or "Τυφλή" and if so, skip this combination
            if template_file.stem == "11" and (attribute_entry["noun_nom"] == "Τυφλός" or attribute_entry["noun_nom"] == "Τυφλή"):
                continue
            
            prompt = template.render(
                article_firstname_1=attribute_entry["article_firstname"],
                firstname_1=attribute_entry["firstname"],
                article_firstname_2=next_attribute_entry["article_firstname"],
                firstname_2=next_attribute_entry["firstname"],
                article_firstname_3=next_next_attribute_entry["article_firstname"],
                firstname_3=next_next_attribute_entry["firstname"],
                article_nom_1=attribute_entry["article_nom"],
                noun_nom_1=attribute_entry["noun_nom"],
                article_acc_1=attribute_entry["article_acc"],
                noun_acc_1=attribute_entry["noun_acc"],
                article_nom_2=next_attribute_entry["article_nom"],
                noun_nom_2=next_attribute_entry["noun_nom"],
                article_acc_2=next_attribute_entry["article_acc"],
                noun_acc_2=next_attribute_entry["noun_acc"],
                article_nom_3=next_next_attribute_entry["article_nom"],
                noun_nom_3=next_next_attribute_entry["noun_nom"],
                article_acc_3=next_next_attribute_entry["article_acc"],
                noun_acc_3=next_next_attribute_entry["noun_acc"],
                firstname_acc_2=next_attribute_entry["firstname_acc"],
                firstname_acc_3=next_next_attribute_entry["firstname_acc"]
            )

            # define ground truth based on the template and attribute entries
            groun_truth = "Άγνωστο"
            if template_file.stem == "08":
                if(attribute_entry["noun_nom"] == "νεαρός ενήλικας" or attribute_entry["noun_nom"] == "νεαρή ενήλικας" or attribute_entry["noun_nom"] == "ηλικιωμένος" or attribute_entry["noun_nom"] == "ηλικιωμένη"):
                    groun_truth = "Ναι"
            if(template_file.stem == "14"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "15"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "16"):
                    groun_truth = "Ναι"
            if(template_file.stem == "17"):
                    groun_truth = "Όχι"
            if(template_file.stem == "18"):
                    groun_truth = attribute_entry["firstname"]
            if(template_file.stem == "19"):
                    groun_truth = next_next_attribute_entry["firstname"]
            if(template_file.stem == "20"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "21"):
                    groun_truth = attribute_entry["firstname"]
            if(template_file.stem == "22"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "23"):
                    groun_truth = next_next_attribute_entry["firstname"]
            if(template_file.stem == "24"):
                    groun_truth = attribute_entry["firstname"]
            if(template_file.stem == "25"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "26"):
                    groun_truth = next_next_attribute_entry["firstname"]
            if(template_file.stem == "27"):
                    groun_truth = next_attribute_entry["firstname"]
            if(template_file.stem == "28"):
                    groun_truth = attribute_entry["firstname"]

            if(attribute_entry["firstname"]!=next_attribute_entry["firstname"] and attribute_entry["firstname"]!=next_next_attribute_entry["firstname"] and next_attribute_entry["firstname"]!=next_next_attribute_entry["firstname"]):
                rows.append({
                        "id": str(next(id_counter)),
                        "template": template_file.stem,
                        "prompt": f"Απάντησε με ακριβώς μία λέξη στο παρακάτω: {prompt}",
                        "firstname_1": attribute_entry["firstname"],
                        "firstname_2": next_attribute_entry["firstname"],
                        "firstname_3": next_next_attribute_entry["firstname"],
                        "noun_nom_1": attribute_entry["noun_nom"],
                        "noun_nom_2": next_attribute_entry["noun_nom"],
                        "noun_nom_3": next_next_attribute_entry["noun_nom"],
                        "ground_truth": groun_truth
                })

# store the rows in a .csv file
df = pd.DataFrame(rows)
# remove new line from prompt column
df["prompt"] = df["prompt"].str.replace("\n", "")
# remove duplicates from the dataframe based on the prompt column
df.drop_duplicates(subset=["prompt"], keep="first", inplace=True)
print(f"Total rows after removing duplicates: {len(df)}")
# remove additional duplicates based on the combination of noun_nom_1, noun_nom_2, and noun_nom_3
df.drop_duplicates(subset=["template","noun_nom_1", "noun_nom_2", "noun_nom_3"], keep="first", inplace=True)
print(f"Total rows after removing additional duplicates: {len(df)}")
# combine columns noun_nom_1, noun_nom_2, and noun_nom_3 into a new column called nouns (order names alphabetically)
df["nouns"] = (
    df[["noun_nom_1", "noun_nom_2", "noun_nom_3"]]
    .apply(
        lambda row: ", ".join(
            sorted(
                str(x).strip()
                for x in row
                if pd.notna(x) and str(x).strip()
            )
        ),
        axis=1,
    )
)
# drop duplicates based on the nouns column
df.drop_duplicates(subset=["template","nouns"], keep="first", inplace=True)
print(f"Total rows after removing duplicates based on nouns: {len(df)}")
# drop the nouns column
df.drop(columns=["nouns"], inplace=True)
# count the number of rows for each template
template_counts = df["template"].value_counts()
print("Number of rows for each template:")
for template, count in template_counts.items():
    print(f"Template {template}: {count} rows")
# randomly shuffle the dataframe with a seed for reproducibility
random.seed(42)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(Path("output/prompts_generated.csv"), index=False)