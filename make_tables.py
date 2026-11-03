# Tables and figures of the paper, from the main run (v2_chat) and the robustness summary.
# Usage: python make_tables.py      (after results_analysis.py v2_chat and robustness_summary.py)
#   writes results/paper_tables/*.csv (one row per cell of the paper table, n in every cell, * = systematic difference)
#   and results/figures/*.pdf and *.png
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from results_analysis import MODELS, MIN_N, load_parsed, wilson

RUN = "v2_chat"
R = Path(f"results/{RUN}")
TABLES = Path("results/paper_tables")
FIGURES = Path("results/figures")
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {"qwen": "Qwen3-4B", "krikri": "Krikri-8B", "gemma": "Gemma-4-E2B"}
ATTRIBUTE_NAMES = {"age": "Age", "disability": "Disability", "ethnicity": "Ethnicity", "gender": "Gender",
                   "migration": "Migration status", "occupation": "Occupation", "religion": "Religion",
                   "sexual_orientation": "Sexual orientation", "socioeconomic": "Socioeconomic status"}
FAMILY_NAMES = {"stereotype": "Stereotype (02-07)", "eligibility": "Eligibility (09, 11, 12)",
                "decision": "Decision (16, 17)", "ambiguous_selection": "Ambiguous selection (01, 10, 13)",
                "selection_positive": "Selection, positive context", "selection_negative": "Selection, negative context"}
TEMPLATE_NAMES = {16: "Promotion (16)", 17: "Resignation (17)"}

# reference categorical palette (slots 1-3 for the models, validated for all pairs), plus one marker shape per model
# so that the figures also work in greyscale print; text and axes use ink colors, never the series colors
MODEL_COLORS = {"qwen": "#2a78d6", "krikri": "#eb6834", "gemma": "#1baf7a"}
MODEL_MARKERS = {"qwen": "o", "krikri": "s", "gemma": "^"}
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def rate(k, n):
    # e.g. "46.5 (67/144)"
    return f"{100 * k / n:.1f} ({int(k)}/{int(n)})" if n else ""


def with_ci(x, low, high, scale=100, digits=1):
    return "" if pd.isna(x) else f"{scale * x:.{digits}f} [{scale * low:.{digits}f}, {scale * high:.{digits}f}]"


def mark(row):
    return "*" if row["systematic"] else ""


def wide(df, keys, cell):
    # one row per group, one column per model
    df = df.assign(cell=[cell(r) for _, r in df.iterrows()])
    table = df.pivot_table(index=keys, columns="model", values="cell", aggfunc="first").reindex(columns=MODELS)
    return table.rename(columns=MODEL_NAMES).fillna("").reset_index()


def save(table, name):
    if "attribute" in table:
        table["attribute"] = table["attribute"].map(ATTRIBUTE_NAMES)
    if "model" in table:
        table["model"] = table["model"].map(MODEL_NAMES).fillna(table["model"])
    table.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")  # with BOM, for spreadsheet programs
    print(f"{name}: {len(table)} rows")


# Table 1. response validity per model and template family
v = pd.read_csv(R / "response_validity.csv")
save(pd.DataFrame({"model": v["model"], "family": v["family"].map(FAMILY_NAMES), "n": v["n"],
                   "exact answer (%)": (100 * v["share_exact"]).round(1),
                   "read from a longer answer (%)": (100 * (v["share_extracted"] + v["share_phrase"])).round(1),
                   "refusal, coded Unknown (%)": (100 * v["share_refusal"]).round(2),
                   "non-compliant (%)": (100 * v["share_none"]).round(2)}), "table_response_validity")

# Table 2. disambiguated selection: accuracy, macro accuracy, TPR per group of the correct person, EOD per dimension
accuracy = pd.read_csv(R / "selection_accuracy.csv")
rows = []
for _, r in accuracy.iterrows():
    k = round(r["accuracy"] * r["n"])
    low, high = wilson(k, r["n"])
    rows.append({"model": r["model"], "n": r["n"], "accuracy % [95% CI]": with_ci(k / r["n"], low, high),
                 "macro accuracy % [bootstrap 95% CI]": with_ci(r["macro_accuracy"], r["macro_ci_low"], r["macro_ci_high"]),
                 "non-compliant (%)": round(100 * r["invalid"], 2)})
save(pd.DataFrame(rows), "table_selection_accuracy")
tpr = pd.read_csv(R / "selection_tpr.csv")
tpr["family"] = tpr["family"].map(FAMILY_NAMES)
save(wide(tpr, ["family", "attribute", "group"], lambda r: rate(round(r["tpr"] * r["n"]), r["n"])), "table_selection_tpr")
eod = pd.read_csv(R / "selection_eod.csv")
save(wide(eod, ["attribute"], lambda r: f"{with_ci(r['eod'], r['ci_low'], r['ci_high'])}{mark(r)} "
                                         f"(lowest: {r['lowest']})"), "table_selection_eod")

# Table 3. ambiguous selection: abstention per template, over-selection index per group
abstention = pd.read_csv(R / "ambiguous_abstention.csv")
save(wide(abstention, ["template"], lambda r: f"{100 * r['unknown']:.1f} (n = {r['n']})"), "table_ambiguous_abstention")
osi = pd.read_csv(R / "ambiguous_selection.csv")
save(wide(osi, ["attribute", "group"], lambda r: f"{r['osi']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]{mark(r)} "
                                                 f"(n = {r['n_present']})"), "table_ambiguous_osi")

# Table 4. stereotype endorsement per group and DPD per dimension
endorsement = pd.read_csv(R / "stereotype_endorsement.csv")
save(wide(endorsement, ["attribute", "group"], lambda r: rate(r["endorsed"], r["n"]) + mark(r)), "table_stereotype_endorsement")
dpd = pd.read_csv(R / "stereotype_dpd.csv")
save(wide(dpd, ["attribute"], lambda r: f"{with_ci(r['dpd'], r['ci_low'], r['ci_high'])}{mark(r)}"), "table_stereotype_dpd")

# Table 5. eligibility questions: denial rate per group
eligibility = pd.read_csv(R / "eligibility.csv")
save(wide(eligibility, ["attribute", "group"], lambda r: rate(r["denied"], r["n"]) + mark(r)), "table_eligibility")

# Table 6. decisions: accuracy and unfavourable answers per group
decisions = pd.read_csv(R / "decisions.csv")
decisions["template"] = decisions["template"].map(TEMPLATE_NAMES)
save(wide(decisions, ["template", "attribute", "group"],
          lambda r: f"{100 * r['accuracy']:.1f}; {rate(r['unfavourable'], r['n'])}{mark(r)}"), "table_decisions")

# Table 7. false selection in negative contexts (robbery, exam leak, phone scam)
false_selection = pd.read_csv(R / "false_selection.csv")
save(wide(false_selection, ["attribute", "group"], lambda r: rate(r["wrongly_selected"], r["n"]) + mark(r)),
     "table_false_selection")

# Table 8. all systematic group differences, with the GEE sensitivity check and the minimum detectable difference
CONTRASTS = {"Stereotype endorsement": (endorsement, "endorsed"), "Eligibility: denied": (eligibility, "denied"),
             "Decision: unfavourable answer": (decisions, "unfavourable"),
             "False selection": (false_selection, "wrongly_selected")}
rows = []
for name, (table, events) in CONTRASTS.items():
    for _, r in table[table["systematic"]].iterrows():
        rows.append({"table": name, "model": r["model"], "template": r.get("template", ""), "attribute": r["attribute"],
                     "group": r["group"], "group rate % (k/n)": rate(r[events], r["n"]),
                     "other groups % (k/n)": rate(r["events_rest"], r["n_rest"]),
                     "difference, points [95% CI]": with_ci(r["difference"], r["diff_ci_low"], r["diff_ci_high"]),
                     "MH odds ratio [95% CI]": with_ci(r["or_vs_rest"], r["or_ci_low"], r["or_ci_high"], scale=1, digits=2),
                     "test": r["test"], "adjusted p": f"{r['p_adjusted']:.3g}",
                     "GEE odds ratio": "" if pd.isna(r["gee_or"]) else f"{r['gee_or']:.2f}",
                     "GEE p": "" if pd.isna(r["gee_p"]) else f"{r['gee_p']:.3g}",
                     "min. detectable difference, points": round(100 * r["min_detectable_difference"], 1)})
save(pd.DataFrame(rows), "table_systematic_differences")

# Table 9. between-model comparisons on the same prompts
comparison = pd.read_csv(R / "model_comparison.csv")
comparison["model_2"] = comparison["model_2"].map(MODEL_NAMES).fillna(comparison["model_2"])
save(pd.DataFrame({"metric": comparison["metric"], "test": comparison["test"], "model": comparison["model_1"],
                   "vs": comparison["model_2"], "n": comparison["n"],
                   "rates %": [("" if pd.isna(a) else f"{100 * a:.1f} vs {100 * b:.1f}")
                               for a, b in zip(comparison["rate_1"], comparison["rate_2"])],
                   "difference, points": (100 * comparison["difference"]).round(1),
                   "adjusted p": comparison["p_adjusted"].map(lambda p: f"{p:.3g}")}), "table_model_comparison")

# Table 10. robustness
agreement = pd.read_csv("results/robustness/agreement.csv")
save(wide(agreement, ["condition"], lambda r: f"{100 * r['agreement']:.1f} ({r['kappa']:.2f})"), "table_robustness_agreement")
fleiss = pd.read_csv("results/robustness/fleiss_kappa.csv")
save(wide(fleiss, ["set"], lambda r: f"{100 * r['same_answer_in_all']:.1f} ({r['fleiss_kappa']:.2f})"),
     "table_robustness_fleiss")
verdicts = pd.read_csv("results/robustness/headline_findings.csv")
save(verdicts[["finding", "v2_chat", "verdict"]], "table_robustness_findings")


# Figure 1. answer distribution per model and template family
def answer_category(d, model):
    answer = d[f"answer_{model}"]
    person = d[[f"firstname_{k}" for k in [1, 2, 3]]].eq(answer, axis=0).any(axis=1)
    category = pd.Series("non-compliant", index=d.index)
    category[answer == "Ναι"] = "Yes"
    category[answer == "Όχι"] = "No"
    category[answer == "Άγνωστο"] = "Unknown"
    category[person & (answer == d["ground_truth"])] = "person: correct"
    category[person & (answer != d["ground_truth"])] = "person: other"
    return category


CATEGORIES = {"Yes": "#2a78d6", "No": "#eb6834", "Unknown": "#1baf7a", "person: correct": "#eda100",
              "person: other": "#e87ba4", "non-compliant": "#a3a29e"}
data = load_parsed(RUN)
data = data[~data["debatable_gold"]]
families = list(FAMILY_NAMES)
distribution = []
fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True)
for ax, model in zip(axes, MODELS):
    shares = pd.crosstab(data["family"], answer_category(data, model), normalize="index")
    shares = shares.reindex(index=families, columns=list(CATEGORIES), fill_value=0)
    distribution.append(shares.assign(model=model))
    left = pd.Series(0.0, index=families)
    for category, color in CATEGORIES.items():
        ax.barh(range(len(families)), shares[category], left=left, height=0.6, color=color, edgecolor="white",
                linewidth=1, label=category)
        left += shares[category]
    ax.set_title(MODEL_NAMES[model], fontsize=8, loc="left")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(families)), [FAMILY_NAMES[f] for f in families])
axes[0].invert_yaxis()
axes[0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.12), handlelength=1, columnspacing=1)
fig.savefig(FIGURES / "fig_answer_distribution.pdf")
fig.savefig(FIGURES / "fig_answer_distribution.png")
plt.close(fig)
distribution = pd.concat(distribution).reset_index()
distribution["family"] = distribution["family"].map(FAMILY_NAMES)
save(distribution.round(4), "figure_answer_distribution_data")


# Figures 2 and 3: dot plots with one row per group and one marker per model (filled = systematic difference)
def dot_plot(rows, estimates, x_label, reference, file_name, height_per_row=0.2):
    # rows: list of (section, label, key); estimates: model -> key -> (value, low, high, systematic)
    # rows without any estimate are dropped, and every section starts with a heading row
    rows = [row for row in rows if any(row[2] in estimates[m] for m in MODELS)]
    display = []  # (label, key), key None for a heading
    for i, (section, label, key) in enumerate(rows):
        if i == 0 or rows[i - 1][0] != section:
            display.append((section, None))
        display.append((label, key))
    fig, ax = plt.subplots(figsize=(6.5, 0.6 + height_per_row * len(display)))
    offsets = {"qwen": -0.22, "krikri": 0, "gemma": 0.22}
    for model in MODELS:
        for y, (_, key) in enumerate(display):
            if key not in estimates[model]:
                continue
            value, low, high, systematic = estimates[model][key]
            ax.plot([low, high], [y + offsets[model]] * 2, color=MODEL_COLORS[model], linewidth=1.2)
            ax.plot(value, y + offsets[model], marker=MODEL_MARKERS[model], markersize=5, color=MODEL_COLORS[model],
                    markerfacecolor=MODEL_COLORS[model] if systematic else "white", markeredgewidth=1.2, linestyle="")
    ax.axvline(reference, color=INK_SECONDARY, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(display)), [label for label, _ in display])
    for y, (tick, (_, key)) in enumerate(zip(ax.get_yticklabels(), display)):
        if key is None:  # section heading, with a separator above it
            tick.set_fontweight("bold")
            if y > 0:
                ax.axhline(y - 0.5, color=GRID, linewidth=0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(display) - 0.5, -0.5)
    ax.set_xlabel(x_label)
    handles = [plt.Line2D([], [], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m], linestyle="", markersize=5,
                          label=MODEL_NAMES[m]) for m in MODELS]
    handles.append(plt.Line2D([], [], marker="o", color=INK_SECONDARY, markerfacecolor="white", linestyle="",
                              markersize=5, label="not systematic"))
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(FIGURES / f"{file_name}.pdf")
    fig.savefig(FIGURES / f"{file_name}.png")
    plt.close(fig)


# Figure 2. group differences (group rate - rate of the other groups of the dimension) for every group with a
# systematic difference in at least one model
rows, estimates = [], {m: {} for m in MODELS}
for name, (table, _) in CONTRASTS.items():
    table = table.copy()
    table["key"] = [(name, r.get("template", ""), r["attribute"], r["group"]) for _, r in table.iterrows()]
    for key in table.loc[table["systematic"], "key"].drop_duplicates().sort_values():
        _, template, attribute, group = key
        prefix = f"{template}, " if template else ""  # promotion or resignation, for the decision templates
        rows.append((name, f"{prefix}{ATTRIBUTE_NAMES[attribute]}: {group}", key))
    for _, r in table.iterrows():
        if r["n"] >= MIN_N:
            estimates[r["model"]][r["key"]] = (100 * r["difference"], 100 * r["diff_ci_low"], 100 * r["diff_ci_high"],
                                               r["systematic"])
dot_plot(rows, estimates, "Difference from the other groups of the dimension (percentage points, 95% CI)", 0,
         "fig_group_differences")

# Figure 3. over-selection index in the ambiguous selection templates (1 = chosen as often as chance)
rows, estimates = [], {m: {} for m in MODELS}
for (attribute, group), _ in osi.groupby(["attribute", "group"]):
    rows.append((ATTRIBUTE_NAMES[attribute], group, (attribute, group)))
for _, r in osi.iterrows():
    if r["n_present"] >= MIN_N:
        estimates[r["model"]][(r["attribute"], r["group"])] = (r["osi"], r["ci_low"], r["ci_high"], r["systematic"])
dot_plot(rows, estimates, "Over-selection index (95% CI); groups with at least 30 presences", 1, "fig_over_selection")
print(f"Figures written to {FIGURES}")


# Figure 1. pipeline of the study (a diagram, so no data colors: grey boxes, the robustness branch in light blue)
def box(ax, x, y, width, height, title, text, fill):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                                   boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=fill,
                                                   edgecolor=INK_SECONDARY, linewidth=0.6))
    ax.text(x, y + height / 2 - 0.022, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(x, y + height / 2 - 0.058, text, ha="center", va="top", fontsize=7, color=INK_SECONDARY, linespacing=1.35)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, linewidth=0.8,
                                                          shrinkA=0, shrinkB=0, mutation_scale=9))


MAIN, SIDE = "#f0efec", "#cde2fb"
steps = [("Templates and lexicon", "27 Jinja2 templates in five task families\n9 dimensions, 40 groups, Greek forms, 8 first names"),
         ("Prompt generation (Algorithm 1)", "every combination of three group forms\nper template and dimension: 23,166 prompts"),
         ("Local inference", "llama.cpp (SYCL) on one Intel Arc A770 GPU\nQwen3-4B, Krikri-8B, Gemma 4 E2B (GGUF Q5_K_M)\nnative chat template, greedy decoding"),
         ("Answer parsing (Algorithm 2)", "Yes / No / Unknown / name, refusal, non-compliant\nrule-based, checked by hand on 344 answers"),
         ("Metrics", "endorsement, favourability, denial, unfavourable decisions,\nover-selection, accuracy, EOD, DPD, false selection"),
         ("Statistics", "Wilson and Newcombe CIs, Mantel–Haenszel odds ratios,\nBH correction, ≥ 5 pp and n ≥ 30, GEE sensitivity")]
fig, ax = plt.subplots(figsize=(6.5, 6.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
x_main, width, height, gap = 0.33, 0.62, 0.13, 0.03
ys = [0.985 - height / 2 - i * (height + gap) for i in range(len(steps))]
for (title, text), y in zip(steps, ys):
    box(ax, x_main, y, width, height, title, text, MAIN)
for y_top, y_bottom in zip(ys[:-1], ys[1:]):
    arrow(ax, (x_main, y_top - height / 2), (x_main, y_bottom + height / 2))
x_side, side_width = 0.83, 0.32
box(ax, x_side, ys[2], side_width, height, "Robustness conditions",
    "sampling (3 seeds), name swap,\nparaphrase, option order,\nprompt format", SIDE)
arrow(ax, (x_side - side_width / 2, ys[2]), (x_main + width / 2, ys[2]))
box(ax, x_side, ys[5], side_width, height, "Robustness summary",
    "agreement with the main run,\nCohen's and Fleiss' κ,\nheadline findings", SIDE)
arrow(ax, (x_main + width / 2, ys[5]), (x_side - side_width / 2, ys[5]))
fig.savefig(FIGURES / "fig1_pipeline.pdf")
fig.savefig(FIGURES / "fig1_pipeline.png")
plt.close(fig)
print("Figure 1 written")
