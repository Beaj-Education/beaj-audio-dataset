# %%
"""
v3_ai_transcription/analysis.py

Compare human transcriptions with AI model transcriptions using Word Error Rate (WER).

Part 1: Inter-grader agreement — pairwise WER between human graders on shared audio files.
Part 2: Human vs AI WER — WER between each AI model's transcription and human transcriptions.
Part 3: Combined pairwise correlation heatmap using word-correct counts (all humans + AI models).

Outputs:
  - data/clean/v3_inter_grader_wer.csv
  - data/clean/v3_human_vs_ai_wer.csv
  - plots/v3_inter_grader_wer_heatmap.png
  - plots/v3_human_vs_ai_wer_heatmap.png
  - plots/v3_combined_correlation_heatmap.png
  - plots/v3_reliable_correlation_heatmap.png
  - V3_AI_TRANSCRIPTION_ANALYSIS.md
"""

import os
import re
import pandas as pd
import numpy as np
from jiwer import wer as compute_wer, process_words
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "v3", "ai_transcribed", "main.csv")
GRADED_V2_DIR = os.path.join(DATA_DIR, "v2", "graded")
CLEAN_DIR = os.path.join(DATA_DIR, "v3", "clean")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots", "v3")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

OUT_INTERGRADER_CSV = os.path.join(CLEAN_DIR, "v3_inter_grader_wer.csv")
OUT_HUMAN_AI_CSV = os.path.join(CLEAN_DIR, "v3_human_vs_ai_wer.csv")
OUT_INTERGRADER_PLOT = os.path.join(PLOTS_DIR, "v3_inter_grader_wer_heatmap.png")
OUT_HUMAN_AI_PLOT = os.path.join(PLOTS_DIR, "v3_human_vs_ai_wer_heatmap.png")
OUT_CORR_PLOT = os.path.join(PLOTS_DIR, "v3_combined_correlation_heatmap.png")
OUT_MARKDOWN = os.path.join(PROJECT_DIR, "V3_AI_TRANSCRIPTION_ANALYSIS.md")

# AI models present in main.csv (excluding gemini-2.5-pro which has 0 transcriptions)
AI_MODELS = [
    "scribe_v2", "gemini-3-pro", "voxtral small", "gemini-3-flash",
    "universal-3-pro", "voxtral mini", "universal",
    "gpt-4o-transcribe (azure)", "gpt-4o-transcribe (openai)",
    "gemini-3.1-pro", "sarvam",
]

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def clean_text(text):
    """Normalize text for WER comparison: lowercase, strip punctuation (keep apostrophes)."""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_wer(reference, hypothesis):
    """Compute WER between two transcriptions. Returns NaN for invalid inputs."""
    ref = clean_text(reference)
    hyp = clean_text(hypothesis)
    if not ref or not hyp:
        return np.nan
    try:
        return compute_wer(ref, hyp)
    except Exception:
        return np.nan


def compute_words_correct(reference, transcription):
    """Count correctly matched words using jiwer alignment (same method as v2_analysis)."""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(transcription)
    if not ref_clean or not hyp_clean:
        return 0
    try:
        result = process_words(ref_clean, hyp_clean)
        return result.hits
    except Exception:
        return 0


# ──────────────────────────────────────────────
# Step 1: Load data
# ──────────────────────────────────────────────
print("=" * 70)
print("Step 1: Loading data")
print("=" * 70)

df_main = pd.read_csv(MAIN_CSV)
print(f"Main CSV: {len(df_main)} audio files")

# Load human grader CSVs into {grader_name: {audio_file: transcription}}
grader_transcriptions = {}
for fname in sorted(os.listdir(GRADED_V2_DIR)):
    if not fname.endswith(".csv"):
        continue
    grader_name = fname.split(" - ")[0].strip()
    fpath = os.path.join(GRADED_V2_DIR, fname)
    df_g = pd.read_csv(fpath)
    df_g = df_g[[c for c in df_g.columns if not c.startswith("Unnamed")]]
    has_trans = df_g["Human Transcription"].notna() & (
        df_g["Human Transcription"].astype(str).str.strip() != ""
    )
    df_valid = df_g[has_trans][["audio_file_name", "Human Transcription"]].copy()
    grader_transcriptions[grader_name] = (
        df_valid.set_index("audio_file_name")["Human Transcription"].to_dict()
    )
    print(f"  {grader_name}: {len(df_valid)} transcriptions")

all_graders = sorted(grader_transcriptions.keys())
print(f"\nAll graders: {all_graders}")

# %%
# ──────────────────────────────────────────────
# Step 2: Build unified dataset
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Building unified dataset")
print("=" * 70)

records = []
for _, row in df_main.iterrows():
    audio = row["audio_file_name"]
    question = row["Question"]
    g1 = row["Grader 1"]
    g2 = row["Grader 2"]

    t1 = grader_transcriptions.get(g1, {}).get(audio) if pd.notna(g1) else None
    t2 = grader_transcriptions.get(g2, {}).get(audio) if pd.notna(g2) else None

    ai_trans = {}
    for model in AI_MODELS:
        val = row.get(model)
        if pd.notna(val) and str(val).strip():
            ai_trans[model] = str(val).strip()

    records.append({
        "audio_file_name": audio,
        "question": question,
        "grader_1": g1,
        "grader_2": g2,
        "transcription_1": t1,
        "transcription_2": t2,
        **{f"ai_{model}": ai_trans.get(model) for model in AI_MODELS},
    })

df = pd.DataFrame(records)
has_both = df["transcription_1"].notna() & df["transcription_2"].notna()
has_any = df["transcription_1"].notna() | df["transcription_2"].notna()

print(f"Total audio files: {len(df)}")
print(f"With at least one human transcription: {has_any.sum()}")
print(f"With both human transcriptions (dual-graded): {has_both.sum()}")

# %%
# ══════════════════════════════════════════════
# PART 1: INTER-GRADER AGREEMENT
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: Inter-grader agreement (WER between paired graders)")
print("=" * 70)

# Per-audio inter-grader WER
df_dual = df[has_both].copy()
df_dual["inter_grader_wer"] = df_dual.apply(
    lambda r: safe_wer(r["transcription_1"], r["transcription_2"]), axis=1
)
valid_ig_wer = df_dual["inter_grader_wer"].dropna()

print(f"\nInter-grader WER statistics ({len(valid_ig_wer)} dual-graded audios):")
print(f"  Mean WER:   {valid_ig_wer.mean():.4f} ({valid_ig_wer.mean()*100:.1f}%)")
print(f"  Median WER: {valid_ig_wer.median():.4f} ({valid_ig_wer.median()*100:.1f}%)")
print(f"  Std WER:    {valid_ig_wer.std():.4f}")
print(f"  Min WER:    {valid_ig_wer.min():.4f}")
print(f"  Max WER:    {valid_ig_wer.max():.4f}")

# Save per-audio inter-grader WER
ig_save = df_dual[
    ["audio_file_name", "question", "grader_1", "grader_2",
     "transcription_1", "transcription_2", "inter_grader_wer"]
].copy()
ig_save.to_csv(OUT_INTERGRADER_CSV, index=False)
print(f"Saved to: {OUT_INTERGRADER_CSV}")

# ──────────────────────────────────────────────
# Pairwise inter-grader WER matrix
# ──────────────────────────────────────────────
print("\nPairwise inter-grader WER matrix:")

ig_pairwise = pd.DataFrame(np.nan, index=all_graders, columns=all_graders)
ig_overlap = pd.DataFrame(0, index=all_graders, columns=all_graders, dtype=int)

for g1 in all_graders:
    for g2 in all_graders:
        if g1 == g2:
            ig_pairwise.loc[g1, g2] = 0.0
            ig_overlap.loc[g1, g2] = len(grader_transcriptions[g1])
            continue
        shared_audios = (
            set(grader_transcriptions[g1].keys()) &
            set(grader_transcriptions[g2].keys())
        )
        ig_overlap.loc[g1, g2] = len(shared_audios)
        if len(shared_audios) >= 3:
            wers = []
            for audio in shared_audios:
                w = safe_wer(
                    grader_transcriptions[g1][audio],
                    grader_transcriptions[g2][audio],
                )
                if not np.isnan(w):
                    wers.append(w)
            if wers:
                ig_pairwise.loc[g1, g2] = np.mean(wers)

ig_pairwise = ig_pairwise.astype(float)

print("Overlap (shared audio files):")
print(ig_overlap.to_string())
print("\nMean WER:")
print(ig_pairwise.round(3).to_string())

# ──────────────────────────────────────────────
# Heatmap: inter-grader WER
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

# Annotation labels: show WER% and overlap count
annot_labels = pd.DataFrame("", index=all_graders, columns=all_graders)
for g1 in all_graders:
    for g2 in all_graders:
        val = ig_pairwise.loc[g1, g2]
        n = ig_overlap.loc[g1, g2]
        if g1 == g2:
            annot_labels.loc[g1, g2] = f"n={n}"
        elif pd.notna(val) and n > 0:
            annot_labels.loc[g1, g2] = f"{val:.0%}\n(n={n})"
        elif n > 0:
            annot_labels.loc[g1, g2] = f"n={n}"

off_diag_vals = ig_pairwise.values[~np.eye(len(all_graders), dtype=bool)]
ig_vmax = float(np.nanmax(off_diag_vals)) if np.any(~np.isnan(off_diag_vals)) else 1.0
ig_vmax = max(ig_vmax, 0.5)  # at least 0.5 for readable color range

sns.heatmap(
    ig_pairwise,
    annot=annot_labels,
    fmt="",
    cmap="YlOrRd",
    vmin=0,
    vmax=ig_vmax,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Mean WER"},
    ax=ax,
    annot_kws={"fontsize": 8},
)
ax.set_title("Pairwise Inter-Grader WER\n(lower = more agreement)", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(OUT_INTERGRADER_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved heatmap to: {OUT_INTERGRADER_PLOT}")
plt.close()

# %%
# ══════════════════════════════════════════════
# PART 2: HUMAN vs AI WER
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Human vs AI WER")
print("=" * 70)

# For each audio, compute WER(human_transcription, ai_transcription)
# Human transcription is the reference (ground truth)
wer_records = []
for _, row in df.iterrows():
    audio = row["audio_file_name"]
    question = row["question"]

    human_trans = {}
    if pd.notna(row.get("transcription_1")):
        human_trans[row["grader_1"]] = row["transcription_1"]
    if pd.notna(row.get("transcription_2")):
        human_trans[row["grader_2"]] = row["transcription_2"]

    if not human_trans:
        continue

    for model in AI_MODELS:
        ai_col = f"ai_{model}"
        ai_val = row.get(ai_col)
        if pd.isna(ai_val) or not str(ai_val).strip():
            continue

        for grader, h_trans in human_trans.items():
            w = safe_wer(h_trans, str(ai_val))
            wer_records.append({
                "audio_file_name": audio,
                "question": question,
                "grader": grader,
                "ai_model": model,
                "human_transcription": h_trans,
                "ai_transcription": str(ai_val),
                "wer": w,
            })

df_wer = pd.DataFrame(wer_records)
df_wer = df_wer.dropna(subset=["wer"])
print(f"Total WER comparisons: {len(df_wer)}")

# Summary table by AI model
summary = df_wer.groupby("ai_model")["wer"].agg(["mean", "median", "std", "count"])
summary.columns = ["mean_wer", "median_wer", "std_wer", "n_comparisons"]
# Round WER columns to 3 decimals (matching .1% display precision)
summary[["mean_wer", "median_wer", "std_wer"]] = summary[["mean_wer", "median_wer", "std_wer"]].round(3)
summary = summary.sort_values("mean_wer")

print("\nMean WER by AI model (vs human transcription as reference):")
print(summary.to_string())

# Also compute WER of AI vs reference text (Question)
print("\n\nWER of AI models vs reference text (Question):")
ref_wer_records = []
for _, row in df.iterrows():
    question = row["question"]
    for model in AI_MODELS:
        ai_col = f"ai_{model}"
        ai_val = row.get(ai_col)
        if pd.isna(ai_val) or not str(ai_val).strip():
            continue
        w = safe_wer(question, str(ai_val))
        ref_wer_records.append({"ai_model": model, "wer_vs_reference": w})

df_ref_wer = pd.DataFrame(ref_wer_records).dropna(subset=["wer_vs_reference"])
ref_summary = df_ref_wer.groupby("ai_model")["wer_vs_reference"].agg(["mean", "median", "std", "count"])
ref_summary.columns = ["mean_wer", "median_wer", "std_wer", "n"]
ref_summary[["mean_wer", "median_wer", "std_wer"]] = ref_summary[["mean_wer", "median_wer", "std_wer"]].round(3)
ref_summary = ref_summary.sort_values("mean_wer")
print(ref_summary.to_string())

# Also compute WER of humans vs reference text
print("\n\nWER of human graders vs reference text (Question):")
human_ref_records = []
for _, row in df.iterrows():
    question = row["question"]
    if pd.notna(row.get("transcription_1")):
        w = safe_wer(question, row["transcription_1"])
        human_ref_records.append({"grader": row["grader_1"], "wer_vs_reference": w})
    if pd.notna(row.get("transcription_2")):
        w = safe_wer(question, row["transcription_2"])
        human_ref_records.append({"grader": row["grader_2"], "wer_vs_reference": w})

df_human_ref = pd.DataFrame(human_ref_records).dropna(subset=["wer_vs_reference"])
human_ref_summary = df_human_ref.groupby("grader")["wer_vs_reference"].agg(["mean", "median", "std", "count"])
human_ref_summary.columns = ["mean_wer", "median_wer", "std_wer", "n"]
human_ref_summary[["mean_wer", "median_wer", "std_wer"]] = human_ref_summary[["mean_wer", "median_wer", "std_wer"]].round(3)
human_ref_summary = human_ref_summary.sort_values("mean_wer")
print(human_ref_summary.to_string())

# Save human vs AI WER CSV
df_wer.to_csv(OUT_HUMAN_AI_CSV, index=False)
print(f"\nSaved to: {OUT_HUMAN_AI_CSV}")

# ──────────────────────────────────────────────
# Heatmap: AI models × human graders mean WER
# ──────────────────────────────────────────────
pivot = df_wer.pivot_table(
    index="ai_model",
    columns="grader",
    values="wer",
    aggfunc="mean",
)
model_order = summary.index.tolist()
grader_order = [g for g in all_graders if g in pivot.columns]
pivot = pivot.reindex(index=model_order, columns=grader_order)

# Add overall mean column
pivot["Overall"] = pivot[grader_order].mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 8))

annot_labels = pivot.map(lambda x: f"{x:.0%}" if pd.notna(x) else "")

sns.heatmap(
    pivot,
    annot=annot_labels,
    fmt="",
    cmap="YlOrRd",
    vmin=0,
    vmax=min(pivot.max().max(), 2.0),
    linewidths=0.5,
    cbar_kws={"label": "Mean WER"},
    ax=ax,
    annot_kws={"fontsize": 9},
)
ax.set_title(
    "Human vs AI Transcription WER\n(human transcription = reference; lower = closer to human)",
    fontsize=13,
)
ax.set_xlabel("Human Grader")
ax.set_ylabel("AI Model")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(OUT_HUMAN_AI_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved heatmap to: {OUT_HUMAN_AI_PLOT}")
plt.close()

# %%
# ══════════════════════════════════════════════
# PART 2b: WER COMPARISON ON DUAL-GRADED AUDIOS ONLY
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2b: WER vs reference — dual-graded audios only")
print("=" * 70)

# Filter to dual-graded audios only (same audio set for all scorers)
dual_audios = set(df_dual["audio_file_name"])
print(f"Dual-graded audios: {len(dual_audios)}")

# Human WER vs reference on dual-graded audios
dual_human_records = []
for _, row in df_dual.iterrows():
    question = row["question"]
    if pd.notna(row.get("transcription_1")):
        w = safe_wer(question, row["transcription_1"])
        dual_human_records.append({"scorer": row["grader_1"], "type": "Human", "wer_vs_reference": w})
    if pd.notna(row.get("transcription_2")):
        w = safe_wer(question, row["transcription_2"])
        dual_human_records.append({"scorer": row["grader_2"], "type": "Human", "wer_vs_reference": w})

# AI WER vs reference on dual-graded audios
dual_ai_records = []
for _, row in df_dual.iterrows():
    question = row["question"]
    for model in AI_MODELS:
        ai_col = f"ai_{model}"
        ai_val = row.get(ai_col)
        if pd.isna(ai_val) or not str(ai_val).strip():
            continue
        w = safe_wer(question, str(ai_val))
        dual_ai_records.append({"scorer": model, "type": "AI", "wer_vs_reference": w})

df_dual_scores = pd.DataFrame(dual_human_records + dual_ai_records).dropna(subset=["wer_vs_reference"])

dual_summary = df_dual_scores.groupby(["scorer", "type"])["wer_vs_reference"].agg(["mean", "median", "std", "count"])
dual_summary.columns = ["mean_wer", "median_wer", "std_wer", "n"]
dual_summary[["mean_wer", "median_wer", "std_wer"]] = dual_summary[["mean_wer", "median_wer", "std_wer"]].round(3)
dual_summary = dual_summary.sort_values("mean_wer")

print("\nWER vs reference text — dual-graded audios only (all scorers on same audio set):")
print(dual_summary.to_string())

# Overall averages by type
dual_means = df_dual_scores.groupby("type")["wer_vs_reference"].mean()
print(f"\nOverall mean WER (dual-graded audios):")
print(f"  Human graders: {dual_means.get('Human', float('nan')):.1%}")
print(f"  AI models:     {dual_means.get('AI', float('nan')):.1%}")

# %%
# ══════════════════════════════════════════════
# PART 3: COMBINED PAIRWISE CORRELATION HEATMAP
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: Combined pairwise correlation heatmap (humans + AI)")
print("=" * 70)

# For each scorer (human grader or AI model), compute words correct per audio
# (their transcription vs the reference text, using jiwer alignment).
# Then compute pairwise Pearson correlations across shared audio files.

# Build {scorer_name: {audio_file: transcription}} for all scorers
all_scorers = {}
for g in all_graders:
    all_scorers[g] = grader_transcriptions[g]

for model in AI_MODELS:
    ai_col = f"ai_{model}"
    model_trans = {}
    for _, row in df.iterrows():
        val = row.get(ai_col)
        if pd.notna(val) and str(val).strip():
            model_trans[row["audio_file_name"]] = str(val).strip()
    all_scorers[model] = model_trans

scorer_names = all_graders + AI_MODELS

# Build {audio_file: reference_text} lookup
audio_to_ref = df.set_index("audio_file_name")["question"].to_dict()

# Compute words correct for each scorer × audio (scorer transcription vs reference text)
scorer_scores = {}
for scorer in scorer_names:
    scores = {}
    for audio, transcription in all_scorers[scorer].items():
        ref = audio_to_ref.get(audio)
        if ref:
            hits = compute_words_correct(ref, transcription)
            scores[audio] = hits
    scorer_scores[scorer] = scores
    print(f"  {scorer}: {len(scores)} word-correct scores computed")

# Build score matrix: rows = audio files, columns = scorers
all_audios = sorted(set().union(*[set(s.keys()) for s in scorer_scores.values()]))
score_matrix = pd.DataFrame(index=all_audios, columns=scorer_names, dtype=float)
for scorer in scorer_names:
    for audio, s in scorer_scores[scorer].items():
        score_matrix.loc[audio, scorer] = s

print(f"\nScore matrix: {score_matrix.shape[0]} audio files × {score_matrix.shape[1]} scorers")
for col in scorer_names:
    print(f"  {col}: {score_matrix[col].notna().sum()} non-NaN")

# Compute pairwise Pearson correlations
corr_matrix = pd.DataFrame(np.nan, index=scorer_names, columns=scorer_names)
overlap_matrix = pd.DataFrame(0, index=scorer_names, columns=scorer_names, dtype=int)

for i, s1 in enumerate(scorer_names):
    for j, s2 in enumerate(scorer_names):
        if i == j:
            corr_matrix.loc[s1, s2] = 1.0
            overlap_matrix.loc[s1, s2] = int(score_matrix[s1].notna().sum())
            continue
        mask = score_matrix[s1].notna() & score_matrix[s2].notna()
        n_shared = mask.sum()
        overlap_matrix.loc[s1, s2] = n_shared
        if n_shared >= 3:
            r, p = pearsonr(score_matrix.loc[mask, s1], score_matrix.loc[mask, s2])
            corr_matrix.loc[s1, s2] = r

corr_matrix = corr_matrix.astype(float)

print(f"\nPairwise overlap counts:")
print(overlap_matrix.to_string())
print(f"\nCorrelation matrix:")
print(corr_matrix.round(3).to_string())

# ──────────────────────────────────────────────
# Heatmap: combined pairwise correlation
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 15))

# Annotation labels: show r value (compact format)
corr_annot = corr_matrix.map(
    lambda x: f".{abs(x):.2f}"[1:] if pd.notna(x) and x != 1.0 else ("1.0" if x == 1.0 else "")
)

sns.heatmap(
    corr_matrix,
    annot=corr_annot,
    fmt="",
    cmap="YlGnBu_r",
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson r"},
    ax=ax,
    annot_kws={"fontsize": 7},
)
ax.set_title(
    "Pairwise Correlation of Words Correct (vs reference text)\n"
    "Human Graders + AI Models",
    fontsize=14,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# Separator between humans and AI
n_humans = len(all_graders)
ax.axhline(y=n_humans, color="white", linewidth=3)
ax.axvline(x=n_humans, color="white", linewidth=3)

plt.tight_layout()
plt.savefig(OUT_CORR_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved correlation heatmap to: {OUT_CORR_PLOT}")
plt.close()

# ──────────────────────────────────────────────
# Heatmap: reliable graders only + AI models
# ──────────────────────────────────────────────
RELIABLE_GRADERS = ["Amna", "Rehma", "Rukhshan", "Semal"]
reliable_scorers = RELIABLE_GRADERS + AI_MODELS

rel_corr = corr_matrix.loc[reliable_scorers, reliable_scorers].copy()

# Friendly display names for labels
DISPLAY_NAMES = {
    "Amna": "Amna (H)", "Rehma": "Rehma (H)", "Rukhshan": "Rukhshan (H)", "Semal": "Semal (H)",
    "scribe_v2": "Scribe v2", "gemini-3-pro": "Gemini 3 Pro", "voxtral small": "Voxtral Small",
    "gemini-3-flash": "Gemini 3 Flash", "universal-3-pro": "Universal 3 Pro",
    "voxtral mini": "Voxtral Mini", "universal": "Universal",
    "gpt-4o-transcribe (azure)": "GPT-4o (Azure)", "gpt-4o-transcribe (openai)": "GPT-4o (OpenAI)",
    "gemini-3.1-pro": "Gemini 3.1 Pro", "sarvam": "Sarvam",
}
display_labels = [DISPLAY_NAMES.get(s, s) for s in reliable_scorers]
rel_corr_display = rel_corr.copy()
rel_corr_display.index = display_labels
rel_corr_display.columns = display_labels

# Narrow color range to actual data spread for better contrast
off_diag_vals = rel_corr.values.copy()
np.fill_diagonal(off_diag_vals, np.nan)
vmin_rel = max(float(np.nanmin(off_diag_vals)) - 0.02, 0)
vmin_rel = round(vmin_rel, 1)  # round to nearest 0.1

fig, ax = plt.subplots(figsize=(12, 10))

rel_annot = rel_corr_display.map(
    lambda x: f"{x:.2f}" if pd.notna(x) and x != 1.0 else ("1.00" if x == 1.0 else "")
)

sns.heatmap(
    rel_corr_display,
    annot=rel_annot,
    fmt="",
    cmap="RdYlGn",
    vmin=vmin_rel,
    vmax=1,
    square=True,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Pearson r", "shrink": 0.8},
    ax=ax,
    annot_kws={"fontsize": 9, "fontweight": "bold"},
)
ax.set_title(
    "Pairwise Correlation — Words Correct\nReliable Human Graders (H) + AI Models",
    fontsize=15, fontweight="bold", pad=15,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

# Separator between humans and AI
n_reliable = len(RELIABLE_GRADERS)
ax.axhline(y=n_reliable, color="black", linewidth=2.5)
ax.axvline(x=n_reliable, color="black", linewidth=2.5)

# Section labels
ax.text(n_reliable / 2, -0.8, "HUMAN", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50")
ax.text(n_reliable + (len(AI_MODELS)) / 2, -0.8, "AI MODELS", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50")

plt.tight_layout()
OUT_RELIABLE_CORR_PLOT = os.path.join(PLOTS_DIR, "v3_reliable_correlation_heatmap.png")
plt.savefig(OUT_RELIABLE_CORR_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved reliable graders correlation heatmap to: {OUT_RELIABLE_CORR_PLOT}")
plt.close()

# %%
# ══════════════════════════════════════════════
# GENERATE MARKDOWN REPORT
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating Markdown report")
print("=" * 70)

md = []
md.append("# AI Transcription Analysis — WER Comparison")
md.append("")
md.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
md.append("")

# ── Overview ──
md.append("## Overview")
md.append("")
md.append(f"- **Total audio files:** {len(df)}")
md.append(f"- **Human graders:** {len(all_graders)} ({', '.join(all_graders)})")
md.append(f"- **AI models evaluated:** {len(AI_MODELS)}")
md.append(f"- **Dual-graded audio files (both graders completed):** {has_both.sum()}")
md.append(f"- **Audio files with at least one human transcription:** {has_any.sum()}")
md.append("")

# ── Data availability ──
md.append("### Transcription Counts")
md.append("")
md.append("#### Human Graders")
md.append("")
md.append("| Grader | Transcriptions |")
md.append("|--------|---------------|")
for g in all_graders:
    md.append(f"| {g} | {len(grader_transcriptions[g])} |")
md.append("")

md.append("#### AI Models")
md.append("")
md.append("| Model | Transcriptions (out of {}) |".format(len(df)))
md.append("|-------|---------------------------|")
for model in AI_MODELS:
    ai_col = f"ai_{model}"
    n = df[ai_col].notna().sum()
    md.append(f"| {model} | {n} |")
md.append("")

# ── Part 1: Inter-Grader Agreement ──
md.append("---")
md.append("")
md.append("## Part 1: Inter-Grader Agreement")
md.append("")
md.append("Word Error Rate (WER) computed between the two human graders' transcriptions")
md.append("for each dual-graded audio file. Lower WER = higher agreement.")
md.append("")

md.append("### Summary Statistics")
md.append("")
md.append("| Metric | Value |")
md.append("|--------|-------|")
md.append(f"| N (dual-graded audios with WER) | {len(valid_ig_wer)} |")
md.append(f"| Mean WER | {valid_ig_wer.mean():.1%} |")
md.append(f"| Median WER | {valid_ig_wer.median():.1%} |")
md.append(f"| Std WER | {valid_ig_wer.std():.1%} |")
md.append(f"| Min WER | {valid_ig_wer.min():.1%} |")
md.append(f"| Max WER | {valid_ig_wer.max():.1%} |")
md.append(f"| Perfect agreement (WER=0) | {(valid_ig_wer == 0).sum()} ({(valid_ig_wer == 0).mean():.1%}) |")
md.append("")

md.append("### Pairwise Inter-Grader WER")
md.append("")
md.append("![Inter-Grader WER Heatmap](plots/v3_inter_grader_wer_heatmap.png)")
md.append("")

md.append("| Grader 1 | Grader 2 | Mean WER | Shared Audios |")
md.append("|----------|----------|----------|---------------|")
for g1 in all_graders:
    for g2 in all_graders:
        if g1 >= g2:
            continue
        val = ig_pairwise.loc[g1, g2]
        n = ig_overlap.loc[g1, g2]
        if pd.notna(val) and n > 0:
            md.append(f"| {g1} | {g2} | {val:.1%} | {n} |")
        elif n > 0:
            md.append(f"| {g1} | {g2} | — | {n} |")
md.append("")

# ── Part 2: Human vs AI WER ──
md.append("---")
md.append("")
md.append("## Part 2: Human vs AI Transcription WER")
md.append("")
md.append("WER computed with **human transcription as reference** and AI transcription as hypothesis.")
md.append("This measures how closely each AI model's transcription matches what the human grader heard.")
md.append("")

md.append("### Mean WER by AI Model (vs Human Transcription)")
md.append("")
md.append("| Rank | AI Model | Mean WER | Median WER | Std WER | N Comparisons |")
md.append("|------|----------|----------|------------|---------|---------------|")
for rank, model in enumerate(summary.index, 1):
    r = summary.loc[model]
    md.append(
        f"| {rank} | {model} | {r['mean_wer']:.1%} | {r['median_wer']:.1%} "
        f"| {r['std_wer']:.1%} | {int(r['n_comparisons'])} |"
    )
md.append("")

md.append("### Mean WER by AI Model (vs Reference Text)")
md.append("")
md.append("WER computed with the **original reference sentence** (what the student was supposed to read)")
md.append("as the reference, and the AI transcription as the hypothesis.")
md.append("")
md.append("| Rank | AI Model | Mean WER | Median WER | Std WER | N |")
md.append("|------|----------|----------|------------|---------|---|")
for rank, model in enumerate(ref_summary.index, 1):
    r = ref_summary.loc[model]
    md.append(
        f"| {rank} | {model} | {r['mean_wer']:.1%} | {r['median_wer']:.1%} "
        f"| {r['std_wer']:.1%} | {int(r['n'])} |"
    )
md.append("")

md.append("### Mean WER by Human Grader (vs Reference Text)")
md.append("")
md.append("For comparison: WER of each human grader's transcription against the reference text.")
md.append("")
md.append("| Grader | Mean WER | Median WER | Std WER | N |")
md.append("|--------|----------|------------|---------|---|")
for grader in human_ref_summary.index:
    r = human_ref_summary.loc[grader]
    md.append(
        f"| {grader} | {r['mean_wer']:.1%} | {r['median_wer']:.1%} "
        f"| {r['std_wer']:.1%} | {int(r['n'])} |"
    )
md.append("")

md.append("### Heatmap: AI Models x Human Graders")
md.append("")
md.append("![Human vs AI WER Heatmap](plots/v3_human_vs_ai_wer_heatmap.png)")
md.append("")

# ── Part 2b: Dual-graded WER comparison ──
md.append("### WER vs Reference — Dual-Graded Audios Only")
md.append("")
md.append(f"Filtering to the {len(dual_audios)} dual-graded audio files so all scorers are evaluated")
md.append("on the same set of audios, removing assignment bias.")
md.append("")
md.append("| Scorer | Type | Mean WER | Median WER | N |")
md.append("|--------|------|----------|------------|---|")
for (scorer, stype), row in dual_summary.iterrows():
    md.append(f"| {scorer} | {stype} | {row['mean_wer']:.1%} | {row['median_wer']:.1%} | {int(row['n'])} |")
md.append("")
md.append(f"**Overall mean WER (dual-graded):** Human = {dual_means.get('Human', float('nan')):.1%}, "
           f"AI = {dual_means.get('AI', float('nan')):.1%}")
md.append("")

# ── Part 3: Correlation heatmap ──
md.append("---")
md.append("")
md.append("## Part 3: Pairwise Correlation (All Scorers)")
md.append("")
md.append("For each scorer (human grader or AI model), we compute **words correct** — the number of")
md.append("words correctly matched (via jiwer alignment) between their transcription and the reference")
md.append("text. Then we compute **pairwise Pearson correlations** of these word-correct counts across")
md.append("shared audio files. Higher correlation = more agreement on student performance.")
md.append("")
md.append("![Combined Correlation Heatmap](plots/v3_combined_correlation_heatmap.png)")
md.append("")

md.append("### Reliable Graders + AI Models")
md.append("")
md.append("Excluding Abdullah, Ali, Dania, and Salman (low overlap or outlier scoring).")
md.append("")
md.append("![Reliable Graders Correlation Heatmap](plots/v3_reliable_correlation_heatmap.png)")
md.append("")

# Correlation table — human-human pairs
md.append("### Human-Human Correlations")
md.append("")
md.append("| Grader 1 | Grader 2 | Pearson r | Shared Audios |")
md.append("|----------|----------|-----------|---------------|")
for g1 in all_graders:
    for g2 in all_graders:
        if g1 >= g2:
            continue
        val = corr_matrix.loc[g1, g2]
        n = overlap_matrix.loc[g1, g2]
        if pd.notna(val) and n > 0:
            md.append(f"| {g1} | {g2} | {val:.3f} | {n} |")
        elif n > 0:
            md.append(f"| {g1} | {g2} | — | {n} |")
md.append("")

# Correlation table — human-AI pairs
md.append("### Human-AI Correlations")
md.append("")
md.append("| Human Grader | AI Model | Pearson r | Shared Audios |")
md.append("|-------------|----------|-----------|---------------|")
for g in all_graders:
    for m in AI_MODELS:
        val = corr_matrix.loc[g, m]
        n = overlap_matrix.loc[g, m]
        if pd.notna(val) and n >= 3:
            md.append(f"| {g} | {m} | {val:.3f} | {n} |")
md.append("")

# Mean correlations summary
md.append("### Mean Correlations by Group")
md.append("")

hh_corrs = []
for g1 in all_graders:
    for g2 in all_graders:
        if g1 < g2 and pd.notna(corr_matrix.loc[g1, g2]):
            hh_corrs.append(corr_matrix.loc[g1, g2])

ha_corrs = []
for g in all_graders:
    for m in AI_MODELS:
        if pd.notna(corr_matrix.loc[g, m]):
            ha_corrs.append(corr_matrix.loc[g, m])

aa_corrs = []
for m1 in AI_MODELS:
    for m2 in AI_MODELS:
        if m1 < m2 and pd.notna(corr_matrix.loc[m1, m2]):
            aa_corrs.append(corr_matrix.loc[m1, m2])

mean_hh_corr = np.mean(hh_corrs) if hh_corrs else np.nan
mean_ha_corr = np.mean(ha_corrs) if ha_corrs else np.nan
mean_aa_corr = np.mean(aa_corrs) if aa_corrs else np.nan

md.append("| Group | Mean Pearson r | N pairs |")
md.append("|-------|---------------|---------|")
if not np.isnan(mean_hh_corr):
    md.append(f"| Human-Human | {mean_hh_corr:.3f} | {len(hh_corrs)} |")
if not np.isnan(mean_ha_corr):
    md.append(f"| Human-AI | {mean_ha_corr:.3f} | {len(ha_corrs)} |")
if not np.isnan(mean_aa_corr):
    md.append(f"| AI-AI | {mean_aa_corr:.3f} | {len(aa_corrs)} |")
md.append("")

# ── Key findings ──
md.append("---")
md.append("")
md.append("## Key Findings")
md.append("")

best_model = summary.index[0]
best_wer = summary.loc[best_model, "mean_wer"]
worst_model = summary.index[-1]
worst_wer = summary.loc[worst_model, "mean_wer"]

md.append(f"1. **Best AI model (WER):** {best_model} with mean WER of {best_wer:.1%} vs human transcription")
md.append(f"2. **Worst AI model (WER):** {worst_model} with mean WER of {worst_wer:.1%} vs human transcription")
md.append(f"3. **Inter-grader agreement:** Mean WER between paired human graders is {valid_ig_wer.mean():.1%} "
           f"(median {valid_ig_wer.median():.1%})")
if not np.isnan(mean_hh_corr):
    md.append(f"4. **Human-Human correlation (words correct):** r = {mean_hh_corr:.3f}")
if not np.isnan(mean_ha_corr):
    md.append(f"5. **Human-AI correlation (words correct):** r = {mean_ha_corr:.3f}")
if not np.isnan(mean_aa_corr):
    md.append(f"6. **AI-AI correlation (words correct):** r = {mean_aa_corr:.3f}")
md.append("")

md_content = "\n".join(md)
with open(OUT_MARKDOWN, "w") as f:
    f.write(md_content)
print(f"Saved markdown to: {OUT_MARKDOWN}")

print("\nDone!")
# %%
