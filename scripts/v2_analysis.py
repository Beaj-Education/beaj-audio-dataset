# %%
"""
v2_analysis.py — Pair-wise correlation heatmap from graded_v2 transcription data.

Reads human transcriptions from graded_v2/ CSVs, derives total scores by
comparing each transcription to the reference text (word match count),
loads AI total scores from ai_responses_extracted.csv, and generates a
pair-wise correlation heatmap across all graders + AI.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import process_words

# Add scripts dir to path so we can import helper_functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_functions import parse_ai_cell, normalize_word

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GRADED_V2_DIR = os.path.join(PROJECT_DIR, "data", "graded_v2")
AI_RESPONSES_PATH = os.path.join(PROJECT_DIR, "data", "clean", "ai_responses_extracted.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "clean", "v2_merged_scores.csv")
OUTPUT_PLOT = os.path.join(PROJECT_DIR, "plots", "v2_pairwise_correlation_heatmap.png")

# ──────────────────────────────────────────────
# Step 1: Load & combine graded_v2 CSVs
# ──────────────────────────────────────────────
print("=" * 70)
print("Step 1: Loading graded_v2 CSVs")
print("=" * 70)

dataframes = []
for fname in sorted(os.listdir(GRADED_V2_DIR)):
    if not fname.endswith(".csv"):
        continue
    grader_name = fname.split(" - ")[0].strip()
    fpath = os.path.join(GRADED_V2_DIR, fname)
    df = pd.read_csv(fpath)
    # Drop unnamed columns (CSV export artifacts)
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    df["grader"] = grader_name
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

# Extract metadata from audio_file_name
# e.g. grade1_profile28390_q3_QK_20251122_101834.mp3
df_all["profile_id"] = df_all["audio_file_name"].str.extract(r"profile(\d+)")[0]
df_all["pre_or_post"] = np.where(df_all["audio_file_name"].str.contains("_LP_"), "pre", "post")

# Normalize question text
df_all["question_clean"] = df_all["Question"].str.strip().str.replace(r"\s+", " ", regex=True)

# Filter to rows with transcriptions
df_transcribed = df_all[df_all["Human Transcription"].notna() & (df_all["Human Transcription"].str.strip() != "")].copy()

print(f"Total rows across all graders: {len(df_all)}")
print(f"Rows with transcriptions: {len(df_transcribed)}")
print(f"Unique audio files with transcriptions: {df_transcribed['audio_file_name'].nunique()}")
print(f"Graders with transcriptions: {sorted(df_transcribed['grader'].unique())}")
for g, grp in df_transcribed.groupby("grader"):
    print(f"  {g}: {len(grp)} transcriptions")

# %%
# ──────────────────────────────────────────────
# Step 2: Derive human total scores from transcriptions
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Deriving human scores from transcriptions")
print("=" * 70)


def clean_text(text):
    """Normalize text for word-level comparison."""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s']", "", s)  # keep apostrophes for words like "it's"
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_words_correct(reference, transcription):
    """Count correctly matched words using jiwer alignment."""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(transcription)

    if not ref_clean or not hyp_clean:
        return 0, len(ref_clean.split()) if ref_clean else 0

    result = process_words(ref_clean, hyp_clean)
    hits = result.hits
    total_ref_words = len(ref_clean.split())
    return hits, total_ref_words


df_transcribed["human_words_correct"], df_transcribed["total_reference_words"] = zip(
    *df_transcribed.apply(
        lambda row: compute_words_correct(row["question_clean"], row["Human Transcription"]),
        axis=1,
    )
)

print(f"Score distribution (words correct):")
print(df_transcribed["human_words_correct"].describe().round(2))
print(f"\nReference word counts: {sorted(df_transcribed['total_reference_words'].unique())}")

# %%
# ──────────────────────────────────────────────
# Step 3: Load AI responses & extract AI total scores
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Loading AI responses and extracting scores")
print("=" * 70)

df_ai = pd.read_csv(AI_RESPONSES_PATH)
df_ai["question_clean"] = df_ai["question_clean"].str.strip().str.replace(r"\s+", " ", regex=True)
df_ai["profile_id"] = df_ai["profile_id"].astype(str)
df_transcribed["profile_id"] = df_transcribed["profile_id"].astype(str)

print(f"AI responses loaded: {len(df_ai)} rows, {df_ai['profile_id'].nunique()} profiles")


def extract_ai_total_score(feedback_json_str):
    """Extract total AI score (sum of normalized_rounded) from feedback JSON."""
    parsed = parse_ai_cell(feedback_json_str)
    if not parsed:
        return np.nan, 0
    total = 0
    n_words = 0
    for word, data in parsed.items():
        if isinstance(data, dict):
            score = data.get("normalized_rounded")
            if score is not None:
                total += float(score)
                n_words += 1
    return total, n_words


df_ai["ai_total_score"], df_ai["ai_n_words"] = zip(
    *df_ai["submitted_feedback_json"].apply(extract_ai_total_score)
)

# Also compute AI words correct (score == 2 means correct)
def extract_ai_words_correct(feedback_json_str):
    """Count words where AI gave score 2 (fully correct)."""
    parsed = parse_ai_cell(feedback_json_str)
    if not parsed:
        return 0
    correct = 0
    for word, data in parsed.items():
        if isinstance(data, dict):
            score = data.get("normalized_rounded")
            if score is not None and float(score) == 2:
                correct += 1
    return correct


df_ai["ai_words_correct"] = df_ai["submitted_feedback_json"].apply(extract_ai_words_correct)

# Merge: graded_v2 rows → AI responses
# Match on profile_id + question_clean + pre_or_post
df_merged = df_transcribed.merge(
    df_ai[["profile_id", "question_clean", "pre_or_post", "ai_transcription",
            "ai_total_score", "ai_n_words", "ai_words_correct"]],
    on=["profile_id", "question_clean", "pre_or_post"],
    how="left",
)

# Some profiles may have multiple AI submissions for same question+pre_or_post
# Keep the one with the most words scored (most complete)
df_merged = df_merged.sort_values("ai_n_words", ascending=False).drop_duplicates(
    subset=["audio_file_name", "grader"], keep="first"
)

matched = df_merged["ai_total_score"].notna().sum()
print(f"Matched {matched}/{len(df_merged)} transcribed rows to AI responses")
print(f"Unmatched: {len(df_merged) - matched}")

# %%
# ──────────────────────────────────────────────
# Step 4: Build score matrix & compute correlations
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4: Building score matrix and computing correlations")
print("=" * 70)

# For humans: score = words_correct
# For AI: score = ai_words_correct (words with normalized_rounded == 2)
# This makes the scales more comparable (both count "correct" words)

# Pivot human scores: audio_file_name × grader → words_correct
human_pivot = df_merged.pivot_table(
    index="audio_file_name",
    columns="grader",
    values="human_words_correct",
    aggfunc="first",
)

# AI scores: one per audio (take from first grader's merge)
ai_scores = df_merged.drop_duplicates(subset=["audio_file_name"])[
    ["audio_file_name", "ai_words_correct"]
].set_index("audio_file_name")

# Combine into one matrix
score_matrix = human_pivot.copy()
score_matrix["AI"] = ai_scores["ai_words_correct"]

# Only keep graders with enough data
min_transcriptions = 10
valid_graders = [c for c in score_matrix.columns if score_matrix[c].notna().sum() >= min_transcriptions]
score_matrix = score_matrix[valid_graders]

print(f"Score matrix: {score_matrix.shape[0]} audio files × {score_matrix.shape[1]} graders")
print(f"Graders included (>= {min_transcriptions} transcriptions): {valid_graders}")
print(f"\nNon-NaN counts per grader:")
for col in score_matrix.columns:
    print(f"  {col}: {score_matrix[col].notna().sum()}")

# Compute pair-wise Pearson correlations
n_graders = len(valid_graders)
corr_matrix = pd.DataFrame(np.nan, index=valid_graders, columns=valid_graders)
overlap_matrix = pd.DataFrame(0, index=valid_graders, columns=valid_graders, dtype=int)

for i, g1 in enumerate(valid_graders):
    for j, g2 in enumerate(valid_graders):
        if i == j:
            corr_matrix.loc[g1, g2] = 1.0
            overlap_matrix.loc[g1, g2] = int(score_matrix[g1].notna().sum())
            continue
        # Get shared audio files
        mask = score_matrix[g1].notna() & score_matrix[g2].notna()
        n_shared = mask.sum()
        overlap_matrix.loc[g1, g2] = n_shared
        if n_shared >= 3:  # need at least 3 points for meaningful correlation
            r, p = pearsonr(score_matrix.loc[mask, g1], score_matrix.loc[mask, g2])
            corr_matrix.loc[g1, g2] = r

corr_matrix = corr_matrix.astype(float)

print(f"\nPair-wise overlap (shared audio files with both scores):")
print(overlap_matrix.to_string())
print(f"\nCorrelation matrix:")
print(corr_matrix.round(3).to_string())

# %%
# ──────────────────────────────────────────────
# Step 5: Plot heatmap
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 5: Generating heatmap")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 6))

# Create annotation labels that show correlation + overlap count
annot_labels = corr_matrix.copy().astype(str)
for i, g1 in enumerate(valid_graders):
    for j, g2 in enumerate(valid_graders):
        r_val = corr_matrix.loc[g1, g2]
        n_val = overlap_matrix.loc[g1, g2]
        if pd.isna(r_val):
            annot_labels.loc[g1, g2] = f"n={n_val}"
        elif i == j:
            annot_labels.loc[g1, g2] = f"{r_val:.2f}"
        else:
            annot_labels.loc[g1, g2] = f"{r_val:.2f}\n(n={n_val})"

sns.heatmap(
    corr_matrix,
    annot=annot_labels,
    fmt="",
    cmap="YlGnBu_r",
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson r"},
    ax=ax,
)
ax.set_title("Pair-wise correlation of TOTAL scores\n(word match count)", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved heatmap to: {OUTPUT_PLOT}")
plt.close()

# %%
# ──────────────────────────────────────────────
# Save merged CSV
# ──────────────────────────────────────────────
save_cols = [
    "audio_file_name", "grader", "profile_id", "pre_or_post",
    "question_clean", "Human Transcription", "ai_transcription",
    "human_words_correct", "total_reference_words",
    "ai_total_score", "ai_words_correct", "ai_n_words",
]
existing_cols = [c for c in save_cols if c in df_merged.columns]
df_merged[existing_cols].to_csv(OUTPUT_CSV, index=False)
print(f"Saved merged scores to: {OUTPUT_CSV}")

print("\nDone!")
# %%
