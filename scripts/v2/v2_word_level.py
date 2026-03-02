# %%
"""
v2_word_level.py — Word-by-word correct/incorrect comparison (v2 data only).

For each audio file and each reference word, flags the word as correct or
incorrect from the human grader's transcription (jiwer alignment) and from
the AI pronunciation assessment (Azure AccuracyScore, strict threshold).

Outputs:
  - data/clean/v2_word_level_grading.csv      (per-word, per-scorer rows)
  - data/clean/v2_word_level_comparison.csv    (wide: human_correct vs ai_correct)
  - data/clean/v2_word_level_summary.csv       (aggregated per-word stats)
  - plots/v2_word_level_correlation_heatmap.png (pairwise correlation matrix)
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from jiwer import process_words
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v1"))
from helper_functions import parse_ai_cell, normalize_word

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

V2_SCORES = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_merged_scores.csv")
AI_RESPONSES = os.path.join(PROJECT_DIR, "data", "v1", "clean", "ai_responses_extracted.csv")

OUTPUT_GRADING = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_word_level_grading.csv")
OUTPUT_COMPARISON = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_word_level_comparison.csv")
OUTPUT_SUMMARY = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_word_level_summary.csv")
OUTPUT_HEATMAP = os.path.join(PROJECT_DIR, "plots", "v2", "v2_word_level_correlation_heatmap.png")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def clean_text(text):
    """Normalize text for word-level comparison (same as v2_analysis.py)."""
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_per_word_alignment(reference, transcription):
    """Run jiwer alignment and return per-word scores for each reference word.

    Returns list of dicts: [{word, position, score}]
    where score = 2 (correct/equal) or 0 (substituted/deleted).
    """
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(transcription)
    ref_words = ref_clean.split()

    if not ref_clean or not hyp_clean:
        return [{"word": w, "position": i, "score": 0}
                for i, w in enumerate(ref_words)]

    result = process_words(ref_clean, hyp_clean)
    scores = [0] * len(ref_words)  # default: incorrect

    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                scores[i] = 2  # correct

    return [{"word": ref_words[i], "position": i, "score": scores[i]}
            for i in range(len(ref_words))]


# ──────────────────────────────────────────────
# Process v2 human transcriptions
# ──────────────────────────────────────────────
def process_v2_human():
    """Extract per-word correct/incorrect from human transcriptions via jiwer."""
    print("Processing v2 human transcriptions...")
    df = pd.read_csv(V2_SCORES)
    rows = []

    for _, r in df.iterrows():
        ref = r["question_clean"]
        human_trans = r["Human Transcription"]
        if pd.isna(human_trans) or str(human_trans).strip() == "":
            continue

        word_scores = get_per_word_alignment(ref, human_trans)
        for ws in word_scores:
            rows.append({
                "audio_file_name": r["audio_file_name"],
                "profile_id": r["profile_id"],
                "pre_or_post": r["pre_or_post"],
                "question_clean": ref,
                "word": ws["word"],
                "word_position": ws["position"],
                "total_ref_words": len(clean_text(ref).split()),
                "scorer": r["grader"],
                "score": ws["score"],
                "score_binary": 1 if ws["score"] >= 2 else 0,
            })

    print(f"  Human word-level rows: {len(rows)}")
    return rows


# ──────────────────────────────────────────────
# Process v2 AI pronunciation scores
# ──────────────────────────────────────────────
def process_v2_ai_pronunciation():
    """Extract per-word AI pronunciation scores from submitted_feedback_json.

    Uses strict threshold: only normalized_rounded == 2 counts as correct.
    """
    print("Processing v2 AI pronunciation scores...")
    ai_df = pd.read_csv(AI_RESPONSES)
    v2_df = pd.read_csv(V2_SCORES)

    v2_dedup = v2_df.drop_duplicates(subset=["audio_file_name"]).copy()
    v2_dedup["profile_id"] = v2_dedup["profile_id"].astype(str)
    ai_df["profile_id"] = ai_df["profile_id"].astype(str)

    v2_dedup["question_clean_norm"] = v2_dedup["question_clean"].str.strip().str.replace(r"\s+", " ", regex=True)
    ai_df["question_clean_norm"] = ai_df["question_clean"].str.strip().str.replace(r"\s+", " ", regex=True)

    merged = v2_dedup.merge(
        ai_df[["profile_id", "question_clean_norm", "pre_or_post", "submitted_feedback_json"]],
        on=["profile_id", "question_clean_norm", "pre_or_post"],
        how="left",
    ).drop_duplicates(subset=["audio_file_name"])

    rows = []
    for _, r in merged.iterrows():
        fbk = r.get("submitted_feedback_json")
        if pd.isna(fbk):
            continue

        parsed = parse_ai_cell(fbk)
        if not parsed:
            continue

        ref = clean_text(r["question_clean"])
        ref_words = ref.split()

        for word_key, data in parsed.items():
            if not isinstance(data, dict):
                continue
            score = data.get("normalized_rounded")
            if score is None:
                continue

            norm_key = normalize_word(word_key)
            position = None
            for i, rw in enumerate(ref_words):
                if normalize_word(rw) == norm_key:
                    position = i
                    break

            rows.append({
                "audio_file_name": r["audio_file_name"],
                "profile_id": r["profile_id"],
                "pre_or_post": r["pre_or_post"],
                "question_clean": r["question_clean"],
                "word": norm_key,
                "word_position": position,
                "total_ref_words": len(ref_words),
                "scorer": "AI_pronunciation",
                "score": int(score),
                "score_binary": 1 if score >= 2 else 0,
            })

    print(f"  AI pronunciation word-level rows: {len(rows)}")
    return rows


# ──────────────────────────────────────────────
# Build comparison CSV (wide format)
# ──────────────────────────────────────────────
def build_comparison(grading_df):
    """Create wide-format comparison: one row per (audio, word) with human vs AI."""
    print("Building human vs AI comparison...")

    human_df = grading_df[grading_df["scorer"] != "AI_pronunciation"].copy()
    ai_df = grading_df[grading_df["scorer"] == "AI_pronunciation"].copy()

    # For human: take first grader's score per (audio, word)
    # (if dual-graded, use first encountered; could average later)
    human_agg = human_df.groupby(["audio_file_name", "word"]).agg(
        human_correct=("score_binary", "first"),
        human_grader=("scorer", "first"),
        profile_id=("profile_id", "first"),
        pre_or_post=("pre_or_post", "first"),
        question_clean=("question_clean", "first"),
        word_position=("word_position", "first"),
        total_ref_words=("total_ref_words", "first"),
    ).reset_index()

    ai_agg = ai_df.groupby(["audio_file_name", "word"]).agg(
        ai_correct=("score_binary", "first"),
    ).reset_index()

    comparison = human_agg.merge(ai_agg, on=["audio_file_name", "word"], how="inner")
    comparison["agree"] = (comparison["human_correct"] == comparison["ai_correct"]).astype(int)

    print(f"  Comparison rows: {len(comparison)}")
    print(f"  Agreement rate: {comparison['agree'].mean() * 100:.1f}%")

    return comparison


# ──────────────────────────────────────────────
# Generate per-word summary
# ──────────────────────────────────────────────
def generate_summary(grading_df):
    """Aggregate per-word statistics across human vs AI."""
    print("Generating word-level summary...")

    human_df = grading_df[grading_df["scorer"] != "AI_pronunciation"]
    ai_df = grading_df[grading_df["scorer"] == "AI_pronunciation"]

    h_summary = human_df.groupby("word").agg(
        n_human=("score_binary", "count"),
        human_correct_pct=("score_binary", "mean"),
    ).reset_index()
    h_summary["human_correct_pct"] = (h_summary["human_correct_pct"] * 100).round(1)

    a_summary = ai_df.groupby("word").agg(
        n_ai=("score_binary", "count"),
        ai_correct_pct=("score_binary", "mean"),
    ).reset_index()
    a_summary["ai_correct_pct"] = (a_summary["ai_correct_pct"] * 100).round(1)

    summary = h_summary.merge(a_summary, on="word", how="outer")

    # Per-word agreement
    human_pairs = human_df.groupby(["audio_file_name", "word"])["score_binary"].first().reset_index()
    human_pairs.columns = ["audio_file_name", "word", "human_binary"]
    ai_pairs = ai_df.groupby(["audio_file_name", "word"])["score_binary"].first().reset_index()
    ai_pairs.columns = ["audio_file_name", "word", "ai_binary"]

    paired = human_pairs.merge(ai_pairs, on=["audio_file_name", "word"], how="inner")
    paired["agree"] = (paired["human_binary"] == paired["ai_binary"]).astype(int)

    agree_by_word = paired.groupby("word").agg(
        n_paired=("agree", "count"),
        agreement_pct=("agree", "mean"),
    ).reset_index()
    agree_by_word["agreement_pct"] = (agree_by_word["agreement_pct"] * 100).round(1)

    summary = summary.merge(agree_by_word, on="word", how="left")
    summary["human_minus_ai"] = (summary["human_correct_pct"] - summary["ai_correct_pct"]).round(1)
    summary = summary.sort_values("agreement_pct", ascending=True)

    return summary


# ──────────────────────────────────────────────
# Print word-level examples
# ──────────────────────────────────────────────
def print_examples(comparison_df):
    """Print a few audio files showing word-level human vs AI flags."""
    print("\n" + "=" * 70)
    print("Word-level examples (Human vs AI pronunciation)")
    print("=" * 70)

    # Pick 3 diverse examples: one high agreement, one medium, one low
    audio_agree = comparison_df.groupby("audio_file_name")["agree"].mean()
    examples = []

    # High agreement
    high = audio_agree[audio_agree >= 0.9]
    if len(high) > 0:
        examples.append(("HIGH AGREEMENT", high.sample(1, random_state=42).index[0]))

    # Medium agreement
    med = audio_agree[(audio_agree >= 0.5) & (audio_agree < 0.9)]
    if len(med) > 0:
        examples.append(("MEDIUM AGREEMENT", med.sample(1, random_state=42).index[0]))

    # Low agreement
    low = audio_agree[audio_agree < 0.5]
    if len(low) > 0:
        examples.append(("LOW AGREEMENT", low.sample(1, random_state=42).index[0]))

    for label, audio in examples:
        rows = comparison_df[comparison_df["audio_file_name"] == audio].sort_values("word_position")
        ref = rows.iloc[0]["question_clean"]
        agree_pct = rows["agree"].mean() * 100
        print(f"\n--- {label} ({agree_pct:.0f}% word agreement) ---")
        print(f"Audio: {audio}")
        print(f"Reference: {ref}")
        print()

        words = rows["word"].tolist()
        h_flags = ["  \u2713" if c == 1 else "  \u2717" for c in rows["human_correct"]]
        a_flags = ["  \u2713" if c == 1 else "  \u2717" for c in rows["ai_correct"]]

        # Format as aligned columns
        col_width = max(max(len(w) for w in words) + 2, 6)
        header = "".join(w.ljust(col_width) for w in words)
        h_line = "".join(f.ljust(col_width) for f in h_flags)
        a_line = "".join(f.ljust(col_width) for f in a_flags)

        print(f"  Words:  {header}")
        print(f"  Human:  {h_line}")
        print(f"  AI:     {a_line}")


# ──────────────────────────────────────────────
# Correlation matrix heatmap
# ──────────────────────────────────────────────
def create_correlation_heatmap(grading_df):
    """Build pairwise correlation matrix of total words correct per audio across scorers."""
    print("\nComputing pairwise correlations for heatmap...")

    # Compute total words correct per audio per scorer
    totals = grading_df.groupby(["audio_file_name", "scorer"])["score_binary"].sum().reset_index()
    totals.columns = ["audio_file_name", "scorer", "words_correct"]

    # Pivot to wide format
    score_matrix = totals.pivot_table(
        index="audio_file_name", columns="scorer", values="words_correct"
    )

    # Only keep reliable graders (>95% completion) + AI
    RELIABLE_GRADERS = ["Amna", "Dania", "Rehma", "Rukhshan", "Semal", "AI_pronunciation"]
    valid_scorers = [c for c in RELIABLE_GRADERS if c in score_matrix.columns]
    score_matrix = score_matrix[valid_scorers]

    print(f"  Score matrix: {score_matrix.shape[0]} audios x {score_matrix.shape[1]} scorers")
    print(f"  Scorers (>95% completion + AI): {valid_scorers}")

    # Compute pairwise Pearson correlations
    corr_matrix = pd.DataFrame(np.nan, index=valid_scorers, columns=valid_scorers)
    overlap_matrix = pd.DataFrame(0, index=valid_scorers, columns=valid_scorers, dtype=int)

    for i, g1 in enumerate(valid_scorers):
        for j, g2 in enumerate(valid_scorers):
            if i == j:
                corr_matrix.loc[g1, g2] = 1.0
                overlap_matrix.loc[g1, g2] = int(score_matrix[g1].notna().sum())
                continue
            mask = score_matrix[g1].notna() & score_matrix[g2].notna()
            n_shared = mask.sum()
            overlap_matrix.loc[g1, g2] = n_shared
            if n_shared >= 3:
                r, _ = pearsonr(score_matrix.loc[mask, g1], score_matrix.loc[mask, g2])
                corr_matrix.loc[g1, g2] = r

    corr_matrix = corr_matrix.astype(float)

    print(f"\n  Correlation matrix:")
    print(corr_matrix.round(3).to_string())

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="YlGnBu_r",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(
        "Pair-wise correlation of words correct per audio\n"
        "(Reliable graders >95% completion + AI pronunciation)",
        fontsize=13,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_HEATMAP), exist_ok=True)
    plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches="tight")
    print(f"\n  Saved heatmap to: {OUTPUT_HEATMAP}")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Word-level correct/incorrect comparison (v2 only)")
    print("=" * 70)

    # Collect rows: human + AI pronunciation only
    all_rows = []
    all_rows.extend(process_v2_human())
    all_rows.extend(process_v2_ai_pronunciation())

    grading_df = pd.DataFrame(all_rows)
    print(f"\nTotal word-level rows: {len(grading_df)}")
    n_human = (grading_df["scorer"] != "AI_pronunciation").sum()
    n_ai = (grading_df["scorer"] == "AI_pronunciation").sum()
    print(f"  Human: {n_human}")
    print(f"  AI pronunciation: {n_ai}")

    # Save grading CSV
    grading_df.to_csv(OUTPUT_GRADING, index=False)
    print(f"\nSaved word-level grading to: {OUTPUT_GRADING}")

    # Build and save comparison CSV
    comparison = build_comparison(grading_df)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    print(f"Saved comparison to: {OUTPUT_COMPARISON}")

    # Generate and save summary
    summary = generate_summary(grading_df)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"Saved summary to: {OUTPUT_SUMMARY}")

    # Print per-word stats
    print("\n" + "=" * 70)
    print("Per-word agreement (lowest first):")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Print visual examples
    print_examples(comparison)

    # Generate correlation heatmap
    create_correlation_heatmap(grading_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
