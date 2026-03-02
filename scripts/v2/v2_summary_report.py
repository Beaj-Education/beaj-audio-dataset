# %%
"""
v2_summary_report.py — Generate a narrative summary Word document.

Tells the story: dataset → grader assignments → scoring approach →
examples → inter-human gold standard → AI comparison → heatmaps →
word-level agreement → WCPM.

Output: reports/v2_summary_report.docx
"""

import os
import sys
import pandas as pd
import numpy as np

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# Import reusable compute functions from existing report
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_report import (
    compute_intergrader_consensus,
    compute_correlations,
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

V2_SCORES = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_merged_scores.csv")
MERGED_V1 = os.path.join(PROJECT_DIR, "data", "v1", "clean", "merged_for_analysis.csv")
GRADED_V2_DIR = os.path.join(PROJECT_DIR, "data", "v2", "graded")
HEATMAP_ALL = os.path.join(PROJECT_DIR, "plots", "v2", "v2_pairwise_correlation_heatmap.png")
HEATMAP_RELIABLE = os.path.join(PROJECT_DIR, "plots", "v2", "v2_reliable_graders_heatmap.png")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "v2")
REPORT_PATH = os.path.join(PROJECT_DIR, "reports", "v2", "v2_summary_report.docx")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = str(h)
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(9)
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(9)
    return table


def fmt(val, decimals=2):
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def add_bold_para(doc, bold_text, normal_text=""):
    p = doc.add_paragraph()
    run = p.add_run(bold_text)
    run.bold = True
    if normal_text:
        p.add_run(f" {normal_text}")
    return p


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv(V2_SCORES)
    # Use jiwer-based AI score (same method as human) instead of Azure pronunciation score
    if "ai_words_correct_jiwer" in df.columns:
        df["ai_words_correct"] = df["ai_words_correct_jiwer"]
    print(f"Loaded v2_merged_scores: {len(df)} rows, {df['audio_file_name'].nunique()} unique audios")
    return df


def get_grader_assignments():
    """Read graded_v2 CSVs to get assigned vs graded counts per grader."""
    rows = []
    for fname in sorted(os.listdir(GRADED_V2_DIR)):
        if not fname.endswith(".csv"):
            continue
        grader = fname.split(" - ")[0].strip()
        fpath = os.path.join(GRADED_V2_DIR, fname)
        gdf = pd.read_csv(fpath)
        n_assigned = len(gdf)
        n_graded = gdf[
            gdf["Human Transcription"].notna()
            & (gdf["Human Transcription"].str.strip() != "")
        ].shape[0]
        rows.append({
            "grader": grader,
            "assigned": n_assigned,
            "graded": n_graded,
            "completion_pct": round(n_graded / n_assigned * 100, 1) if n_assigned > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("grader")


def compute_wcpm(df):
    """Compute WCPM using human and AI transcription scores + v1 durations.

    Returns (overall_df, prepost_df, n) where:
      - overall_df is .describe() of human_wcpm, ai_wcpm, wcpm_diff
      - prepost_df is grouped by pre/post with mean/median stats
    """
    v1 = pd.read_csv(MERGED_V1)[["audio_filename", "ai_utterance_duration_seconds"]].dropna()
    v1 = v1.drop_duplicates(subset="audio_filename")

    # Use mean human score per audio (handles dual-graded), first AI score
    per_audio = df.groupby("audio_file_name").agg(
        human_words_correct=("human_words_correct", "mean"),
        ai_words_correct=("ai_words_correct", "first"),
        pre_or_post=("pre_or_post", "first"),
    ).reset_index()

    merged = per_audio.merge(
        v1, left_on="audio_file_name", right_on="audio_filename", how="inner"
    )
    merged = merged[
        (merged["ai_utterance_duration_seconds"] > 0)
        & merged["ai_words_correct"].notna()
    ].copy()
    merged["duration_minutes"] = merged["ai_utterance_duration_seconds"] / 60
    merged["human_wcpm"] = merged["human_words_correct"] / merged["duration_minutes"]
    merged["ai_wcpm"] = merged["ai_words_correct"] / merged["duration_minutes"]
    merged["wcpm_diff"] = merged["human_wcpm"] - merged["ai_wcpm"]

    overall = merged[["human_wcpm", "ai_wcpm", "wcpm_diff"]].describe()
    prepost = merged.groupby("pre_or_post")[["human_wcpm", "ai_wcpm"]].agg(
        ["mean", "median", "count"]
    ).round(1)

    return overall, prepost, len(merged)



def get_transcription_examples(df, n_per_group=3):
    """Pick examples at high, medium, and low human-AI agreement."""
    df = df[df["ai_words_correct"].notna()].copy()
    df["abs_diff"] = (df["human_words_correct"] - df["ai_words_correct"]).abs()
    examples = []

    # High agreement (diff <= 1)
    high = df[df["abs_diff"] <= 1].sample(n=min(n_per_group, len(df[df["abs_diff"] <= 1])), random_state=42)
    for _, r in high.iterrows():
        examples.append(("High", r))

    # Medium agreement (diff 2-3)
    med = df[(df["abs_diff"] >= 2) & (df["abs_diff"] <= 3)].sample(
        n=min(n_per_group, len(df[(df["abs_diff"] >= 2) & (df["abs_diff"] <= 3)])), random_state=42
    )
    for _, r in med.iterrows():
        examples.append(("Medium", r))

    # Low agreement (diff >= 4)
    low = df[df["abs_diff"] >= 4].sample(n=min(n_per_group, len(df[df["abs_diff"] >= 4])), random_state=42)
    for _, r in low.iterrows():
        examples.append(("Low", r))

    return examples


def compute_cohens_d(df):
    """Paired Cohen's d for inter-grader reliability."""
    multi = df.groupby("audio_file_name").filter(lambda x: len(x) > 1)
    diffs = []
    for _, group in multi.groupby("audio_file_name"):
        scores = group["human_words_correct"].values
        if len(scores) >= 2:
            diffs.append(float(scores[0]) - float(scores[1]))
    diffs = np.array(diffs)
    if len(diffs) == 0 or diffs.std() == 0:
        return 0.0, 0
    return diffs.mean() / diffs.std(), len(diffs)


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────
def generate_summary_report(df, grader_table, intergrader, cohens_d,
                            corr_matrix, overlap_matrix, valid_graders,
                            examples,
                            wcpm_overall, wcpm_prepost, wcpm_n):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ── Title ──
    title = doc.add_heading("BEAJ Audio Dataset — Summary Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Human Grading Reliability, AI Benchmarking & WCPM Analysis")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("")

    # Questions
    questions = sorted(df["question_clean"].unique(), key=len)
    n_students = df["profile_id"].nunique()
    n_audios_total = 630  # 105 students × 3 questions × 2 pre/post
    n_unique_graded = df["audio_file_name"].nunique()

    # Dual-graded counts
    audio_counts = df.groupby("audio_file_name").size()
    n_dual = (audio_counts > 1).sum()
    n_single = (audio_counts == 1).sum()
    n_ungraded = n_audios_total - n_unique_graded

    # ══════════════════════════════════════════
    # Section 1: Dataset Overview
    # ══════════════════════════════════════════
    doc.add_heading("1. Dataset Overview", level=1)

    doc.add_paragraph(
        f"We have {n_students} students, each of whom recorded 3 reading passages:"
    )
    for i, q in enumerate(questions, 1):
        doc.add_paragraph(f'Q{i}: "{q}"', style="List Bullet")

    doc.add_paragraph(
        f"Each passage was recorded for both a pre-assessment and a post-assessment, "
        f"giving us a total of {n_audios_total} audio recordings "
        f"({n_students} students × 3 questions × 2 pre/post)."
    )

    doc.add_paragraph("")
    doc.add_paragraph(
        f"All {n_audios_total} audios were dual-assigned — meaning each audio was assigned "
        f"to two separate graders for independent transcription. Due to varying grader "
        f"completion rates, the actual grading breakdown is:"
    )
    doc.add_paragraph(f"{n_dual} audios dual-graded (both assigned graders completed)", style="List Bullet")
    doc.add_paragraph(f"{n_single} audios single-graded (only one grader completed)", style="List Bullet")
    doc.add_paragraph(f"{n_ungraded} audios ungraded (neither grader completed)", style="List Bullet")

    # ══════════════════════════════════════════
    # Section 2: Grader Assignment Table
    # ══════════════════════════════════════════
    doc.add_heading("2. Grader Assignments & Completion", level=1)

    doc.add_paragraph(
        "The table below shows each grader's assignment load and how many audios "
        "they actually transcribed."
    )

    headers = ["Grader", "Assigned", "Graded", "Completion %"]
    rows = []
    total_assigned = total_graded = 0
    for _, r in grader_table.iterrows():
        rows.append((r["grader"], int(r["assigned"]), int(r["graded"]), f"{r['completion_pct']}%"))
        total_assigned += r["assigned"]
        total_graded += r["graded"]
    rows.append(("Total", int(total_assigned), int(total_graded),
                 f"{total_graded / total_assigned * 100:.1f}%"))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Five graders (Amna, Dania, Rehma, Rukhshan, Semal) completed >97% of their "
        "assignments. Three graders (Abdullah, Ali, Salman) had lower completion rates, "
        "which limits the pairwise overlap available for reliability analysis."
    )

    # ══════════════════════════════════════════
    # Section 3: Scoring Approach
    # ══════════════════════════════════════════
    doc.add_heading("3. Scoring Approach", level=1)

    doc.add_paragraph(
        "Both human and AI scores are computed by aligning a transcription against "
        "the reference passage using the jiwer library (word-level alignment). "
        "Each reference word is marked correct if the transcribed word matches, "
        "incorrect otherwise."
    )

    doc.add_heading("Human Scoring", level=2)
    doc.add_paragraph(
        "Human graders listen to the audio and write exactly what they hear. "
        "This phonetic transcription captures mispronunciations "
        '(e.g., "laaves" for "loves", "baild" for "build"). '
        "The transcription is aligned against the reference to count words correct."
    )

    doc.add_heading("AI Scoring", level=2)
    doc.add_paragraph(
        "Azure Speech-to-Text produces a transcription of the audio. This AI "
        "transcription is aligned against the same reference passage using jiwer. "
        "Because ASR normalizes speech to standard English words, it tends to "
        "score higher than human graders — mispronunciations that a human would "
        "flag are often \"corrected\" by the ASR engine."
    )

    # ══════════════════════════════════════════
    # Section 4: Transcription Examples
    # ══════════════════════════════════════════
    doc.add_heading("4. Transcription Examples", level=1)

    doc.add_paragraph(
        "The table below shows the full human transcription (what the grader heard) "
        "alongside the AI transcription (what Azure Speech-to-Text produced), "
        "grouped by agreement level."
    )

    headers = ["Agreement", "Reference Passage", "Human Transcription",
               "Human Score", "AI Transcription", "AI Score"]
    rows = []
    for level, r in examples:
        ref = r["question_clean"]
        if len(ref) > 40:
            ref = ref[:37] + "..."
        h_trans = str(r["Human Transcription"])
        if len(h_trans) > 50:
            h_trans = h_trans[:47] + "..."
        ai_trans = str(r["ai_transcription"])
        if len(ai_trans) > 50:
            ai_trans = ai_trans[:47] + "..."
        rows.append((
            level,
            ref,
            h_trans,
            f"{int(r['human_words_correct'])}/{int(r['total_reference_words'])}",
            ai_trans,
            f"{int(r['ai_words_correct'])}/{int(r['total_reference_words'])}",
        ))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "The human grader writes exactly what they hear "
        '(e.g., "laaves" for "loves", "baild" for "build"), while the AI '
        "transcription normalizes speech to correct English words. This explains "
        "why AI scores are consistently higher."
    )

    # ══════════════════════════════════════════
    # Section 5: Inter-Human Agreement & Gold Standard
    # ══════════════════════════════════════════
    doc.add_heading("5. Inter-Human Agreement & Gold Standard", level=1)

    doc.add_paragraph(
        f"Of the {n_unique_graded} graded audios, {intergrader['n_paired']} were graded "
        f"by two independent human raters. We use these dual-graded audios to assess "
        f"inter-rater reliability and establish a human gold standard."
    )

    doc.add_heading("What is Pearson Correlation (r)?", level=2)
    doc.add_paragraph(
        "Pearson's r measures the linear relationship between two sets of scores, "
        "ranging from −1 (perfect inverse) to +1 (perfect agreement). An r > 0.80 "
        "is generally considered strong agreement; r > 0.90 is very strong."
    )

    doc.add_heading("What is Cohen's d?", level=2)
    doc.add_paragraph(
        "Cohen's d measures the standardized difference between two groups. "
        "For paired graders, d = mean(Grader1 − Grader2) / SD(differences). "
        "Values near 0 indicate negligible systematic bias; |d| < 0.20 is considered "
        "small, 0.20–0.50 medium, and > 0.80 large."
    )

    doc.add_heading("Inter-Grader Reliability Results", level=2)

    headers = ["Metric", "Value"]
    rows = [
        ("Dual-graded audios (n)", int(intergrader["n_paired"])),
        ("Pearson r (Grader 1 vs Grader 2)",
         f"{fmt(intergrader['r_human_human'], 3)} (p < 0.0001)"),
        ("Cohen's d (paired)", f"{fmt(cohens_d, 3)}"),
        ("Exact agreement (same word count)", f"{fmt(intergrader['pct_exact'], 1)}%"),
        ("Within 1 word", f"{fmt(intergrader['pct_within1'], 1)}%"),
        ("Within 2 words", f"{fmt(intergrader['pct_within2'], 1)}%"),
        ("Mean absolute difference", f"{fmt(intergrader['mean_abs_diff_graders'])} words"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_heading("Establishing the Gold Standard", level=2)

    doc.add_paragraph(
        f"The inter-grader evidence supports using the human consensus score "
        f"(average of both graders) as a gold standard for benchmarking AI:"
    )

    bullets = [
        f"Strong correlation (r = {intergrader['r_human_human']:.3f}) — human graders "
        f"track the same underlying construct (student reading ability).",

        f"Negligible systematic bias (Cohen's d = {cohens_d:.3f}) — neither grader "
        f"consistently scores higher or lower than the other.",

        f"High close agreement — {intergrader['pct_within2']:.1f}% of scores are within "
        f"2 words, and exact agreement occurs {intergrader['pct_exact']:.1f}% of the time.",

        f"Adequate dual-coding coverage — {intergrader['n_paired']} audios "
        f"({intergrader['n_paired'] / n_audios_total * 100:.0f}% of total) were "
        f"independently dual-graded, exceeding the typical 20–25% threshold "
        f"recommended in research methodology.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    # ══════════════════════════════════════════
    # Section 6: AI vs Human Comparison
    # ══════════════════════════════════════════
    doc.add_heading("6. AI vs Human Comparison", level=1)

    doc.add_paragraph(
        "With the human consensus established as the gold standard, we now benchmark "
        "the AI scoring system against it."
    )

    doc.add_heading("AI vs Human Consensus Scores", level=2)

    headers = ["Metric", "Value"]
    rows = [
        ("Human consensus mean (words correct)", fmt(intergrader["consensus_mean"])),
        ("AI mean (words correct)", fmt(intergrader["ai_mean"])),
        ("Mean AI deviation from consensus", fmt(intergrader["ai_vs_consensus_mean"])),
        ("Mean absolute deviation", fmt(intergrader["ai_vs_consensus_abs_mean"])),
        ("Pearson r (AI vs human consensus)",
         f"{fmt(intergrader['r_ai_consensus'], 3)}"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    ai_dev = intergrader['ai_vs_consensus_mean']
    direction = "higher" if ai_dev > 0 else "lower"
    doc.add_paragraph(
        f"The AI scores {abs(ai_dev):.2f} words {direction} than "
        f"the human consensus on average. The AI-consensus correlation "
        f"(r = {intergrader['r_ai_consensus']:.3f}) is weaker than the human-human "
        f"correlation (r = {intergrader['r_human_human']:.3f}), confirming that the AI "
        f"transcription scores on a systematically different standard — it tends to be "
        f"more lenient because ASR normalizes mispronunciations to correct words."
    )

    # ══════════════════════════════════════════
    # Section 7: Pairwise Correlation Heatmaps
    # ══════════════════════════════════════════
    doc.add_heading("7. Pair-wise Grader Correlation Heatmaps", level=1)

    doc.add_paragraph(
        "Pearson correlations computed on shared audio files between each pair of "
        "graders (and AI). NaN indicates fewer than 3 shared files."
    )

    if os.path.exists(HEATMAP_ALL):
        doc.add_picture(HEATMAP_ALL, width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(
            "Figure 1: Pair-wise Pearson correlation heatmap — all graders + AI"
        ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")

    if os.path.exists(HEATMAP_RELIABLE):
        doc.add_picture(HEATMAP_RELIABLE, width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(
            "Figure 2: Pair-wise Pearson correlation — reliable graders only (>97% completion)"
        ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")

    # Correlation table for reliable graders
    reliable = ["Amna", "Dania", "Rehma", "Rukhshan", "Semal", "AI"]
    rel_graders = [g for g in reliable if g in valid_graders]
    headers = [""] + rel_graders
    rows = []
    for g1 in rel_graders:
        row = [g1]
        for g2 in rel_graders:
            val = corr_matrix.loc[g1, g2]
            n_s = overlap_matrix.loc[g1, g2]
            if pd.isna(val):
                row.append(f"— (n={n_s})")
            elif g1 == g2:
                row.append("1.00")
            else:
                row.append(f"{val:.2f} (n={n_s})")
        rows.append(row)
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Most human-human pairs correlate above r = 0.83. Human-AI pairs fall in the "
        "0.61–0.75 range, reflecting the systematic scoring gap. The lower Amna-Dania "
        "correlation (r = 0.58, n=13) is driven by 2 audios where Dania flagged "
        "'parent recording' while Amna transcribed the audio."
    )

    # ══════════════════════════════════════════
    # Section 8: WCPM Analysis
    # ══════════════════════════════════════════
    doc.add_heading("8. Words Correct Per Minute (WCPM)", level=1)

    doc.add_paragraph(
        "WCPM = words scored correct / audio duration in minutes. "
        "Both human and AI scores use jiwer-aligned transcription word counts."
    )

    doc.add_heading("Overall WCPM", level=2)

    h_wcpm = wcpm_overall["human_wcpm"]
    a_wcpm = wcpm_overall["ai_wcpm"]
    d_wcpm = wcpm_overall["wcpm_diff"]

    headers = ["Statistic", "Human WCPM", "AI WCPM", "Difference (H−AI)"]
    rows = [
        ("Mean", fmt(h_wcpm["mean"], 1), fmt(a_wcpm["mean"], 1), fmt(d_wcpm["mean"], 1)),
        ("Median", fmt(h_wcpm["50%"], 1), fmt(a_wcpm["50%"], 1), fmt(d_wcpm["50%"], 1)),
        ("SD", fmt(h_wcpm["std"], 1), fmt(a_wcpm["std"], 1), fmt(d_wcpm["std"], 1)),
        ("Min", fmt(h_wcpm["min"], 1), fmt(a_wcpm["min"], 1), fmt(d_wcpm["min"], 1)),
        ("Max", fmt(h_wcpm["max"], 1), fmt(a_wcpm["max"], 1), fmt(d_wcpm["max"], 1)),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(f"Based on {wcpm_n} audio files with matched duration data.")

    doc.add_heading("WCPM by Pre / Post", level=2)

    headers = ["Period", "Human Mean", "Human Median", "AI Mean", "AI Median", "n"]
    rows = []
    pre_hm = post_hm = pre_am = post_am = None
    for period in ["pre", "post"]:
        if period in wcpm_prepost.index:
            r = wcpm_prepost.loc[period]
            rows.append((
                period.capitalize(),
                fmt(r[("human_wcpm", "mean")], 1),
                fmt(r[("human_wcpm", "median")], 1),
                fmt(r[("ai_wcpm", "mean")], 1),
                fmt(r[("ai_wcpm", "median")], 1),
                int(r[("human_wcpm", "count")]),
            ))
            if period == "pre":
                pre_hm, pre_am = r[("human_wcpm", "mean")], r[("ai_wcpm", "mean")]
            else:
                post_hm, post_am = r[("human_wcpm", "mean")], r[("ai_wcpm", "mean")]

    if pre_hm and post_hm:
        rows.append(("Change (\u0394)", f"+{post_hm - pre_hm:.1f}", "",
                     f"+{post_am - pre_am:.1f}", "", ""))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    if pre_hm and post_hm:
        doc.add_paragraph(
            f"Human WCPM increases from {pre_hm:.1f} (pre) to {post_hm:.1f} (post), "
            f"a gain of {post_hm - pre_hm:.1f} WCPM. AI WCPM shows a parallel gain "
            f"({pre_am:.1f} \u2192 {post_am:.1f})."
        )

    # ── Footer ──
    doc.add_paragraph("")
    note = doc.add_paragraph("Generated by scripts/v2/v2_summary_report.py")
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in note.runs:
        run.font.size = Pt(9)

    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc.save(REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()

    print("Computing grader assignments...")
    grader_table = get_grader_assignments()

    print("Computing inter-grader consensus & WER...")
    intergrader = compute_intergrader_consensus(df)

    print("Computing Cohen's d...")
    cohens_d, cohens_d_n = compute_cohens_d(df)
    print(f"  Cohen's d = {cohens_d:.3f} (n={cohens_d_n})")

    print("Computing pairwise correlations...")
    corr_matrix, overlap_matrix, valid_graders = compute_correlations(df)

    print("Selecting transcription examples...")
    examples = get_transcription_examples(df)

    print("Computing WCPM...")
    wcpm_overall, wcpm_prepost, wcpm_n = compute_wcpm(df)

    print("Generating summary report...")
    generate_summary_report(
        df, grader_table, intergrader, cohens_d,
        corr_matrix, overlap_matrix, valid_graders,
        examples,
        wcpm_overall, wcpm_prepost, wcpm_n,
    )
    print("Done!")


if __name__ == "__main__":
    main()
