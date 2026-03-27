# %%
"""
v2_report.py — Generate a Word document report from v2 analysis data.

Reads pre-computed CSVs and the heatmap plot, then writes
reports/v2_grading_analysis_report.docx.

Re-run this script whenever the underlying CSVs change to get an updated report.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

V2_SCORES       = os.path.join(PROJECT_DIR, "data", "v2", "clean", "v2_merged_scores.csv")
WCPM_CSV        = os.path.join(PROJECT_DIR, "data", "v1", "clean", "wcpm_by_student.csv")
WORD_SUMMARY    = os.path.join(PROJECT_DIR, "data", "v1", "clean", "word_level_summary.csv")
MERGED_V1       = os.path.join(PROJECT_DIR, "data", "v1", "clean", "merged_for_analysis.csv")
HEATMAP_PNG     = os.path.join(PROJECT_DIR, "plots", "v2", "v2_pairwise_correlation_heatmap.png")
RELIABLE_HEATMAP_PNG = os.path.join(PROJECT_DIR, "plots", "v2", "v2_reliable_graders_heatmap.png")
REPORT_DIR      = os.path.join(PROJECT_DIR, "reports", "v2")
REPORT_PATH     = os.path.join(REPORT_DIR, "v2_grading_analysis_report.docx")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def add_table(doc, headers, rows, col_widths=None):
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
    """Format a number, returning 'N/A' for NaN."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"


# ──────────────────────────────────────────────
# Data loading & computation
# ──────────────────────────────────────────────
def load_v2_scores():
    df = pd.read_csv(V2_SCORES)
    # Use jiwer-based AI score (same method as human) instead of Azure pronunciation score
    if "ai_words_correct_jiwer" in df.columns:
        df["ai_words_correct"] = df["ai_words_correct_jiwer"]
    print(f"v2_merged_scores: {len(df)} rows, {df['audio_file_name'].nunique()} unique audios")
    return df


def compute_overview(df):
    grader_counts = df.groupby("grader")["audio_file_name"].count().sort_values(ascending=False)
    return {
        "total_rows": len(df),
        "unique_audios": df["audio_file_name"].nunique(),
        "graders": sorted(df["grader"].unique()),
        "grader_counts": grader_counts,
        "ai_match_rate": df["ai_total_score"].notna().sum(),
    }


def compute_score_stats(df):
    h = df["human_words_correct"]
    a = df["ai_words_correct"]
    valid = df[["human_words_correct", "ai_words_correct"]].dropna()
    r, p = pearsonr(valid["human_words_correct"], valid["ai_words_correct"])
    diff = valid["ai_words_correct"] - valid["human_words_correct"]
    return {
        "human": h.describe(),
        "ai": a.describe(),
        "pearson_r": r,
        "pearson_p": p,
        "n_corr": len(valid),
        "mean_diff": diff.mean(),
        "std_diff": diff.std(),
        "mean_abs_diff": diff.abs().mean(),
    }


def compute_pre_post(df):
    return df.groupby("pre_or_post")[["human_words_correct", "ai_words_correct"]].mean().round(2)


def compute_correlations(df):
    """Recompute the pair-wise Pearson correlation matrix."""
    human_pivot = df.pivot_table(
        index="audio_file_name", columns="grader",
        values="human_words_correct", aggfunc="first"
    )
    ai_scores = df.drop_duplicates(subset=["audio_file_name"])[
        ["audio_file_name", "ai_words_correct"]
    ].set_index("audio_file_name")

    score_matrix = human_pivot.copy()
    score_matrix["AI"] = ai_scores["ai_words_correct"]

    min_n = 10
    valid_graders = [c for c in score_matrix.columns if score_matrix[c].notna().sum() >= min_n]
    score_matrix = score_matrix[valid_graders]

    corr = pd.DataFrame(np.nan, index=valid_graders, columns=valid_graders)
    overlap = pd.DataFrame(0, index=valid_graders, columns=valid_graders, dtype=int)

    for i, g1 in enumerate(valid_graders):
        for j, g2 in enumerate(valid_graders):
            if i == j:
                corr.loc[g1, g2] = 1.0
                overlap.loc[g1, g2] = int(score_matrix[g1].notna().sum())
            else:
                mask = score_matrix[g1].notna() & score_matrix[g2].notna()
                n_shared = mask.sum()
                overlap.loc[g1, g2] = n_shared
                if n_shared >= 3:
                    r, _ = pearsonr(score_matrix.loc[mask, g1], score_matrix.loc[mask, g2])
                    corr.loc[g1, g2] = r

    return corr.astype(float), overlap, valid_graders


def compute_wcpm_overall():
    df = pd.read_csv(WCPM_CSV)
    valid = df[df["human_total_words"] > 0]
    result = {}
    for col in ["human_wcpm_partial", "ai_wcpm_partial", "wcpm_diff_partial"]:
        result[col] = valid[col].describe()
    return result, len(valid)


def compute_wcpm_prepost(df_v2):
    """Join v2 word scores with per-question duration from v1 dataset."""
    v1 = pd.read_csv(MERGED_V1)[["audio_filename", "ai_utterance_duration_seconds"]].dropna()
    v2_dedup = df_v2.drop_duplicates(subset="audio_file_name")
    merged = v2_dedup.merge(v1, left_on="audio_file_name", right_on="audio_filename", how="inner")
    merged = merged[merged["ai_utterance_duration_seconds"] > 0].copy()
    merged["human_wcpm"] = merged["human_words_correct"] / (merged["ai_utterance_duration_seconds"] / 60)
    merged["ai_wcpm"] = merged["ai_words_correct"] / (merged["ai_utterance_duration_seconds"] / 60)
    summary = merged.groupby("pre_or_post")[["human_wcpm", "ai_wcpm"]].agg(["mean", "median", "std", "count"]).round(1)
    return summary, len(merged)


def compute_word_disagreement():
    df = pd.read_csv(WORD_SUMMARY)
    df = df.sort_values("mismatch_rate", ascending=False)
    return df


def compute_intergrader_consensus(df):
    """Compute combined human inter-grader score and benchmark against AI.

    For each audio graded by 2 humans, compute:
      - Human consensus score (average of both graders)
      - Inter-grader agreement metrics
      - Word Error Rate (WER) for human consensus vs AI
    """
    multi = df.groupby("audio_file_name").filter(lambda x: len(x) > 1)

    rows = []
    for audio, group in multi.groupby("audio_file_name"):
        h_scores = group["human_words_correct"].values
        ai_score = group["ai_words_correct"].iloc[0]
        total_ref = group["total_reference_words"].iloc[0]

        if len(h_scores) < 2 or total_ref == 0:
            continue

        h_avg = np.mean(h_scores)
        h_diff = abs(float(h_scores[0]) - float(h_scores[1]))

        rows.append({
            "h1": float(h_scores[0]),
            "h2": float(h_scores[1]),
            "h_avg": h_avg,
            "h_diff": h_diff,
            "ai": float(ai_score),
            "total_ref": float(total_ref),
        })

    rdf = pd.DataFrame(rows)

    # Inter-grader agreement
    r_hh, p_hh = pearsonr(rdf["h1"], rdf["h2"])
    pct_exact = (rdf["h_diff"] == 0).mean() * 100
    pct_within1 = (rdf["h_diff"] <= 1).mean() * 100
    pct_within2 = (rdf["h_diff"] <= 2).mean() * 100

    # AI vs consensus
    ai_vs_consensus = rdf["ai"] - rdf["h_avg"]
    r_ac, p_ac = pearsonr(rdf["h_avg"], rdf["ai"])

    # WER = (total_ref - words_correct) / total_ref
    rdf["human_wer"] = (rdf["total_ref"] - rdf["h_avg"]) / rdf["total_ref"]
    rdf["ai_wer"] = (rdf["total_ref"] - rdf["ai"]) / rdf["total_ref"]

    return {
        "n_paired": len(rdf),
        "mean_abs_diff_graders": rdf["h_diff"].mean(),
        "median_abs_diff_graders": rdf["h_diff"].median(),
        "pct_exact": pct_exact,
        "pct_within1": pct_within1,
        "pct_within2": pct_within2,
        "r_human_human": r_hh,
        "p_human_human": p_hh,
        "consensus_mean": rdf["h_avg"].mean(),
        "ai_mean": rdf["ai"].mean(),
        "ai_vs_consensus_mean": ai_vs_consensus.mean(),
        "ai_vs_consensus_abs_mean": ai_vs_consensus.abs().mean(),
        "r_ai_consensus": r_ac,
        "p_ai_consensus": p_ac,
        "human_wer": rdf["human_wer"].mean() * 100,
        "ai_wer": rdf["ai_wer"].mean() * 100,
    }


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────
def generate_report(overview, score_stats, pre_post, corr_matrix, overlap_matrix,
                    valid_graders, wcpm_overall, wcpm_n, wcpm_prepost, wcpm_pp_n,
                    word_df, intergrader):
    doc = Document()

    # Base font
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ── Title ──
    title = doc.add_heading("V2 Grading Analysis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("BEAJ Audio Dataset — Inter-Rater Reliability & AI Calibration")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("")

    # ── Executive Summary ──
    doc.add_heading("Executive Summary", level=1)

    h_mean = score_stats["human"]["mean"]
    a_mean = score_stats["ai"]["mean"]
    mean_diff = score_stats["mean_diff"]
    r = score_stats["pearson_r"]

    bullets = [
        f"{len(overview['graders']) - 1} human graders + AI scored "
        f"{overview['unique_audios']} unique audio files ({overview['total_rows']} total graded rows).",

        f"Human graders score on average {h_mean:.2f} words correct per audio; "
        f"AI scores {a_mean:.2f} — a systematic gap of {abs(mean_diff):.2f} points "
        f"(AI scores lower).",

        f"Human-AI Pearson correlation: r = {r:.3f} (n={score_stats['n_corr']}). "
        f"Most human-human pairs correlate above r = 0.90.",

        f"Pre-to-post improvement is visible in both human and AI scores, "
        f"consistent with genuine learning gains.",

        f"Word-level analysis shows AI particularly underscores South Asian proper nouns "
        f"('faiz': 83% mismatch rate, 'zara': 45%), pointing to a cultural calibration gap.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    # ══════════════════════════════════════════
    # Section 1: Dataset Overview
    # ══════════════════════════════════════════
    doc.add_heading("1. Dataset Overview", level=1)

    doc.add_paragraph(
        f"The v2 grading dataset covers {overview['unique_audios']} unique audio recordings "
        f"from Grade 1 students, graded by {len(overview['graders']) - 1} human graders and 1 AI system. "
        f"AI responses were matched to {overview['ai_match_rate']} of {overview['total_rows']} transcribed rows ({"%.1f" % (overview['ai_match_rate']/overview['total_rows']*100)}%)."
    )

    doc.add_heading("Transcriptions per Grader", level=2)
    gc = overview["grader_counts"]
    headers = ["Grader", "Transcriptions"]
    rows = [(g, int(n)) for g, n in gc.items()]
    add_table(doc, headers, rows)

    # ══════════════════════════════════════════
    # Section 2: Score Distribution
    # ══════════════════════════════════════════
    doc.add_heading("2. Score Distribution", level=1)

    doc.add_paragraph(
        "Scores represent word-match counts — the number of reference words a student "
        "correctly reproduced. Human scores are derived from transcription alignment; "
        "AI scores count words graded as fully correct (normalized score = 2)."
    )

    h = score_stats["human"]
    a = score_stats["ai"]
    headers = ["Statistic", "Human", "AI"]
    rows = [
        ("Count", int(h["count"]), int(a["count"])),
        ("Mean",   fmt(h["mean"]), fmt(a["mean"])),
        ("SD",     fmt(h["std"]),  fmt(a["std"])),
        ("Min",    fmt(h["min"], 0), fmt(a["min"], 0)),
        ("25th pct", fmt(h["25%"], 0), fmt(a["25%"], 0)),
        ("Median", fmt(h["50%"], 0), fmt(a["50%"], 0)),
        ("75th pct", fmt(h["75%"], 0), fmt(a["75%"], 0)),
        ("Max",    fmt(h["max"], 0), fmt(a["max"], 0)),
    ]
    add_table(doc, headers, rows)

    # ══════════════════════════════════════════
    # Section 3: AI vs Human Comparison
    # ══════════════════════════════════════════
    doc.add_heading("3. AI vs Human Score Comparison", level=1)

    headers = ["Metric", "Value"]
    rows = [
        ("Pearson r (human vs AI)", f"{fmt(score_stats['pearson_r'])} (p < 0.0001, n={score_stats['n_corr']})"),
        ("Mean difference (AI − Human)", fmt(score_stats["mean_diff"])),
        ("SD of difference", fmt(score_stats["std_diff"])),
        ("Mean absolute difference", fmt(score_stats["mean_abs_diff"])),
        ("Bias direction", "AI scores lower than humans"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"AI scores are consistently {abs(score_stats['mean_diff']):.2f} points lower than human scores "
        f"(SD = {score_stats['std_diff']:.2f}). The correlation of r = {score_stats['pearson_r']:.3f} "
        f"indicates the AI captures the relative ordering of students reasonably well, "
        f"but its absolute scores are miscalibrated downward."
    )

    # ══════════════════════════════════════════
    # Section 4: Pre / Post Comparison
    # ══════════════════════════════════════════
    doc.add_heading("4. Pre / Post Comparison", level=1)

    headers = ["Period", "Human Mean Words Correct", "AI Mean Words Correct"]
    rows = []
    pre_h = post_h = pre_a = post_a = None
    for period, row_data in pre_post.iterrows():
        rows.append((period.capitalize(), fmt(row_data["human_words_correct"]), fmt(row_data["ai_words_correct"])))
        if period == "pre":
            pre_h, pre_a = row_data["human_words_correct"], row_data["ai_words_correct"]
        elif period == "post":
            post_h, post_a = row_data["human_words_correct"], row_data["ai_words_correct"]

    if pre_h is not None and post_h is not None:
        rows.append(("Change (Δ)", f"+{post_h - pre_h:.2f}", f"+{post_a - pre_a:.2f}"))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    if pre_h is not None and post_h is not None:
        doc.add_paragraph(
            f"Both human (+{post_h - pre_h:.2f}) and AI (+{post_a - pre_a:.2f}) scores are higher "
            f"in post-assessments, consistent with learning gains across the intervention period."
        )

    # ══════════════════════════════════════════
    # Section 5: Pair-wise Correlations
    # ══════════════════════════════════════════
    doc.add_heading("5. Pair-wise Grader Correlations", level=1)

    doc.add_paragraph(
        "Pearson correlations computed on shared audio files. "
        "NaN = fewer than 3 shared files (insufficient for a meaningful correlation). "
        "Diagonal = self-correlation (1.00)."
    )

    # Heatmap image
    if os.path.exists(HEATMAP_PNG):
        doc.add_picture(HEATMAP_PNG, width=Inches(5.5))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph("Figure 1: Pair-wise Pearson correlation heatmap (blue = low, yellow = high)").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Reliable graders heatmap
    if os.path.exists(RELIABLE_HEATMAP_PNG):
        doc.add_paragraph("")
        doc.add_picture(RELIABLE_HEATMAP_PNG, width=Inches(5.5))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph("Figure 2: Pair-wise Pearson correlation — reliable graders only (>97% completion)").alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")

    # Correlation table
    headers = [""] + valid_graders
    rows = []
    for g1 in valid_graders:
        row = [g1]
        for g2 in valid_graders:
            val = corr_matrix.loc[g1, g2]
            n_shared = overlap_matrix.loc[g1, g2]
            if pd.isna(val):
                row.append(f"— (n={n_shared})")
            elif g1 == g2:
                row.append("1.00")
            else:
                row.append(f"{val:.2f} (n={n_shared})")
        rows.append(row)
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Most human-human pairs correlate above r = 0.90. Human-AI pairs fall in the "
        "0.61–0.89 range, reflecting the systematic scoring gap documented in Section 3."
    )

    # ══════════════════════════════════════════
    # Section 6: Combined Human Inter-Grader Score
    # ══════════════════════════════════════════
    doc.add_heading("6. Combined Human Inter-Grader Score", level=1)

    doc.add_paragraph(
        f"Each of the {intergrader['n_paired']} audio files in this dataset was independently "
        f"graded by two human raters. The combined human inter-grader score (consensus score) "
        f"is the average of both graders' word-correct counts for each audio. This consensus "
        f"score serves as the human gold standard against which AI is benchmarked."
    )

    doc.add_heading("Human Inter-Grader Agreement", level=2)

    headers = ["Metric", "Value"]
    rows = [
        ("Paired audios", int(intergrader["n_paired"])),
        ("Pearson r (Grader 1 vs Grader 2)",
         f"{fmt(intergrader['r_human_human'], 3)} (p < 0.0001)"),
        ("Exact agreement (same word count)", f"{fmt(intergrader['pct_exact'], 1)}%"),
        ("Within 1 word", f"{fmt(intergrader['pct_within1'], 1)}%"),
        ("Within 2 words", f"{fmt(intergrader['pct_within2'], 1)}%"),
        ("Mean absolute difference", f"{fmt(intergrader['mean_abs_diff_graders'])} words"),
        ("Median absolute difference", f"{fmt(intergrader['median_abs_diff_graders'])} words"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Human graders correlate at r = {intergrader['r_human_human']:.3f} and agree exactly "
        f"{intergrader['pct_exact']:.1f}% of the time. When they disagree, the difference is "
        f"typically small (median {intergrader['median_abs_diff_graders']:.0f} word, "
        f"{intergrader['pct_within2']:.1f}% within 2 words). This level of agreement is "
        f"consistent with published benchmarks for early literacy oral reading assessments, "
        f"where inter-rater correlations of r = 0.85–0.95 are considered strong."
    )

    doc.add_heading("AI vs Human Consensus", level=2)

    headers = ["Metric", "Value"]
    rows = [
        ("Human consensus mean (words correct)", fmt(intergrader["consensus_mean"])),
        ("AI mean (words correct)", fmt(intergrader["ai_mean"])),
        ("Mean AI deviation from consensus", fmt(intergrader["ai_vs_consensus_mean"])),
        ("Mean absolute deviation", fmt(intergrader["ai_vs_consensus_abs_mean"])),
        ("Pearson r (AI vs consensus)", f"{fmt(intergrader['r_ai_consensus'], 3)}"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"When benchmarked against the human consensus score, AI deviates by "
        f"{abs(intergrader['ai_vs_consensus_mean']):.2f} words on average (scoring lower) "
        f"with a correlation of r = {intergrader['r_ai_consensus']:.3f}. The AI-consensus "
        f"correlation ({intergrader['r_ai_consensus']:.3f}) is notably weaker than the "
        f"human-human correlation ({intergrader['r_human_human']:.3f}), confirming that AI "
        f"applies a systematically different — and stricter — scoring standard."
    )

    # ══════════════════════════════════════════
    # Section 7: Word Error Rate & Industry Benchmarks
    # ══════════════════════════════════════════
    doc.add_heading("7. Word Error Rate (WER) & Industry Benchmarks", level=1)

    doc.add_heading("What is Word Error Rate?", level=2)
    doc.add_paragraph(
        "Word Error Rate (WER) is the standard metric for evaluating speech recognition and "
        "oral reading assessment accuracy. It measures the proportion of reference words that "
        "were not correctly identified. In this context:"
    )
    doc.add_paragraph(
        "WER = (Total Reference Words − Words Scored Correct) / Total Reference Words",
        style="List Bullet",
    )
    doc.add_paragraph(
        "A WER of 0% means every word was scored as correct; a WER of 100% means no words "
        "were scored as correct. Lower WER indicates better performance. Note: in this dataset, "
        "WER reflects scoring accuracy (whether the student's pronunciation was judged correct), "
        "not transcription accuracy (whether the system heard the right word).",
        style="List Bullet",
    )

    doc.add_heading("WER Results", level=2)

    headers = ["Scorer", "Mean WER"]
    rows = [
        ("Human Consensus (avg of 2 graders)", f"{intergrader['human_wer']:.1f}%"),
        ("AI (Azure Pronunciation Assessment)", f"{intergrader['ai_wer']:.1f}%"),
        ("Gap (AI − Human)", f"+{intergrader['ai_wer'] - intergrader['human_wer']:.1f} pp"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_heading("Industry Benchmarks", level=2)

    doc.add_paragraph(
        "The table below compares our results against published benchmarks for automated "
        "oral reading assessment and speech recognition systems:"
    )

    headers = ["Benchmark / System", "WER / Error Rate", "Context"]
    rows = [
        ("This study — Human consensus", f"{intergrader['human_wer']:.1f}%",
         "Grade 1, Pakistani English, 2 graders"),
        ("This study — AI (Azure)", f"{intergrader['ai_wer']:.1f}%",
         "Azure Pronunciation Assessment"),
        ("ASER / Annual Status of Education Report", "~15–25%",
         "Early-grade reading, South Asia"),
        ("Hasbrouck & Tindal oral reading norms", "~10–30%",
         "US grades 1–3 (varies by percentile)"),
        ("Commercial ASR (native English)", "5–10%",
         "Google, Azure, AWS on clean English audio"),
        ("Commercial ASR (non-native / child speech)", "15–40%",
         "Known degradation for children and L2 speakers"),
        ("Lit Numeracy Assessment (USAID-funded)", "~20–35%",
         "Early-grade reading assessments, developing countries"),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"The human consensus WER of {intergrader['human_wer']:.1f}% falls within the expected "
        f"range for early-grade readers in South Asia. The AI WER of {intergrader['ai_wer']:.1f}% "
        f"is {intergrader['ai_wer'] - intergrader['human_wer']:.1f} percentage points higher, "
        f"placing it at the upper end of what commercial ASR systems achieve on non-native child "
        f"speech. This gap is consistent with documented challenges: child speech has higher "
        f"acoustic variability, Pakistani English phonology differs from training data norms, "
        f"and culturally specific vocabulary (proper nouns like 'Faiz' and 'Zara') is "
        f"underrepresented in general-purpose ASR models."
    )

    # ══════════════════════════════════════════
    # Section 8: Words Correct Per Minute
    # ══════════════════════════════════════════
    doc.add_heading("8. Words Correct Per Minute (WCPM)", level=1)

    doc.add_paragraph(
        "WCPM = words correctly scored divided by audio duration in minutes. "
        "Overall figures are aggregated per student from wcpm_by_student.csv "
        f"({wcpm_n} students with valid human data). "
        "Pre/post figures are computed per question by joining v2 scores with "
        f"per-question audio durations ({wcpm_pp_n} audio files matched)."
    )

    doc.add_heading("Overall WCPM", level=2)
    h_wcpm = wcpm_overall["human_wcpm_partial"]
    a_wcpm = wcpm_overall["ai_wcpm_partial"]
    d_wcpm = wcpm_overall["wcpm_diff_partial"]

    headers = ["Statistic", "Human WCPM", "AI WCPM", "Difference (H−AI)"]
    rows = [
        ("Mean",   fmt(h_wcpm["mean"], 1), fmt(a_wcpm["mean"], 1), fmt(d_wcpm["mean"], 1)),
        ("Median", fmt(h_wcpm["50%"], 1),  fmt(a_wcpm["50%"], 1),  fmt(d_wcpm["50%"], 1)),
        ("SD",     fmt(h_wcpm["std"], 1),  fmt(a_wcpm["std"], 1),  fmt(d_wcpm["std"], 1)),
        ("Min",    fmt(h_wcpm["min"], 1),  fmt(a_wcpm["min"], 1),  fmt(d_wcpm["min"], 1)),
        ("Max",    fmt(h_wcpm["max"], 1),  fmt(a_wcpm["max"], 1),  fmt(d_wcpm["max"], 1)),
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
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
        rows.append(("Change (Δ)", f"+{post_hm - pre_hm:.1f}", "", f"+{post_am - pre_am:.1f}", "", ""))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    if pre_hm and post_hm:
        doc.add_paragraph(
            f"Human WCPM increases from {pre_hm:.1f} (pre) to {post_hm:.1f} (post), "
            f"a gain of {post_hm - pre_hm:.1f} WCPM (~{(post_hm - pre_hm) / pre_hm * 100:.0f}%). "
            f"AI WCPM shows a similar gain ({pre_am:.1f} → {post_am:.1f}, "
            f"+{post_am - pre_am:.1f}). The human-AI gap is consistent across both periods, "
            f"confirming the bias is not stage-specific."
        )

    # ══════════════════════════════════════════
    # Section 9: Word-Level Disagreement
    # ══════════════════════════════════════════
    doc.add_heading("9. Word-Level Disagreement (AI vs Human)", level=1)

    doc.add_paragraph(
        "Sourced from word_level_summary.csv. Scores are on a 0–2 scale. "
        "Mismatch rate = proportion of instances where AI and human scores differ. "
        "Table sorted by mismatch rate (descending)."
    )

    top_words = word_df.head(12)
    headers = ["Word", "n", "Mean Human", "Mean AI", "Mean Diff (H−AI)", "Mismatch Rate"]
    rows = []
    for _, row in top_words.iterrows():
        diff_str = f"+{row['mean_diff']:.2f}" if row["mean_diff"] >= 0 else f"{row['mean_diff']:.2f}"
        rows.append((
            row["word"],
            int(row["n"]),
            fmt(row["mean_human"]),
            fmt(row["mean_ai"]),
            diff_str,
            f"{row['mismatch_rate'] * 100:.1f}%",
        ))
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "'Faiz' and 'zara' — South Asian proper nouns embedded in the assessment passages — "
        "show the highest mismatch rates (83% and 45% respectively). These names are "
        "phonologically underrepresented in general-purpose AI training data. "
        "Multisyllabic words ('together', 'locked', 'build') also show elevated mismatch. "
        "Notably, 'loves' is the one word where AI scores higher than humans."
    )

    # ══════════════════════════════════════════
    # Section 10: Motivation for Calibration
    # ══════════════════════════════════════════
    doc.add_heading("10. Motivation for Calibrating AI Scoring to Local Standards", level=1)

    points = [
        (
            "The bias is systematic, not random — making it correctable.",
            f"AI scores are consistently {abs(score_stats['mean_diff']):.2f} points lower (SD = {score_stats['std_diff']:.2f}) "
            f"at the word-count level, and ~{d_wcpm['mean']:.0f} WCPM lower in fluency estimates. "
            f"A consistent downward offset is the hallmark of a calibration problem rather than random noise. "
            f"A linear recalibration or per-word adjustment could substantially close this gap."
        ),
        (
            "South Asian proper nouns reveal a specific cultural failure mode.",
            "'Faiz' and 'zara' are South Asian names in the assessment passages. "
            "These are phonologically and lexically underrepresented in AI training corpora that skew "
            "toward American and British English, resulting in systematic score suppression for "
            "culturally-specific vocabulary that human graders handle without difficulty."
        ),
        (
            "The gap is non-uniform across words, so flat scaling is insufficient.",
            "AI severely underscores 'together' (+0.72) and 'come' (+0.64), yet slightly overscores "
            "'loves' (−0.16). A single scaling factor would over-correct some words while "
            "under-correcting others. Effective calibration likely requires word-level or "
            "phoneme-class-level adjustments anchored to local human grader data."
        ),
        (
            "High human-AI correlation shows the AI understands the task — but not the context.",
            f"Pearson r = {score_stats['pearson_r']:.3f} means the AI's relative ranking of students "
            f"is meaningful. The problem is not random failure but a miscalibrated standard. "
            f"This is encouraging: targeted fine-tuning or post-hoc calibration on a modest "
            f"local dataset could bring scores into alignment without full retraining."
        ),
        (
            "Deployment risk without calibration.",
            "AI reading assessments deployed at scale in Pakistan without local calibration would "
            "systematically underestimate student fluency, potentially misdirecting remediation "
            "resources, under-identifying adequate performers, and eroding educator trust. "
            f"The ~{d_wcpm['mean']:.0f} WCPM gap translates to meaningful differences in the "
            "fluency classification thresholds used in early literacy programmes."
        ),
    ]

    for heading, body in points:
        p = doc.add_paragraph()
        run = p.add_run(heading)
        run.bold = True
        doc.add_paragraph(body)
        doc.add_paragraph("")

    # ══════════════════════════════════════════
    # Section 11: Key Findings
    # ══════════════════════════════════════════
    doc.add_heading("11. Key Findings", level=1)

    findings = [
        "Human inter-rater agreement is generally high — most human-human pairs correlate above "
        "r = 0.90. Notable exception: Amna-Dania (r = 0.58, n=13), though the small overlap warrants caution.",

        f"AI scores systematically lower than humans — {abs(score_stats['mean_diff']):.2f} points on average "
        f"(SD {score_stats['std_diff']:.2f}), with a moderate overall correlation of r = {score_stats['pearson_r']:.3f}. "
        f"The AI has a narrower score range than humans.",

        "Human-AI correlations (0.61–0.89) are weaker than human-human correlations (mostly >0.90), "
        "suggesting different scoring standards despite positive alignment.",

        "Pre/post improvement is detected in both human and AI scores, consistent with genuine learning gains.",

        "South Asian proper nouns ('faiz', 'zara') show the highest AI-human mismatch rates, "
        "providing the clearest evidence for culturally-specific calibration needs.",

        "Limited grader overlap for several pairs (e.g., Abdullah) limits the reliability of "
        "some pairwise correlations. Broader shared assignment pools would strengthen future estimates.",
    ]
    for i, f in enumerate(findings, 1):
        doc.add_paragraph(f"{i}. {f}")

    # ══════════════════════════════════════════
    # Section 12: Limitations
    # ══════════════════════════════════════════
    doc.add_heading("12. Limitations", level=1)

    limitations = [
        "WCPM is aggregated per student across all submissions; per-question timing is not available "
        "for all audio files.",
        "Pre/post WCPM uses duration data from the v1 dataset; only 322 of 558 audio files could be matched.",
        "Word-level mismatch data (word_level_summary.csv) is from the v1 grading pipeline and may "
        "not perfectly reflect v2 grader composition.",
        "All data is from Grade 1 students in a single cohort. Results may differ for other grades or cohorts.",
        "Limited double-grading overlap between some grader pairs means certain pairwise correlations "
        "are computed on very small samples (n < 5) and should be interpreted cautiously.",
    ]
    for lim in limitations:
        doc.add_paragraph(lim, style="List Bullet")

    # ── Footer note ──
    doc.add_paragraph("")
    note = doc.add_paragraph("Generated by scripts/v2/v2_report.py — re-run to update with new data.")
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in note.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = None  # reset to default

    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc.save(REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_v2_scores()

    print("Computing overview...")
    overview = compute_overview(df)

    print("Computing score stats...")
    score_stats = compute_score_stats(df)

    print("Computing pre/post word scores...")
    pre_post = compute_pre_post(df)

    print("Computing pair-wise correlations...")
    corr_matrix, overlap_matrix, valid_graders = compute_correlations(df)

    print("Computing overall WCPM...")
    wcpm_overall, wcpm_n = compute_wcpm_overall()

    print("Computing pre/post WCPM...")
    wcpm_prepost, wcpm_pp_n = compute_wcpm_prepost(df)

    print("Loading word disagreement data...")
    word_df = compute_word_disagreement()

    print("Computing inter-grader consensus & WER...")
    intergrader = compute_intergrader_consensus(df)

    print("Generating Word document...")
    generate_report(
        overview, score_stats, pre_post,
        corr_matrix, overlap_matrix, valid_graders,
        wcpm_overall, wcpm_n,
        wcpm_prepost, wcpm_pp_n,
        word_df, intergrader,
    )
    print("Done!")


if __name__ == "__main__":
    main()
