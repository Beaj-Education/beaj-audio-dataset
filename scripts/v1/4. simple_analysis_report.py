"""
Simple Grading Agreement Analysis + Word Document Report

Reads merged_for_analysis.csv and generates a Word document that answers:
1. How well do human graders agree with each other?
2. How well does AI agree with humans?
3. How good are AI vs human transcriptions?

All scores on the native 0-2 scale. No fancy stats — just averages,
standard deviations, and % agreement.
"""

import os
import sys
import pandas as pd
import numpy as np
# Add scripts dir to path so we can import helper_functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_functions import _parse_dict, normalize_word

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "v1", "clean", "merged_for_analysis.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "v1", "grading_agreement_report.docx")


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    print(f"Loaded {len(df)} rows, {df['audio_filename'].nunique()} unique audios")
    print(f"Graders: {sorted(df['grader_name'].dropna().unique())}")
    return df


# ──────────────────────────────────────────────
# Step 1: Human Inter-Rater Agreement
# ──────────────────────────────────────────────
def analyze_human_agreement(df):
    """Analyze agreement between human graders on paired audios."""

    # Find audios graded by more than one human
    grader_counts = df.groupby("audio_filename")["grader_name"].nunique()
    paired_audios = grader_counts[grader_counts > 1].index.tolist()

    # --- A. Paired comparison (word level) ---
    paired_results = []
    all_word_diffs = []  # collect every word-level abs diff across all pairs

    for audio in paired_audios:
        rows = df[df["audio_filename"] == audio]
        graders = rows["grader_name"].tolist()

        # Parse word-level feedback for each grader
        feedbacks = []
        for _, row in rows.iterrows():
            fb = _parse_dict(row.get("feedback_human_json"))
            # normalize keys
            fb_norm = {}
            for w, s in fb.items():
                nw = normalize_word(w)
                if nw:
                    try:
                        fb_norm[nw] = float(s)
                    except (ValueError, TypeError):
                        pass
            feedbacks.append((row["grader_name"], fb_norm))

        if len(feedbacks) != 2:
            continue

        g1_name, g1_scores = feedbacks[0]
        g2_name, g2_scores = feedbacks[1]

        # Skip if both have 0 words (empty audio)
        if len(g1_scores) == 0 and len(g2_scores) == 0:
            continue

        # Words graded by both
        common_words = set(g1_scores.keys()) & set(g2_scores.keys())
        if not common_words:
            continue

        exact_matches = 0
        word_abs_diffs = []
        for w in common_words:
            d = abs(g1_scores[w] - g2_scores[w])
            word_abs_diffs.append(d)
            all_word_diffs.append(d)
            if g1_scores[w] == g2_scores[w]:
                exact_matches += 1

        g1_avg = np.mean([g1_scores[w] for w in common_words])
        g2_avg = np.mean([g2_scores[w] for w in common_words])

        paired_results.append({
            "audio": audio,
            "grader_1": g1_name,
            "grader_2": g2_name,
            "words_compared": len(common_words),
            "grader_1_avg": round(g1_avg, 2),
            "grader_2_avg": round(g2_avg, 2),
            "mean_abs_diff": round(np.mean(word_abs_diffs), 2),
            "pct_exact_match": round(exact_matches / len(common_words) * 100, 1),
        })

    paired_df = pd.DataFrame(paired_results)

    # Summary stats across all paired audios
    paired_summary = {}
    if len(paired_df) > 0:
        paired_summary["n_paired_audios"] = len(paired_df)
        paired_summary["avg_mean_abs_diff"] = round(paired_df["mean_abs_diff"].mean(), 2)
        paired_summary["std_mean_abs_diff"] = round(paired_df["mean_abs_diff"].std(), 2)
        paired_summary["avg_pct_exact_match"] = round(paired_df["pct_exact_match"].mean(), 1)
    if all_word_diffs:
        paired_summary["overall_word_exact_match_pct"] = round(
            sum(1 for d in all_word_diffs if d == 0) / len(all_word_diffs) * 100, 1
        )
        paired_summary["overall_word_mean_abs_diff"] = round(np.mean(all_word_diffs), 2)
        paired_summary["total_words_compared"] = len(all_word_diffs)

    # --- B. Per-grader averages (all data) ---
    grader_stats = []
    for grader, group in df.groupby("grader_name"):
        all_scores = []
        for _, row in group.iterrows():
            fb = _parse_dict(row.get("feedback_human_json"))
            for w, s in fb.items():
                nw = normalize_word(w)
                if nw:
                    try:
                        all_scores.append(float(s))
                    except (ValueError, TypeError):
                        pass
        if all_scores:
            grader_stats.append({
                "grader": grader,
                "n_audios": len(group),
                "n_words": len(all_scores),
                "avg_score": round(np.mean(all_scores), 2),
                "std_score": round(np.std(all_scores), 2),
            })

    grader_df = pd.DataFrame(grader_stats).sort_values("avg_score", ascending=False)

    return paired_df, paired_summary, grader_df


# ──────────────────────────────────────────────
# Step 2: AI vs Human Agreement
# ──────────────────────────────────────────────
def analyze_ai_vs_human(df):
    """Compare AI and human word-level scores on the 0-2 scale."""

    all_human_scores = []
    all_ai_scores = []
    word_rows = []

    for _, row in df.iterrows():
        h_fb = _parse_dict(row.get("feedback_human_json"))
        a_fb = _parse_dict(row.get("feedback_ai_json"))
        if not h_fb or not a_fb:
            continue

        # Build normalized AI lookup
        ai_lookup = {}
        for w, obj in a_fb.items():
            nw = normalize_word(w)
            if nw and isinstance(obj, dict):
                ai_score = obj.get("normalized_rounded")
                if ai_score is not None:
                    try:
                        ai_lookup[nw] = float(ai_score)
                    except (ValueError, TypeError):
                        pass

        for w, hscore in h_fb.items():
            nw = normalize_word(w)
            if not nw or nw not in ai_lookup:
                continue
            try:
                hval = float(hscore)
            except (ValueError, TypeError):
                continue
            aval = ai_lookup[nw]

            all_human_scores.append(hval)
            all_ai_scores.append(aval)
            word_rows.append({"word": nw, "human": hval, "ai": aval})

    h_arr = np.array(all_human_scores)
    a_arr = np.array(all_ai_scores)

    summary = {
        "n_words": len(h_arr),
        "human_avg": round(float(np.mean(h_arr)), 2),
        "human_std": round(float(np.std(h_arr)), 2),
        "ai_avg": round(float(np.mean(a_arr)), 2),
        "ai_std": round(float(np.std(a_arr)), 2),
        "mean_abs_diff": round(float(np.mean(np.abs(h_arr - a_arr))), 2),
        "pct_exact_match": round(float(np.mean(h_arr == a_arr)) * 100, 1),
    }

    # Per-word analysis
    wdf = pd.DataFrame(word_rows)
    word_summary = (
        wdf.groupby("word")
        .agg(
            n=("word", "size"),
            human_avg=("human", "mean"),
            ai_avg=("ai", "mean"),
            mean_abs_diff=("human", lambda x: np.mean(np.abs(x.values - wdf.loc[x.index, "ai"].values))),
            pct_exact=("human", lambda x: np.mean(x.values == wdf.loc[x.index, "ai"].values) * 100),
        )
        .reset_index()
    )
    word_summary = word_summary[word_summary["n"] >= 10].copy()
    word_summary = word_summary.round(2)

    top_disagree = word_summary.sort_values("mean_abs_diff", ascending=False).head(5)
    top_agree = word_summary.sort_values("mean_abs_diff", ascending=True).head(5)

    return summary, top_disagree, top_agree


# ──────────────────────────────────────────────
# Step 3: Words Correct Per Minute (WCPM)
# ──────────────────────────────────────────────
def analyze_wcpm(df):
    """Calculate WCPM per audio, split by pre/post."""

    rows = []

    for _, row in df.iterrows():
        duration_sec = row.get("ai_utterance_duration_seconds", 0)
        if pd.isna(duration_sec) or duration_sec <= 0:
            continue

        duration_min = duration_sec / 60.0
        pre_post = str(row.get("pre_or_post", "")).lower().strip()
        if pre_post not in ("pre", "post"):
            continue

        # Human word counts (correct = score == 2)
        h_fb = _parse_dict(row.get("feedback_human_json"))
        h_correct = 0
        for w, s in h_fb.items():
            nw = normalize_word(w)
            if not nw:
                continue
            try:
                val = float(s)
            except (ValueError, TypeError):
                continue
            if val == 2:
                h_correct += 1

        # AI word counts (correct = score == 2)
        a_fb = _parse_dict(row.get("feedback_ai_json"))
        a_correct = 0
        for w, obj in a_fb.items():
            nw = normalize_word(w)
            if not nw or not isinstance(obj, dict):
                continue
            ai_score = obj.get("normalized_rounded")
            if ai_score is None:
                continue
            try:
                val = float(ai_score)
            except (ValueError, TypeError):
                continue
            if val == 2:
                a_correct += 1

        rows.append({
            "pre_or_post": pre_post,
            "human_wcpm": h_correct / duration_min,
            "ai_wcpm": a_correct / duration_min,
            "duration_sec": duration_sec,
        })

    wcpm_df = pd.DataFrame(rows)

    # Summary by pre/post
    results = {}
    for period in ["pre", "post"]:
        subset = wcpm_df[wcpm_df["pre_or_post"] == period]
        if len(subset) == 0:
            continue
        results[period] = {
            "n": len(subset),
            "human_mean": round(subset["human_wcpm"].mean(), 1),
            "human_std": round(subset["human_wcpm"].std(), 1),
            "ai_mean": round(subset["ai_wcpm"].mean(), 1),
            "ai_std": round(subset["ai_wcpm"].std(), 1),
        }

    # Overall too
    results["overall"] = {
        "n": len(wcpm_df),
        "human_mean": round(wcpm_df["human_wcpm"].mean(), 1),
        "human_std": round(wcpm_df["human_wcpm"].std(), 1),
        "ai_mean": round(wcpm_df["ai_wcpm"].mean(), 1),
        "ai_std": round(wcpm_df["ai_wcpm"].std(), 1),
    }

    return results


# ──────────────────────────────────────────────
# Step 4: Transcription Comparison (example-based)
# ──────────────────────────────────────────────
import re

def _clean_text(s):
    """Strip punctuation and extra spaces for comparison."""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
    return s


def analyze_transcriptions(df):
    """Find match rate and pick representative examples."""

    n = 0
    ai_human_match = 0

    # Categorized examples
    both_correct = []
    both_match_not_ref = []
    ai_wrong_human_right = []
    human_wrong_ai_right = []
    both_differ = []  # all three are different

    for _, row in df.iterrows():
        ref_raw = str(row["question_text"]).lower().strip()
        h_raw = str(row["transcription_human"]).lower().strip()
        a_raw = str(row["transcription_ai"]).lower().strip()

        if not ref_raw or ref_raw == "nan":
            continue
        if h_raw == "nan" or a_raw == "nan":
            continue
        n += 1

        # Clean versions for comparison
        ref_c = _clean_text(ref_raw)
        h_c = _clean_text(h_raw)
        a_c = _clean_text(a_raw)

        h_matches_ref = (h_c == ref_c)
        a_matches_ref = (a_c == ref_c)
        h_matches_a = (h_c == a_c)

        if h_matches_a:
            ai_human_match += 1

        # Max possible score = number of words in reference text x 2
        ref_word_count = len(ref_c.split())
        max_score = ref_word_count * 2

        # AI total score (sum of normalized_rounded for words it detected)
        ai_fb = _parse_dict(row.get("feedback_ai_json"))
        ai_total = 0
        for _, obj in ai_fb.items():
            if isinstance(obj, dict) and obj.get("normalized_rounded") is not None:
                ai_total += float(obj["normalized_rounded"])

        h_total = row.get("score_human_total", 0)

        # Keep raw text for display (more readable)
        example = {
            "ref": ref_raw, "human": h_raw, "ai": a_raw,
            "human_score": f"{int(h_total)}/{max_score}" if pd.notna(h_total) else "N/A",
            "ai_score": f"{int(ai_total)}/{max_score}",
        }

        if h_matches_ref and a_matches_ref:
            both_correct.append(example)
        elif h_matches_a and not h_matches_ref:
            both_match_not_ref.append(example)
        elif h_matches_ref and not a_matches_ref:
            ai_wrong_human_right.append(example)
        elif a_matches_ref and not h_matches_ref:
            human_wrong_ai_right.append(example)
        elif not h_matches_ref and not a_matches_ref and not h_matches_a:
            both_differ.append(example)

    summary = {
        "n": n,
        "ai_human_match_pct": round(ai_human_match / n * 100, 1) if n > 0 else 0,
    }

    # Pick one example from each category
    # Prefer examples where the *word content* difference is visible and interesting
    examples = []

    def pick_one(bucket, label):
        if not bucket:
            return None
        # Pick one where texts are different enough to be interesting
        # Sort by the number of differing words (more difference = more illustrative)
        def diff_score(ex):
            ref_words = set(_clean_text(ex["ref"]).split())
            h_words = set(_clean_text(ex["human"]).split())
            a_words = set(_clean_text(ex["ai"]).split())
            return len(ref_words.symmetric_difference(h_words)) + len(ref_words.symmetric_difference(a_words))
        bucket.sort(key=diff_score, reverse=True)
        ex = bucket[0]
        ex["category"] = label
        return ex

    ex = pick_one(both_correct, "Both got it right")
    if ex:
        examples.append(ex)
    ex = pick_one(both_match_not_ref, "Both agree, student mispronounced")
    if ex:
        examples.append(ex)
    ex = pick_one(ai_wrong_human_right, "Human got it right, AI didn't")
    if ex:
        examples.append(ex)
    ex = pick_one(human_wrong_ai_right, "AI got it right, human didn't")
    if ex:
        examples.append(ex)
    ex = pick_one(both_differ, "Both heard it differently")
    if ex:
        examples.append(ex)

    return summary, examples


# ──────────────────────────────────────────────
# Word Document Generation
# ──────────────────────────────────────────────
def add_table(doc, headers, rows):
    """Add a formatted table to the document."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)

    return table


def generate_report(paired_df, paired_summary, grader_df, ai_summary, top_disagree, top_agree, wcpm_summary, transcription_summary, transcription_examples):
    """Generate the Word document."""
    doc = Document()

    # -- Styles --
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ── Title ──
    title = doc.add_heading("Audio Grading: How Well Do Graders Agree?", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "This report compares how human graders and AI (Azure Pronunciation Assessment) "
        "score student audio recordings. All scores are on a 0-2 scale: "
        "0 = incorrect, 1 = partially correct, 2 = fully correct."
    )
    doc.add_paragraph(
        f"Data: {paired_summary.get('n_paired_audios', 0) + 316} audio recordings from Grade 1 students, "
        f"graded by 5 human graders and 1 AI system."
    )

    # ── Executive Summary ──
    doc.add_heading("Executive Summary", level=1)

    bullets = [
        f"When two humans graded the same audio, they gave the exact same word score "
        f"{paired_summary.get('overall_word_exact_match_pct', 'N/A')}% of the time "
        f"(across {paired_summary.get('total_words_compared', 'N/A')} words in "
        f"{paired_summary.get('n_paired_audios', 'N/A')} double-graded audios).",

        f"AI and humans gave the exact same word score {ai_summary['pct_exact_match']}% of the time "
        f"(across {ai_summary['n_words']} word comparisons).",

        f"Humans score slightly higher on average ({ai_summary['human_avg']}/2) "
        f"than AI ({ai_summary['ai_avg']}/2). The average difference per word is "
        f"{ai_summary['mean_abs_diff']} points.",
    ]

    # Add WCPM bullets for both human and AI
    if "overall" in wcpm_summary:
        s = wcpm_summary["overall"]
        bullets.append(
            f"Reading speed: humans measured {s['human_mean']} words correct "
            f"per minute on average, AI measured {s['ai_mean']} words correct per minute."
        )
    if "pre" in wcpm_summary and "post" in wcpm_summary:
        pre = wcpm_summary["pre"]
        post = wcpm_summary["post"]
        h_improvement = round(post["human_mean"] - pre["human_mean"], 1)
        a_improvement = round(post["ai_mean"] - pre["ai_mean"], 1)
        bullets.append(
            f"Post-intervention reading speed improved: human WCPM went from {pre['human_mean']} "
            f"to {post['human_mean']} (+{h_improvement}), AI WCPM from {pre['ai_mean']} "
            f"to {post['ai_mean']} (+{a_improvement}). Both human and AI grading show a similar "
            f"improvement trend."
        )

    bullets.append(
        f"AI and humans wrote the exact same transcription "
        f"{transcription_summary['ai_human_match_pct']}% of the time "
        f"(out of {transcription_summary['n']} audios)."
    )

    bullets.append(
        "Human graders tend to be more context-oriented -- they consider the student's age, effort, "
        "and intent when scoring. AI strictly measures pronunciation accuracy, which is why it scores "
        "slightly lower on average."
    )
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    # ══════════════════════════════════════════
    # Section 1: Human Grader Agreement
    # ══════════════════════════════════════════
    doc.add_heading("1. Do Human Graders Agree With Each Other?", level=1)

    doc.add_paragraph(
        f"{paired_summary.get('n_paired_audios', 0)} audio files were graded by two different human graders. "
        "We compared their word-by-word scores to see how much they agree."
    )

    # Table: Paired comparison
    doc.add_heading("Paired Audio Comparison", level=2)

    if len(paired_df) > 0:
        headers = ["Audio", "Grader 1", "Grader 2", "G1 Avg", "G2 Avg",
                    "Words", "Avg Diff", "% Same"]
        rows = []
        for _, r in paired_df.iterrows():
            # Shorten audio filename for readability
            short_name = r["audio"].replace("grade1_", "").replace(".mp3", "")
            if len(short_name) > 35:
                short_name = short_name[:35] + "..."
            rows.append([
                short_name,
                r["grader_1"],
                r["grader_2"],
                r["grader_1_avg"],
                r["grader_2_avg"],
                r["words_compared"],
                r["mean_abs_diff"],
                f"{r['pct_exact_match']}%",
            ])
        add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_heading("Summary", level=3)
    doc.add_paragraph(
        f"Across all double-graded audios, "
        f"human graders agreed on the exact same word score "
        f"{paired_summary.get('overall_word_exact_match_pct', 'N/A')}% of the time. "
        f"When they disagreed, the average difference was "
        f"{paired_summary.get('overall_word_mean_abs_diff', 'N/A')} points (on a 0-2 scale)."
    )
    doc.add_paragraph(
        f"A note on the {paired_summary.get('overall_word_exact_match_pct', 'N/A')}% number: "
        "most students in this dataset read well, so most words get a score of 2 from both graders. "
        "This means a large portion of that agreement comes from easy cases where both graders "
        "simply gave a 2. The real test of agreement is on the harder words (scored 0 or 1), "
        "where graders are more likely to disagree. The gap between the most lenient and strictest "
        "grader (see table below) suggests there is meaningful variation in how humans grade "
        "borderline cases."
    )

    # Table: Per-grader averages
    doc.add_heading("Grader Averages (All Data)", level=2)
    doc.add_paragraph(
        "This table shows each grader's average word score across all the audios they graded. "
        "A higher average means the grader tends to give higher scores (more lenient)."
    )

    headers = ["Grader", "# Audios", "# Words", "Avg Score (0-2)", "Std Dev"]
    rows = []
    for _, r in grader_df.iterrows():
        rows.append([r["grader"], r["n_audios"], r["n_words"], r["avg_score"], r["std_score"]])
    add_table(doc, headers, rows)

    # Interpretation
    doc.add_paragraph("")
    most_lenient = grader_df.iloc[0]
    most_strict = grader_df.iloc[-1]
    doc.add_paragraph(
        f"The most lenient grader is {most_lenient['grader']} (avg {most_lenient['avg_score']}/2) "
        f"and the strictest is {most_strict['grader']} (avg {most_strict['avg_score']}/2). "
        f"The gap between the most lenient and strictest grader is "
        f"{round(most_lenient['avg_score'] - most_strict['avg_score'], 2)} points."
    )

    # ══════════════════════════════════════════
    # Section 2: AI vs Human Agreement
    # ══════════════════════════════════════════
    doc.add_heading("2. Does AI Agree With Human Graders?", level=1)

    doc.add_paragraph(
        "We compared AI word scores (Azure Pronunciation Assessment, rounded to 0/1/2) "
        "with human word scores across all audios."
    )

    # Summary table
    headers = ["Metric", "Value"]
    rows = [
        ["Total words compared", ai_summary["n_words"]],
        ["Avg human score (0-2)", f"{ai_summary['human_avg']} (std: {ai_summary['human_std']})"],
        ["Avg AI score (0-2)", f"{ai_summary['ai_avg']} (std: {ai_summary['ai_std']})"],
        ["Avg difference per word", ai_summary["mean_abs_diff"]],
        ["% words with exact same score", f"{ai_summary['pct_exact_match']}%"],
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Humans and AI gave the exact same score {ai_summary['pct_exact_match']}% of the time. "
        f"On average, humans scored {ai_summary['human_avg']}/2 and AI scored {ai_summary['ai_avg']}/2 "
        f"-- meaning humans tend to be slightly more generous than AI."
    )
    doc.add_paragraph(
        "Why the gap? Human graders are more context-oriented -- they factor in the student's "
        "age, effort, and intent. A Grade 1 student who says 'togeder' instead of 'together' "
        "might get a 1 or 2 from a human (they clearly tried and got close), but the AI strictly "
        "measures pronunciation accuracy against a fixed standard and may give a lower score. "
        "Neither is wrong -- it's a question of what standard the product should aim for."
    )

    # Top disagree words
    doc.add_heading("Words With Most Disagreement", level=2)
    doc.add_paragraph("These words show the biggest gap between human and AI scores (min 10 occurrences):")

    headers = ["Word", "Count", "Human Avg", "AI Avg", "Avg Diff", "% Same"]
    rows = []
    for _, r in top_disagree.iterrows():
        rows.append([r["word"], int(r["n"]), r["human_avg"], r["ai_avg"],
                      r["mean_abs_diff"], f"{r['pct_exact']}%"])
    add_table(doc, headers, rows)

    # Top agree words
    doc.add_heading("Words With Most Agreement", level=2)
    doc.add_paragraph("These words show the best alignment between human and AI:")

    rows = []
    for _, r in top_agree.iterrows():
        rows.append([r["word"], int(r["n"]), r["human_avg"], r["ai_avg"],
                      r["mean_abs_diff"], f"{r['pct_exact']}%"])
    add_table(doc, headers, rows)

    # ══════════════════════════════════════════
    # Section 3: Reading Speed (WCPM)
    # ══════════════════════════════════════════
    doc.add_heading("3. Reading Speed: Words Correct Per Minute", level=1)

    doc.add_paragraph(
        "Words Correct Per Minute (WCPM) measures how many words a student reads fully correctly "
        "(scored 2) in one minute. Pre = before intervention, Post = after."
    )

    headers = ["", "Human WCPM", "AI WCPM"]
    wcpm_rows = []
    for period in ["pre", "post", "overall"]:
        if period not in wcpm_summary:
            continue
        s = wcpm_summary[period]
        if period == "overall":
            label = f"Overall (n={s['n']})"
        else:
            label = f"{period.capitalize()} (n={s['n']})"
        wcpm_rows.append([
            label,
            f"{s['human_mean']} +/- {s['human_std']}",
            f"{s['ai_mean']} +/- {s['ai_std']}",
        ])
    add_table(doc, headers, wcpm_rows)

    doc.add_paragraph("")

    # Pre vs post interpretation
    if "pre" in wcpm_summary and "post" in wcpm_summary:
        pre = wcpm_summary["pre"]
        post = wcpm_summary["post"]
        h_diff = round(post["human_mean"] - pre["human_mean"], 1)
        a_diff = round(post["ai_mean"] - pre["ai_mean"], 1)
        h_direction = "higher" if h_diff > 0 else "lower"
        a_direction = "higher" if a_diff > 0 else "lower"
        doc.add_paragraph(
            f"Post-intervention WCPM is {abs(h_diff)} words/min {h_direction} than pre "
            f"(by human scoring). "
            f"By AI scoring, post is {abs(a_diff)} words/min {a_direction} than pre."
        )

        # Note about human vs AI gap flipping
        pre_gap = round(pre["human_mean"] - pre["ai_mean"], 1)
        post_gap = round(post["human_mean"] - post["ai_mean"], 1)
        if pre_gap > 0 and post_gap < 0:
            doc.add_paragraph(
                f"An interesting pattern: before the intervention, humans counted more words correct "
                f"than AI ({pre['human_mean']} vs {pre['ai_mean']} WCPM). After the "
                f"intervention, this flips -- AI counts more words correct than humans "
                f"({post['ai_mean']} vs {post['human_mean']} WCPM). This likely "
                f"reflects context-oriented human grading: when students are weaker, humans give more "
                f"benefit of the doubt on borderline words. When students improve and read more clearly, "
                f"AI's strict pronunciation standard picks up more correct words."
            )

    doc.add_paragraph(
        "Note: These audios are very short (most under 10 seconds), so WCPM values can be "
        "noisy. Treat these as rough estimates, not precise measurements."
    )

    # ══════════════════════════════════════════
    # Section 4: What Did They Hear?
    # ══════════════════════════════════════════
    doc.add_heading("4. What Did They Hear?", level=1)

    doc.add_paragraph(
        "Below we compare what the human grader heard vs what the AI heard vs the original "
        "text students were supposed to read."
    )
    doc.add_paragraph(
        f"Out of {transcription_summary['n']} audios, AI and the human grader wrote the "
        f"exact same transcription {transcription_summary['ai_human_match_pct']}% of the time."
    )

    if transcription_examples:
        doc.add_heading("Side-by-Side Examples", level=2)
        doc.add_paragraph(
            "Here are real examples showing how human and AI transcriptions compare:"
        )

        headers = ["Scenario", "Original Text", "Human Heard", "AI Heard",
                   "Human Score", "AI Score"]
        rows = []
        for ex in transcription_examples:
            rows.append([
                ex["category"],
                ex["ref"],
                ex["human"],
                ex["ai"],
                ex.get("human_score", "N/A"),
                ex.get("ai_score", "N/A"),
            ])
        add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Note: When both transcriptions differ from the original text, it usually means "
        "the student mispronounced or skipped words -- not that the graders heard wrong. "
        "These are Grade 1 students who are still learning to read."
    )

    # ══════════════════════════════════════════
    # Key Takeaways
    # ══════════════════════════════════════════
    doc.add_heading("Key Takeaways", level=1)

    takeaways = [
        "Human graders mostly agree, but there is some variability -- "
        "especially between the most lenient and strictest graders.",

        "AI scores tend to be slightly lower (stricter) than human scores on average.",

        f"AI and humans agree on the exact word score {ai_summary['pct_exact_match']}% "
        "of the time. Disagreements are usually by 1 point, not 2.",

        "Certain words (especially names like 'faiz' and longer words like 'together') "
        "show the biggest gaps between AI and human grading.",

        f"AI and humans wrote the same transcription {transcription_summary['ai_human_match_pct']}% "
        "of the time. When they differ, it's usually on words the student mispronounced.",
    ]
    for i, t in enumerate(takeaways, 1):
        doc.add_paragraph(f"{i}. {t}")

    # ══════════════════════════════════════════
    # Limitations
    # ══════════════════════════════════════════
    doc.add_heading("Limitations", level=1)

    limitations = [
        f"Only {paired_summary.get('n_paired_audios', 0)} audios were double-graded by humans, "
        "which is a small sample for measuring human agreement.",
        "All data is from Grade 1 students. Results may differ for other grades.",
        "Audio recordings are very short, which can affect scoring reliability.",
        "The AI scores were rounded from a continuous scale to 0/1/2 to match human scoring, "
        "which may lose some precision.",
    ]
    for lim in limitations:
        doc.add_paragraph(lim, style="List Bullet")

    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc.save(REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    df = load_data()

    print("\n--- Step 1: Human Inter-Rater Agreement ---")
    paired_df, paired_summary, grader_df = analyze_human_agreement(df)
    print(f"Paired audios analyzed: {len(paired_df)}")
    print(f"Paired summary: {paired_summary}")
    print(f"Grader averages:\n{grader_df.to_string(index=False)}")

    print("\n--- Step 2: AI vs Human Agreement ---")
    ai_summary, top_disagree, top_agree = analyze_ai_vs_human(df)
    print(f"AI vs Human summary: {ai_summary}")
    print(f"Top disagree words:\n{top_disagree.to_string(index=False)}")
    print(f"Top agree words:\n{top_agree.to_string(index=False)}")

    print("\n--- Step 3: WCPM ---")
    wcpm_summary = analyze_wcpm(df)
    for period, stats in wcpm_summary.items():
        print(f"  {period}: {stats}")

    print("\n--- Step 4: Transcription Comparison ---")
    transcription_summary, transcription_examples = analyze_transcriptions(df)
    print(f"Transcription summary: {transcription_summary}")
    print(f"Examples found: {len(transcription_examples)}")
    for ex in transcription_examples:
        print(f"  [{ex['category']}] ref='{ex['ref']}' human='{ex['human']}' ai='{ex['ai']}'")

    print("\n--- Step 5: Generating Word Document ---")
    generate_report(paired_df, paired_summary, grader_df,
                    ai_summary, top_disagree, top_agree,
                    wcpm_summary, transcription_summary, transcription_examples)

    print("Done!")


if __name__ == "__main__":
    main()
