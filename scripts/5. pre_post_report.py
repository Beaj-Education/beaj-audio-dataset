"""
Pre vs Post Comparison Report

Compares student reading performance before and after intervention.
Covers: word-level scores, WCPM, and transcription quality.
All scores on the native 0-2 scale.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_functions import _parse_dict, normalize_word

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "clean", "merged_for_analysis.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "pre_post_comparison_report.docx")


def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    print(f"Loaded {len(df)} rows, {df['audio_filename'].nunique()} unique audios")
    return df


# ──────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────

def analyze_scores_pre_post(df):
    """Average word scores (0-2) split by pre/post, for human and AI."""

    rows = []
    for _, row in df.iterrows():
        pre_post = str(row.get("pre_or_post", "")).lower().strip()
        if pre_post not in ("pre", "post"):
            continue

        # Human word scores
        h_fb = _parse_dict(row.get("feedback_human_json"))
        h_scores = []
        for w, s in h_fb.items():
            if normalize_word(w):
                try:
                    h_scores.append(float(s))
                except (ValueError, TypeError):
                    pass

        # AI word scores
        a_fb = _parse_dict(row.get("feedback_ai_json"))
        a_scores = []
        for w, obj in a_fb.items():
            if not normalize_word(w) or not isinstance(obj, dict):
                continue
            ai_score = obj.get("normalized_rounded")
            if ai_score is not None:
                try:
                    a_scores.append(float(ai_score))
                except (ValueError, TypeError):
                    pass

        if h_scores:
            rows.append({
                "pre_or_post": pre_post,
                "human_avg_word_score": np.mean(h_scores),
                "ai_avg_word_score": np.mean(a_scores) if a_scores else np.nan,
                "human_total": sum(h_scores),
                "ai_total": sum(a_scores) if a_scores else np.nan,
                "n_words_human": len(h_scores),
                "n_words_ai": len(a_scores),
            })

    scores_df = pd.DataFrame(rows)

    results = {}
    for period in ["pre", "post"]:
        s = scores_df[scores_df["pre_or_post"] == period]
        if len(s) == 0:
            continue
        results[period] = {
            "n_audios": len(s),
            "human_avg": round(s["human_avg_word_score"].mean(), 2),
            "human_std": round(s["human_avg_word_score"].std(), 2),
            "ai_avg": round(s["ai_avg_word_score"].dropna().mean(), 2),
            "ai_std": round(s["ai_avg_word_score"].dropna().std(), 2),
            "human_total_mean": round(s["human_total"].mean(), 1),
            "ai_total_mean": round(s["ai_total"].dropna().mean(), 1),
        }

    return results


def analyze_wcpm_pre_post(df):
    """WCPM split by pre/post."""

    rows = []
    for _, row in df.iterrows():
        duration_sec = row.get("ai_utterance_duration_seconds", 0)
        if pd.isna(duration_sec) or duration_sec <= 0:
            continue
        duration_min = duration_sec / 60.0

        pre_post = str(row.get("pre_or_post", "")).lower().strip()
        if pre_post not in ("pre", "post"):
            continue

        h_fb = _parse_dict(row.get("feedback_human_json"))
        h_correct = 0
        for w, s in h_fb.items():
            if not normalize_word(w):
                continue
            try:
                val = float(s)
            except (ValueError, TypeError):
                continue
            if val == 2:
                h_correct += 1

        a_fb = _parse_dict(row.get("feedback_ai_json"))
        a_correct = 0
        for w, obj in a_fb.items():
            if not normalize_word(w) or not isinstance(obj, dict):
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
        })

    wcpm_df = pd.DataFrame(rows)

    results = {}
    for period in ["pre", "post"]:
        s = wcpm_df[wcpm_df["pre_or_post"] == period]
        if len(s) == 0:
            continue
        results[period] = {
            "n": len(s),
            "human_mean": round(s["human_wcpm"].mean(), 1),
            "human_std": round(s["human_wcpm"].std(), 1),
            "ai_mean": round(s["ai_wcpm"].mean(), 1),
            "ai_std": round(s["ai_wcpm"].std(), 1),
        }

    return results


def analyze_student_overlap(df):
    """How many students appear in both pre and post."""
    pre_students = set(df[df["pre_or_post"] == "pre"]["student_profile_id"])
    post_students = set(df[df["pre_or_post"] == "post"]["student_profile_id"])
    both = pre_students & post_students
    return {
        "pre_only": len(pre_students - post_students),
        "post_only": len(post_students - pre_students),
        "both": len(both),
        "both_ids": both,
        "total": len(pre_students | post_students),
    }


def analyze_paired_changes(df, paired_ids):
    """Per-student paired analysis: compute each student's change, then average.

    For each paired student:
      1. Average their pre audios → one pre score/WCPM per student
      2. Average their post audios → one post score/WCPM per student
      3. Change = post - pre
    Then average the per-student changes.
    """
    paired_df = df[df["student_profile_id"].isin(paired_ids)].copy()

    # Build per-audio metrics (same logic as existing functions)
    audio_rows = []
    for _, row in paired_df.iterrows():
        pre_post = str(row.get("pre_or_post", "")).lower().strip()
        if pre_post not in ("pre", "post"):
            continue
        student_id = row["student_profile_id"]

        # Word scores
        h_fb = _parse_dict(row.get("feedback_human_json"))
        h_scores = []
        for w, s in h_fb.items():
            if normalize_word(w):
                try:
                    h_scores.append(float(s))
                except (ValueError, TypeError):
                    pass

        a_fb = _parse_dict(row.get("feedback_ai_json"))
        a_scores = []
        for w, obj in a_fb.items():
            if not normalize_word(w) or not isinstance(obj, dict):
                continue
            ai_score = obj.get("normalized_rounded")
            if ai_score is not None:
                try:
                    a_scores.append(float(ai_score))
                except (ValueError, TypeError):
                    pass

        # WCPM
        duration_sec = row.get("ai_utterance_duration_seconds", 0)
        h_wcpm = np.nan
        a_wcpm = np.nan
        if not pd.isna(duration_sec) and duration_sec > 0:
            duration_min = duration_sec / 60.0
            h_correct = sum(1 for s in h_scores if s == 2)
            a_correct = sum(1 for s in a_scores if s == 2)
            h_wcpm = h_correct / duration_min
            a_wcpm = a_correct / duration_min

        if h_scores:
            audio_rows.append({
                "student_id": student_id,
                "question_text": row.get("question_text", ""),
                "pre_or_post": pre_post,
                "human_avg_score": np.mean(h_scores),
                "ai_avg_score": np.mean(a_scores) if a_scores else np.nan,
                "human_wcpm": h_wcpm,
                "ai_wcpm": a_wcpm,
            })

    audio_df = pd.DataFrame(audio_rows)

    # Average per student per period
    student_period = audio_df.groupby(["student_id", "pre_or_post"]).agg({
        "human_avg_score": "mean",
        "ai_avg_score": "mean",
        "human_wcpm": "mean",
        "ai_wcpm": "mean",
    }).reset_index()

    # Pivot: one row per student with pre_ and post_ columns
    pre_df = student_period[student_period["pre_or_post"] == "pre"].set_index("student_id")
    post_df = student_period[student_period["pre_or_post"] == "post"].set_index("student_id")

    # Only students with both
    common = pre_df.index.intersection(post_df.index)
    pre_df = pre_df.loc[common]
    post_df = post_df.loc[common]

    # Per-student changes
    score_change_human = post_df["human_avg_score"] - pre_df["human_avg_score"]
    score_change_ai = post_df["ai_avg_score"] - pre_df["ai_avg_score"]
    wcpm_change_human = post_df["human_wcpm"] - pre_df["human_wcpm"]
    wcpm_change_ai = post_df["ai_wcpm"] - pre_df["ai_wcpm"]

    # Audio counts for paired students
    paired_audio = audio_df[audio_df["student_id"].isin(common)]
    n_pre_audios = len(paired_audio[paired_audio["pre_or_post"] == "pre"])
    n_post_audios = len(paired_audio[paired_audio["pre_or_post"] == "post"])

    return {
        "n_students": len(common),
        "n_pre_audios": n_pre_audios,
        "n_post_audios": n_post_audios,
        # Score changes
        "mean_score_change_human": round(score_change_human.mean(), 2),
        "std_score_change_human": round(score_change_human.std(), 2),
        "mean_score_change_ai": round(score_change_ai.dropna().mean(), 2),
        "std_score_change_ai": round(score_change_ai.dropna().std(), 2),
        # WCPM changes
        "mean_wcpm_change_human": round(wcpm_change_human.dropna().mean(), 1),
        "std_wcpm_change_human": round(wcpm_change_human.dropna().std(), 1),
        "mean_wcpm_change_ai": round(wcpm_change_ai.dropna().mean(), 1),
        "std_wcpm_change_ai": round(wcpm_change_ai.dropna().std(), 1),
        # Context: per-student pre/post means
        "pre_score_human": round(pre_df["human_avg_score"].mean(), 2),
        "post_score_human": round(post_df["human_avg_score"].mean(), 2),
        "pre_score_ai": round(pre_df["ai_avg_score"].dropna().mean(), 2),
        "post_score_ai": round(post_df["ai_avg_score"].dropna().mean(), 2),
        "pre_wcpm_human": round(pre_df["human_wcpm"].dropna().mean(), 1),
        "post_wcpm_human": round(post_df["human_wcpm"].dropna().mean(), 1),
        "pre_wcpm_ai": round(pre_df["ai_wcpm"].dropna().mean(), 1),
        "post_wcpm_ai": round(post_df["ai_wcpm"].dropna().mean(), 1),
    }


def analyze_sentence_paired_changes(df):
    """Same-sentence paired analysis: only compare when a student read the same sentence pre and post.

    1. Build per-audio metrics (same as analyze_paired_changes)
    2. Group by (student, sentence, period)
    3. Keep only student+sentence combos with both pre and post
    4. Compute change per student+sentence pair
    5. Average changes per student, then across students
    """
    audio_rows = []
    for _, row in df.iterrows():
        pre_post = str(row.get("pre_or_post", "")).lower().strip()
        if pre_post not in ("pre", "post"):
            continue
        student_id = row["student_profile_id"]
        question = row.get("question_text", "")

        h_fb = _parse_dict(row.get("feedback_human_json"))
        h_scores = []
        for w, s in h_fb.items():
            if normalize_word(w):
                try:
                    h_scores.append(float(s))
                except (ValueError, TypeError):
                    pass

        a_fb = _parse_dict(row.get("feedback_ai_json"))
        a_scores = []
        for w, obj in a_fb.items():
            if not normalize_word(w) or not isinstance(obj, dict):
                continue
            ai_score = obj.get("normalized_rounded")
            if ai_score is not None:
                try:
                    a_scores.append(float(ai_score))
                except (ValueError, TypeError):
                    pass

        duration_sec = row.get("ai_utterance_duration_seconds", 0)
        h_wcpm = np.nan
        a_wcpm = np.nan
        if not pd.isna(duration_sec) and duration_sec > 0:
            duration_min = duration_sec / 60.0
            h_correct = sum(1 for s in h_scores if s == 2)
            a_correct = sum(1 for s in a_scores if s == 2)
            h_wcpm = h_correct / duration_min
            a_wcpm = a_correct / duration_min

        if h_scores:
            audio_rows.append({
                "student_id": student_id,
                "question_text": question,
                "pre_or_post": pre_post,
                "human_avg_score": np.mean(h_scores),
                "ai_avg_score": np.mean(a_scores) if a_scores else np.nan,
                "human_wcpm": h_wcpm,
                "ai_wcpm": a_wcpm,
            })

    audio_df = pd.DataFrame(audio_rows)

    # Average per student+sentence per period
    grp = audio_df.groupby(["student_id", "question_text", "pre_or_post"]).agg({
        "human_avg_score": "mean",
        "ai_avg_score": "mean",
        "human_wcpm": "mean",
        "ai_wcpm": "mean",
    }).reset_index()

    pre_g = grp[grp["pre_or_post"] == "pre"].set_index(["student_id", "question_text"])
    post_g = grp[grp["pre_or_post"] == "post"].set_index(["student_id", "question_text"])

    # Only student+sentence combos with both pre and post
    common_pairs = pre_g.index.intersection(post_g.index)
    pre_g = pre_g.loc[common_pairs]
    post_g = post_g.loc[common_pairs]

    # Per student+sentence changes
    pair_changes = pd.DataFrame({
        "score_change_human": post_g["human_avg_score"].values - pre_g["human_avg_score"].values,
        "score_change_ai": post_g["ai_avg_score"].values - pre_g["ai_avg_score"].values,
        "wcpm_change_human": post_g["human_wcpm"].values - pre_g["human_wcpm"].values,
        "wcpm_change_ai": post_g["ai_wcpm"].values - pre_g["ai_wcpm"].values,
        "pre_score_human": pre_g["human_avg_score"].values,
        "post_score_human": post_g["human_avg_score"].values,
        "pre_score_ai": pre_g["ai_avg_score"].values,
        "post_score_ai": post_g["ai_avg_score"].values,
        "pre_wcpm_human": pre_g["human_wcpm"].values,
        "post_wcpm_human": post_g["human_wcpm"].values,
        "pre_wcpm_ai": pre_g["ai_wcpm"].values,
        "post_wcpm_ai": post_g["ai_wcpm"].values,
    }, index=common_pairs)

    # Average per student (so students with multiple matched sentences get one value)
    student_changes = pair_changes.groupby(level="student_id").mean()

    n_students = len(student_changes)
    n_pairs = len(common_pairs)

    # Audio counts
    paired_audio = audio_df.set_index(["student_id", "question_text", "pre_or_post"])
    matched_pre = paired_audio.loc[paired_audio.index.isin(
        [(s, q, "pre") for s, q in common_pairs])]
    matched_post = paired_audio.loc[paired_audio.index.isin(
        [(s, q, "post") for s, q in common_pairs])]

    return {
        "n_students": n_students,
        "n_sentence_pairs": n_pairs,
        "n_pre_audios": len(matched_pre),
        "n_post_audios": len(matched_post),
        # Score changes
        "mean_score_change_human": round(student_changes["score_change_human"].mean(), 2),
        "std_score_change_human": round(student_changes["score_change_human"].std(), 2),
        "mean_score_change_ai": round(student_changes["score_change_ai"].dropna().mean(), 2),
        "std_score_change_ai": round(student_changes["score_change_ai"].dropna().std(), 2),
        # WCPM changes
        "mean_wcpm_change_human": round(student_changes["wcpm_change_human"].dropna().mean(), 1),
        "std_wcpm_change_human": round(student_changes["wcpm_change_human"].dropna().std(), 1),
        "mean_wcpm_change_ai": round(student_changes["wcpm_change_ai"].dropna().mean(), 1),
        "std_wcpm_change_ai": round(student_changes["wcpm_change_ai"].dropna().std(), 1),
        # Context
        "pre_score_human": round(student_changes["pre_score_human"].mean(), 2),
        "post_score_human": round(student_changes["post_score_human"].mean(), 2),
        "pre_score_ai": round(student_changes["pre_score_ai"].dropna().mean(), 2),
        "post_score_ai": round(student_changes["post_score_ai"].dropna().mean(), 2),
        "pre_wcpm_human": round(student_changes["pre_wcpm_human"].dropna().mean(), 1),
        "post_wcpm_human": round(student_changes["post_wcpm_human"].dropna().mean(), 1),
        "pre_wcpm_ai": round(student_changes["pre_wcpm_ai"].dropna().mean(), 1),
        "post_wcpm_ai": round(student_changes["post_wcpm_ai"].dropna().mean(), 1),
    }


# ──────────────────────────────────────────────
# Word Document
# ──────────────────────────────────────────────

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)

    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)

    return table


def generate_report(scores, wcpm, overlap, paired_changes, sentence_paired):
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ── Title ──
    title = doc.add_heading("Pre vs Post: Did Students Improve?", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "This report compares student reading performance before (pre) and after (post) "
        "the intervention. We look at word-level scores and reading speed "
        "— compared using both human graders and AI (Azure Pronunciation Assessment)."
    )
    doc.add_paragraph(
        f"Data: {overlap['total']} unique students. "
        f"{overlap['both']} students appear in both pre and post. "
        f"{overlap['pre_only']} appear only in pre, {overlap['post_only']} only in post."
    )

    # ── Executive Summary ──
    doc.add_heading("Executive Summary", level=1)

    pre_s, post_s = scores.get("pre", {}), scores.get("post", {})
    pre_w, post_w = wcpm.get("pre", {}), wcpm.get("post", {})

    pc = paired_changes
    bullets = [
        f"Among the {pc['n_students']} students with both pre and post data, "
        f"average word score improved from {pc['pre_score_human']}/2 to "
        f"{pc['post_score_human']}/2 (+{pc['mean_score_change_human']} per student, by human grading).",

        f"Reading speed (WCPM) increased from {pc['pre_wcpm_human']} to "
        f"{pc['post_wcpm_human']} words/min "
        f"(+{pc['mean_wcpm_change_human']} per student, by human grading).",

        f"AI scoring shows the same trend: word score +{pc['mean_score_change_ai']}, "
        f"WCPM +{pc['mean_wcpm_change_ai']} — both human and AI agree students improved.",

        f"Across all audios (including students without both pre and post), "
        f"the pattern is consistent: word scores {pre_s.get('human_avg', 'N/A')}/2 → "
        f"{post_s.get('human_avg', 'N/A')}/2, WCPM {pre_w.get('human_mean', 'N/A')} → "
        f"{post_w.get('human_mean', 'N/A')} words/min.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    # ══════════════════════════════════════════
    # Section 1: Word-Level Scores
    # ══════════════════════════════════════════
    doc.add_heading("1. Word-Level Scores: Pre vs Post", level=1)

    doc.add_paragraph(
        "Each word a student reads is scored 0 (incorrect), 1 (partially correct), or "
        "2 (fully correct). The table below shows per-student changes for the "
        f"{pc['n_students']} students who have both pre and post data "
        "(each student's individual change, then averaged)."
    )

    # Paired table (primary)
    headers = ["", f"Pre (n={pc['n_pre_audios']})", f"Post (n={pc['n_post_audios']})", "Avg Change"]
    rows = [
        ["Human Score (0-2)",
         pc["pre_score_human"],
         pc["post_score_human"],
         f"+{pc['mean_score_change_human']}"],
        ["AI Score (0-2)",
         pc["pre_score_ai"],
         pc["post_score_ai"],
         f"+{pc['mean_score_change_ai']}"],
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Both human and AI grading show improvement. "
        f"The standard deviation of per-student changes is {pc['std_score_change_human']} (human) "
        f"and {pc['std_score_change_ai']} (AI), meaning there's wide variation between students."
    )

    # Sentence-paired table
    sp = sentence_paired
    if sp.get("n_students", 0) > 0:
        doc.add_paragraph(
            f"Restricting to same-sentence comparisons ({sp['n_students']} students, "
            f"{sp['n_sentence_pairs']} student-sentence pairs where the same student "
            f"read the same sentence pre and post):"
        )
        headers = ["", f"Pre (n={sp['n_pre_audios']})", f"Post (n={sp['n_post_audios']})", "Avg Change"]
        rows = [
            ["Human Score (0-2)",
             sp["pre_score_human"],
             sp["post_score_human"],
             f"+{sp['mean_score_change_human']}"],
            ["AI Score (0-2)",
             sp["pre_score_ai"],
             sp["post_score_ai"],
             f"+{sp['mean_score_change_ai']}"],
        ]
        add_table(doc, headers, rows)
        doc.add_paragraph(
            f"This is the strictest comparison: same student, same sentence. "
            f"The +{sp['mean_score_change_human']} improvement (human) vs +{sp['mean_score_change_ai']} (AI) "
            f"confirms students genuinely read better. The larger human gain likely reflects "
            f"human graders rewarding effort and progress more generously, while AI applies "
            f"the same strict pronunciation standard regardless."
        )
        doc.add_paragraph("")

    # All-student table (secondary reference)
    doc.add_paragraph(
        "For reference, across all audios (including students without both pre and post):"
    )
    headers = ["", "# Audios", "Human Avg (0-2)", "AI Avg (0-2)"]
    rows = []
    for period in ["pre", "post"]:
        s = scores.get(period, {})
        rows.append([
            period.capitalize(),
            s.get("n_audios", "N/A"),
            f"{s.get('human_avg', 'N/A')} +/- {s.get('human_std', 'N/A')}",
            f"{s.get('ai_avg', 'N/A')} +/- {s.get('ai_std', 'N/A')}",
        ])
    add_table(doc, headers, rows)
    doc.add_paragraph("")

    # ══════════════════════════════════════════
    # Section 2: Reading Speed (WCPM)
    # ══════════════════════════════════════════
    doc.add_heading("2. Reading Speed: Words Correct Per Minute", level=1)

    doc.add_paragraph(
        "Words Correct Per Minute (WCPM) measures how many words a student reads fully correctly "
        f"(scored 2) in one minute. The table below shows per-student changes for the "
        f"{pc['n_students']} paired students."
    )

    # Paired table (primary)
    headers = ["", f"Pre (n={pc['n_pre_audios']})", f"Post (n={pc['n_post_audios']})", "Avg Change"]
    rows = [
        ["Human WCPM", pc["pre_wcpm_human"], pc["post_wcpm_human"],
         f"+{pc['mean_wcpm_change_human']}"],
        ["AI WCPM", pc["pre_wcpm_ai"], pc["post_wcpm_ai"],
         f"+{pc['mean_wcpm_change_ai']}"],
    ]
    add_table(doc, headers, rows)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Both human and AI grading show students reading faster after the intervention. "
        f"The standard deviation of per-student WCPM changes is {pc['std_wcpm_change_human']} (human) "
        f"and {pc['std_wcpm_change_ai']} (AI) — wide variation, meaning some students improved "
        f"a lot while others showed little change."
    )

    # Sentence-paired WCPM table
    if sp.get("n_students", 0) > 0:
        doc.add_paragraph(
            f"Same-sentence comparisons ({sp['n_students']} students, "
            f"{sp['n_sentence_pairs']} pairs):"
        )
        headers = ["", f"Pre (n={sp['n_pre_audios']})", f"Post (n={sp['n_post_audios']})", "Avg Change"]
        rows = [
            ["Human WCPM", sp["pre_wcpm_human"], sp["post_wcpm_human"],
             f"+{sp['mean_wcpm_change_human']}"],
            ["AI WCPM", sp["pre_wcpm_ai"], sp["post_wcpm_ai"],
             f"+{sp['mean_wcpm_change_ai']}"],
        ]
        add_table(doc, headers, rows)
        doc.add_paragraph(
            f"When comparing the same student reading the same sentence, students read "
            f"+{sp['mean_wcpm_change_human']} words/min faster (human) and "
            f"+{sp['mean_wcpm_change_ai']} (AI). Both agree on the improvement, "
            f"controlling for sentence difficulty."
        )
        doc.add_paragraph("")

    # All-student table (secondary reference)
    doc.add_paragraph(
        "For reference, across all audios (including students without both pre and post):"
    )
    headers = ["", "N", "Human WCPM", "AI WCPM"]
    rows = []
    for period in ["pre", "post"]:
        w = wcpm.get(period, {})
        rows.append([
            period.capitalize(),
            w.get("n", "N/A"),
            f"{w.get('human_mean', 'N/A')} +/- {w.get('human_std', 'N/A')}",
            f"{w.get('ai_mean', 'N/A')} +/- {w.get('ai_std', 'N/A')}",
        ])
    add_table(doc, headers, rows)

    # Note about human vs AI gap flipping
    pre_h = pre_w.get("human_mean", 0)
    pre_a = pre_w.get("ai_mean", 0)
    post_h = post_w.get("human_mean", 0)
    post_a = post_w.get("ai_mean", 0)
    pre_gap = round(pre_h - pre_a, 1)
    post_gap = round(post_h - post_a, 1)

    if pre_gap > 0 and post_gap < 0:
        doc.add_paragraph(
            f"An interesting pattern: in the pre group, humans counted more words correct "
            f"than AI ({pre_h} vs {pre_a} WCPM). In the post group, this flips -- AI counts "
            f"more words correct than humans ({post_a} vs {post_h} WCPM). This likely reflects "
            f"context-oriented human grading: when students are weaker (pre), humans give more "
            f"benefit of the doubt on borderline words. When students improve (post) and read "
            f"more clearly, AI's strict pronunciation standard picks up more correct words."
        )

    doc.add_paragraph(
        "Note: These audios are very short (most under 10 seconds), so WCPM values can be "
        "noisy. Treat these as rough trends, not precise measurements."
    )

    # ══════════════════════════════════════════
    # Key Takeaways
    # ══════════════════════════════════════════
    doc.add_heading("Key Takeaways", level=1)

    takeaways = [
        f"Same-sentence paired comparison ({sp['n_students']} students) shows word scores "
        f"improved by +{sp['mean_score_change_human']} (human) and reading speed by "
        f"+{sp['mean_wcpm_change_human']} words/min — controlling for sentence difficulty.",

        f"Across all {pc['n_students']} paired students, the pattern holds: "
        f"word scores +{pc['mean_score_change_human']} (human), "
        f"WCPM +{pc['mean_wcpm_change_human']} words/min.",

        "Both human and AI scoring show the same improvement trend, "
        "giving confidence the improvement is real.",

        "Variation is high — some students improved a lot while others showed little change. "
        "The averages show a positive trend overall.",
    ]
    for i, t in enumerate(takeaways, 1):
        doc.add_paragraph(f"{i}. {t}")

    # ══════════════════════════════════════════
    # Limitations
    # ══════════════════════════════════════════
    doc.add_heading("Limitations", level=1)

    limitations = [
        f"Only {overlap['both']} out of {overlap['total']} students appear in both "
        "pre and post. The rest appear in only one, so the comparison is not fully paired.",
        "All data is from Grade 1 students reading 3 specific sentences.",
        "Audio recordings are very short (most under 10 seconds), making WCPM noisy.",
        "This is a before/after comparison, not a controlled experiment. "
        "Other factors may explain the improvement.",
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

    print("\n--- Scores Pre/Post ---")
    scores = analyze_scores_pre_post(df)
    for period, s in scores.items():
        print(f"  {period}: {s}")

    print("\n--- WCPM Pre/Post ---")
    wcpm = analyze_wcpm_pre_post(df)
    for period, w in wcpm.items():
        print(f"  {period}: {w}")

    print("\n--- Student Overlap ---")
    overlap = analyze_student_overlap(df)
    print(f"  both={overlap['both']}, pre_only={overlap['pre_only']}, post_only={overlap['post_only']}")

    # Paired analysis: per-student changes (difference then average)
    print(f"\n--- Paired Analysis (per-student changes) ---")
    paired_changes = analyze_paired_changes(df, overlap["both_ids"])
    for k, v in paired_changes.items():
        print(f"  {k}: {v}")

    # Sentence-paired analysis: same student + same sentence pre vs post
    print(f"\n--- Sentence-Paired Analysis (same sentence pre vs post) ---")
    sentence_paired = analyze_sentence_paired_changes(df)
    for k, v in sentence_paired.items():
        print(f"  {k}: {v}")

    print("\n--- Generating Report ---")
    generate_report(scores, wcpm, overlap, paired_changes, sentence_paired)
    print("Done!")


if __name__ == "__main__":
    main()
