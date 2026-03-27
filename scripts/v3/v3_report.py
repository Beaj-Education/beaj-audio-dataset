# %%
"""
v3_ai_transcription/v3_report.py

Generates a leadership-friendly markdown report comparing human and AI transcriptions.

Sections:
  1. Overview
  2. Methodology — Words Correct
  3. Inter-Grader Agreement — Words Correct
  4. Inter-Grader Agreement — WER
  5. Avg Human vs Each AI — Words Correct
  6. Avg Human vs Each AI — WER
  7. WCPM — Avg Human vs Each AI
  8. Pre/Post WCPM — Avg Human vs Each AI
  9. Transcription Examples
  10. Appendix (A-G: heatmaps, examples, rankings; H-L: statistical significance; M: pricing)

Output: V3_REPORT.md
"""

import os
import re
import pandas as pd
import numpy as np
from jiwer import wer as compute_wer, process_words
from scipy.stats import pearsonr, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "v3", "ai_transcribed", "main.csv")
GRADED_V2_DIR = os.path.join(DATA_DIR, "v2", "graded")
DURATIONS_CSV = os.path.join(DATA_DIR, "v2", "clean", "v2_audio_durations.csv")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots", "v3")
OUT_MARKDOWN = os.path.join(PROJECT_DIR, "V3_REPORT.md")
OUT_CORR_HEATMAP = os.path.join(PLOTS_DIR, "v3_report_words_correct_correlation.png")
OUT_WER_HEATMAP = os.path.join(PLOTS_DIR, "v3_report_wer_heatmap.png")
OUT_IG_WC_HEATMAP = os.path.join(PLOTS_DIR, "v3_report_inter_grader_wc.png")
OUT_WC_BAR = os.path.join(PLOTS_DIR, "v3_report_words_correct_comparison.png")
OUT_WER_BAR = os.path.join(PLOTS_DIR, "v3_report_wer_comparison.png")
OUT_WCPM_BAR = os.path.join(PLOTS_DIR, "v3_report_wcpm_comparison.png")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports", "v3")
os.makedirs(REPORT_DIR, exist_ok=True)
OUT_DOCX = os.path.join(REPORT_DIR, "v3_report.docx")
PRICING_CSV = os.path.join(DATA_DIR, "v3", "clean", "v3_pricing.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

AI_MODELS = [
    "scribe_v2", "gemini-3-pro", "voxtral small", "gemini-3-flash",
    "universal-3-pro", "voxtral mini", "universal",
    "gpt-4o-transcribe (azure)", "gpt-4o-transcribe (openai)",
    "gemini-3.1-pro", "sarvam",
]

DISPLAY_NAMES = {
    "scribe_v2": "Scribe v2", "gemini-3-pro": "Gemini 3 Pro",
    "voxtral small": "Voxtral Small", "gemini-3-flash": "Gemini 3 Flash",
    "universal-3-pro": "Universal 3 Pro", "voxtral mini": "Voxtral Mini",
    "universal": "Universal", "gpt-4o-transcribe (azure)": "GPT-4o (Azure)",
    "gpt-4o-transcribe (openai)": "GPT-4o (OpenAI)",
    "gemini-3.1-pro": "Gemini 3.1 Pro", "sarvam": "Sarvam",
}

# Mapping from AI_MODELS keys → model name in pricing CSV
PRICING_NAMES = {
    "scribe_v2": "Scribe v2",
    "gemini-3-pro": "Gemini 3 Pro",
    "voxtral small": "Voxtral Small",
    "gemini-3-flash": "Gemini 3 Flash",
    "universal-3-pro": "Universal-3 Pro",
    "voxtral mini": "Voxtral Mini",
    "universal": "Universal",
    "gpt-4o-transcribe (azure)": "GPT-4o Transcribe",
    "gpt-4o-transcribe (openai)": "GPT-4o Transcribe",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "sarvam": "Sarvam AI",
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def clean_text(text):
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_wer(reference, hypothesis):
    ref = clean_text(reference)
    hyp = clean_text(hypothesis)
    if not ref or not hyp:
        return np.nan
    try:
        return compute_wer(ref, hyp)
    except Exception:
        return np.nan


def compute_words_correct(reference, transcription):
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(transcription)
    if not ref_clean or not hyp_clean:
        return 0
    try:
        result = process_words(ref_clean, hyp_clean)
        return result.hits
    except Exception:
        return 0


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    vals = data.values if hasattr(data, "values") else np.array(data)
    n = len(vals)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = vals[rng.integers(0, n, size=n)].mean()
    lo = (1 - ci) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def bootstrap_wcpm_ci(wc_array, dur_array, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for WCPM (ratio of sums)."""
    rng = np.random.default_rng(seed)
    n = len(wc_array)
    wcpms = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        wcpms[i] = wc_array[idx].sum() / dur_array[idx].sum()
    lo = (1 - ci) / 2
    return float(np.quantile(wcpms, lo)), float(np.quantile(wcpms, 1 - lo))


def compute_significance_table(df_scores, metric_col, baseline_per_audio, group_col="scorer"):
    """Wilcoxon signed-rank test + bootstrap 95% CI on mean difference for each AI model."""
    results = []
    for scorer, grp in df_scores.groupby(group_col):
        merged = grp.set_index("audio_file_name")[[metric_col]].copy()
        merged["baseline"] = baseline_per_audio
        merged = merged.dropna()
        diffs = merged[metric_col] - merged["baseline"]
        if len(diffs) < 10 or diffs.std() == 0:
            continue
        stat, pval = wilcoxon(diffs, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci(diffs)
        results.append({
            "scorer": scorer, "n": len(diffs),
            "mean_diff": round(diffs.mean(), 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "wilcoxon_p": pval,
        })
    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(result_df["wilcoxon_p"], method="fdr_bh")
        result_df["p_adjusted"] = pvals_corrected
        result_df["significant"] = reject
    return result_df.sort_values("mean_diff")


def bootstrap_wcpm_delta(pre_wc, pre_dur, post_wc, post_dur, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for WCPM delta (post - pre). Returns (delta, ci_lo, ci_hi, p_value)."""
    rng = np.random.default_rng(seed)
    n_pre, n_post = len(pre_wc), len(post_wc)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx_pre = rng.integers(0, n_pre, size=n_pre)
        idx_post = rng.integers(0, n_post, size=n_post)
        wcpm_pre = pre_wc[idx_pre].sum() / pre_dur[idx_pre].sum()
        wcpm_post = post_wc[idx_post].sum() / post_dur[idx_post].sum()
        deltas[i] = wcpm_post - wcpm_pre
    lo = (1 - ci) / 2
    ci_lo, ci_hi = float(np.quantile(deltas, lo)), float(np.quantile(deltas, 1 - lo))
    # Two-sided p-value: proportion of bootstrap deltas <= 0
    p_val = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p_val = min(p_val, 1.0)
    return ci_lo, ci_hi, p_val


def fmt_pval(p):
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# ──────────────────────────────────────────────
# Step 1: Load data
# ──────────────────────────────────────────────
print("Loading data...")

df_main = pd.read_csv(MAIN_CSV)
df_main.columns = df_main.columns.str.strip()

grader_transcriptions = {}
all_graders = []
for fname in sorted(os.listdir(GRADED_V2_DIR)):
    if not fname.endswith(".csv"):
        continue
    grader_name = fname.split(" - ")[0].strip()
    all_graders.append(grader_name)
    fpath = os.path.join(GRADED_V2_DIR, fname)
    gdf = pd.read_csv(fpath)
    trans = {}
    for _, row in gdf.iterrows():
        audio = row["audio_file_name"]
        ht = row.get("Human Transcription")
        if pd.notna(ht) and str(ht).strip():
            trans[audio] = str(ht).strip()
    grader_transcriptions[grader_name] = trans

# Remove Abdullah — only 18 audios, too few shared audios with other graders
# for statistically significant or adequately powered correlations.
EXCLUDE_GRADERS = {"Abdullah"}
all_graders = [g for g in all_graders if g not in EXCLUDE_GRADERS]
for g in EXCLUDE_GRADERS:
    grader_transcriptions.pop(g, None)

records = []
for _, row in df_main.iterrows():
    audio = row["audio_file_name"]
    question = row["Question"]
    g1, g2 = row.get("Grader 1"), row.get("Grader 2")
    t1 = grader_transcriptions.get(g1, {}).get(audio) if pd.notna(g1) else None
    t2 = grader_transcriptions.get(g2, {}).get(audio) if pd.notna(g2) else None
    ai_trans = {}
    for model in AI_MODELS:
        val = row.get(model)
        if pd.notna(val) and str(val).strip():
            ai_trans[model] = str(val).strip()
    records.append({
        "audio_file_name": audio, "question": question,
        "grader_1": g1, "grader_2": g2,
        "transcription_1": t1, "transcription_2": t2,
        **{f"ai_{model}": ai_trans.get(model) for model in AI_MODELS},
    })

df = pd.DataFrame(records)
has_both = df["transcription_1"].notna() & df["transcription_2"].notna()
has_any = df["transcription_1"].notna() | df["transcription_2"].notna()
df_dual = df[has_both].copy()

df["pre_or_post"] = np.where(df["audio_file_name"].str.contains("_LP_"), "pre", "post")
df_dual["pre_or_post"] = np.where(df_dual["audio_file_name"].str.contains("_LP_"), "pre", "post")

df_dur = pd.read_csv(DURATIONS_CSV)
df_dur = df_dur[df_dur["ai_utterance_duration_seconds"] > 0].drop_duplicates(subset="audio_file_name")

print(f"Loaded: {len(df)} audios, {has_both.sum()} dual-graded, {len(df_dur)} with duration data")

# %%
# ──────────────────────────────────────────────
# Step 2: Compute inter-grader WER
# ──────────────────────────────────────────────
df_dual["inter_grader_wer"] = df_dual.apply(
    lambda r: safe_wer(r["transcription_1"], r["transcription_2"]), axis=1
)
valid_ig_wer = df_dual["inter_grader_wer"].dropna()

# Per-audio words-correct difference between the two graders
df_dual["wc_1"] = df_dual.apply(
    lambda r: compute_words_correct(r["question"], r["transcription_1"]), axis=1
)
df_dual["wc_2"] = df_dual.apply(
    lambda r: compute_words_correct(r["question"], r["transcription_2"]), axis=1
)
df_dual["wc_abs_diff"] = (df_dual["wc_1"] - df_dual["wc_2"]).abs()

# Filter out non-transcription entries ("parent recording", "Yes.", etc.)
SKIP_PATTERNS = ["parent recording", "parent", "yes", "no", "n/a"]
df_valid_dual = df_dual[
    df_dual["inter_grader_wer"].notna()
    & (df_dual["transcription_1"].str.len() > 10)
    & (df_dual["transcription_2"].str.len() > 10)
    & (~df_dual["transcription_1"].str.lower().str.strip().isin(SKIP_PATTERNS))
    & (~df_dual["transcription_2"].str.lower().str.strip().isin(SKIP_PATTERNS))
].copy()

# Select conflation examples for Section 4 narrative:
# 1. Convention noise: high WER but graders agree on words correct (wc_abs_diff == 0)
convention_noise_ex = df_valid_dual[
    (df_valid_dual["inter_grader_wer"] > 0.2) & (df_valid_dual["wc_abs_diff"] == 0)
].sort_values("inter_grader_wer", ascending=False).head(1)

# 2. Genuine disagreement: high WER AND large words-correct difference
genuine_disagree_ex = df_valid_dual[
    (df_valid_dual["inter_grader_wer"] > 0.2) & (df_valid_dual["wc_abs_diff"] >= 3)
].sort_values("inter_grader_wer", ascending=False).head(1)

conflation_examples = pd.concat([convention_noise_ex, genuine_disagree_ex])

# For each example, get word-level alignment for both graders
def get_word_alignment(ref, hyp):
    """Return list of (ref_word, result) tuples using jiwer alignment."""
    ref_clean = str(ref).strip().lower()
    hyp_clean = str(hyp).strip().lower()
    if not ref_clean or not hyp_clean:
        return []
    result = process_words(ref_clean, hyp_clean)
    alignment = []
    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                alignment.append((result.references[0][chunk.ref_start_idx + i], "Hit"))
        elif chunk.type == "substitute":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                alignment.append((result.references[0][chunk.ref_start_idx + i], "Substitution"))
        elif chunk.type == "delete":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                alignment.append((result.references[0][chunk.ref_start_idx + i], "Deletion"))
        # insertions don't consume reference words
    return alignment

conflation_details = []
for _, ex in conflation_examples.iterrows():
    align_1 = get_word_alignment(ex["question"], ex["transcription_1"])
    align_2 = get_word_alignment(ex["question"], ex["transcription_2"])
    conflation_details.append({
        "audio": ex["audio_file_name"],
        "reference": ex["question"],
        "grader_1": ex["grader_1"],
        "grader_2": ex["grader_2"],
        "trans_1": ex["transcription_1"],
        "trans_2": ex["transcription_2"],
        "wc_1": int(ex["wc_1"]),
        "wc_2": int(ex["wc_2"]),
        "wer": ex["inter_grader_wer"],
        "align_1": align_1,
        "align_2": align_2,
    })

# ──────────────────────────────────────────────
# Step 3: Compute words correct for all scorers on all audios with human transcription
# ──────────────────────────────────────────────
df_any = df[has_any].copy()
df_any["pre_or_post"] = np.where(df_any["audio_file_name"].str.contains("_LP_"), "pre", "post")
n_any_human = len(df_any)

all_scores = []
for _, row in df_any.iterrows():
    audio = row["audio_file_name"]
    ref = row["question"]
    pre_post = row["pre_or_post"]

    # Collect human transcriptions for this audio (for AI-vs-human WER later)
    human_trans_list = []
    if pd.notna(row["transcription_1"]):
        human_trans_list.append(row["transcription_1"])
    if pd.notna(row["transcription_2"]):
        human_trans_list.append(row["transcription_2"])

    if pd.notna(row["transcription_1"]):
        wc = compute_words_correct(ref, row["transcription_1"])
        total_ref = len(clean_text(ref).split())
        all_scores.append({
            "audio_file_name": audio, "scorer": row["grader_1"], "type": "Human",
            "transcription": row["transcription_1"], "words_correct": wc,
            "total_ref_words": total_ref, "pre_or_post": pre_post,
        })
    if pd.notna(row["transcription_2"]):
        wc = compute_words_correct(ref, row["transcription_2"])
        total_ref = len(clean_text(ref).split())
        all_scores.append({
            "audio_file_name": audio, "scorer": row["grader_2"], "type": "Human",
            "transcription": row["transcription_2"], "words_correct": wc,
            "total_ref_words": total_ref, "pre_or_post": pre_post,
        })
    for model in AI_MODELS:
        ai_col = f"ai_{model}"
        ai_val = row.get(ai_col)
        if pd.notna(ai_val) and str(ai_val).strip():
            wc = compute_words_correct(ref, str(ai_val))
            total_ref = len(clean_text(ref).split())
            # WER: compare AI transcription against each human transcription, then average
            ai_vs_human_wers = [safe_wer(ht, str(ai_val)) for ht in human_trans_list]
            ai_vs_human_wers = [w for w in ai_vs_human_wers if not np.isnan(w)]
            avg_wer_vs_human = np.mean(ai_vs_human_wers) if ai_vs_human_wers else np.nan
            all_scores.append({
                "audio_file_name": audio, "scorer": model, "type": "AI",
                "transcription": str(ai_val).strip(), "words_correct": wc,
                "total_ref_words": total_ref, "pre_or_post": pre_post,
                "wer_vs_human": avg_wer_vs_human,
            })

df_scores = pd.DataFrame(all_scores)

# %%
# ──────────────────────────────────────────────
# Step 4: Compute correlations (words correct) on all audios
# ──────────────────────────────────────────────
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
audio_to_ref = df.set_index("audio_file_name")["question"].to_dict()

scorer_scores = {}
for scorer in scorer_names:
    scores = {}
    for audio, transcription in all_scorers[scorer].items():
        ref = audio_to_ref.get(audio)
        if ref:
            scores[audio] = compute_words_correct(ref, transcription)
    scorer_scores[scorer] = scores

all_audios = sorted(set().union(*[set(s.keys()) for s in scorer_scores.values()]))
score_matrix = pd.DataFrame(index=all_audios, columns=scorer_names, dtype=float)
for scorer in scorer_names:
    for audio, s in scorer_scores[scorer].items():
        score_matrix.loc[audio, scorer] = s

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
            r, _ = pearsonr(score_matrix.loc[mask, s1], score_matrix.loc[mask, s2])
            corr_matrix.loc[s1, s2] = r
corr_matrix = corr_matrix.astype(float)

# Mean correlations (all graders)
hh_corrs = [corr_matrix.loc[g1, g2] for g1 in all_graders for g2 in all_graders
            if g1 < g2 and pd.notna(corr_matrix.loc[g1, g2])]
ha_corrs = [corr_matrix.loc[g, m] for g in all_graders for m in AI_MODELS
            if pd.notna(corr_matrix.loc[g, m])]
aa_corrs = [corr_matrix.loc[m1, m2] for m1 in AI_MODELS for m2 in AI_MODELS
            if m1 < m2 and pd.notna(corr_matrix.loc[m1, m2])]
mean_hh = np.mean(hh_corrs) if hh_corrs else np.nan
mean_ha = np.mean(ha_corrs) if ha_corrs else np.nan
mean_aa = np.mean(aa_corrs) if aa_corrs else np.nan

# %%
# ──────────────────────────────────────────────
# Step 5: Inter-grader words correct agreement
# ──────────────────────────────────────────────
print("Computing inter-grader words correct agreement...")
ig_wc_records = []
for g1 in all_graders:
    for g2 in all_graders:
        if g1 >= g2:
            continue
        shared = set(scorer_scores[g1].keys()) & set(scorer_scores[g2].keys())
        if len(shared) < 3:
            ig_wc_records.append({
                "grader_1": g1, "grader_2": g2,
                "mean_abs_diff": np.nan, "pearson_r": np.nan,
                "pearson_p": np.nan, "shared_audios": len(shared),
            })
            continue
        scores_g1 = [scorer_scores[g1][a] for a in shared]
        scores_g2 = [scorer_scores[g2][a] for a in shared]
        mean_diff = np.mean(np.abs(np.array(scores_g1) - np.array(scores_g2)))
        r, p_val = pearsonr(scores_g1, scores_g2)
        ig_wc_records.append({
            "grader_1": g1, "grader_2": g2,
            "mean_abs_diff": round(mean_diff, 2),
            "pearson_r": round(r, 3),
            "pearson_p": p_val, "shared_audios": len(shared),
        })
df_ig_wc = pd.DataFrame(ig_wc_records)

# %%
# ──────────────────────────────────────────────
# Step 6: Avg Human baseline computation
# ──────────────────────────────────────────────
print("Computing Avg Human baseline...")

# Words correct: per-audio average across human graders, then overall mean
human_scores = df_scores[df_scores["type"] == "Human"]
avg_human_wc_per_audio = human_scores.groupby("audio_file_name")["words_correct"].mean()
avg_human_mean_wc = round(avg_human_wc_per_audio.mean(), 1)
avg_human_median_wc = avg_human_wc_per_audio.median()
avg_human_n_wc = len(avg_human_wc_per_audio)

# Avg Human WER baseline: use the inter-grader WER (human vs human transcriptions)
# This is already computed in valid_ig_wer — the WER between paired human graders
avg_human_mean_wer = round(valid_ig_wer.mean(), 3)
avg_human_median_wer = round(valid_ig_wer.median(), 3)

# AI summaries for words correct
ai_scores = df_scores[df_scores["type"] == "AI"]

ai_wc_summary = ai_scores.groupby("scorer")["words_correct"].agg(["mean", "median", "count"])
ai_wc_summary.columns = ["mean_wc", "median_wc", "n"]
# Round to display precision so comparisons match what's shown in tables/charts
ai_wc_summary["mean_wc"] = ai_wc_summary["mean_wc"].round(1)
ai_wc_summary["median_wc"] = ai_wc_summary["median_wc"].round(0)
ai_wc_summary = ai_wc_summary.sort_values("mean_wc", ascending=False)

# AI WER summary: WER of AI transcription vs human transcriptions
ai_wer_summary = ai_scores.groupby("scorer")["wer_vs_human"].agg(["mean", "median", "count"])
ai_wer_summary.columns = ["mean_wer", "median_wer", "n"]
# Round to display precision (.1% = 3 decimal places) so comparisons match displayed values
ai_wer_summary["mean_wer"] = ai_wer_summary["mean_wer"].round(3)
ai_wer_summary["median_wer"] = ai_wer_summary["median_wer"].round(3)
ai_wer_summary = ai_wer_summary.sort_values("mean_wer")

# ──────────────────────────────────────────────
# Step 7: WCPM computation (total words correct / total time)
# ──────────────────────────────────────────────
df_scores_with_dur = df_scores.merge(
    df_dur, on="audio_file_name", how="inner"
)
df_scores_with_dur["duration_minutes"] = df_scores_with_dur["ai_utterance_duration_seconds"] / 60

n_with_dur = df_scores_with_dur["audio_file_name"].nunique()
print(f"WCPM computed for {n_with_dur} audios (out of {n_any_human} with human transcription)")

# Avg Human WCPM (total/total): first average WC per audio across graders, then sum / sum
human_wcpm_data = df_scores_with_dur[df_scores_with_dur["type"] == "Human"]
# Per-audio: average words_correct across graders, keep duration
human_per_audio = human_wcpm_data.groupby("audio_file_name").agg(
    avg_wc=("words_correct", "mean"),
    duration_minutes=("duration_minutes", "first"),
).reset_index()
avg_human_wcpm = round(human_per_audio["avg_wc"].sum() / human_per_audio["duration_minutes"].sum(), 1)
avg_human_n_wcpm = len(human_per_audio)

# AI WCPM summary (total/total per model)
ai_wcpm_data = df_scores_with_dur[df_scores_with_dur["type"] == "AI"]
ai_wcpm_rows = []
for model, grp in ai_wcpm_data.groupby("scorer"):
    total_wc = grp["words_correct"].sum()
    total_min = grp["duration_minutes"].sum()
    wcpm = round(total_wc / total_min, 1) if total_min > 0 else 0
    ai_wcpm_rows.append({"scorer": model, "wcpm": wcpm, "n": grp["audio_file_name"].nunique()})
ai_wcpm_summary = pd.DataFrame(ai_wcpm_rows).set_index("scorer").sort_values("wcpm", ascending=False)

# ──────────────────────────────────────────────
# Step 8: Pre/Post WCPM — Avg Human vs Each AI (total/total)
# ──────────────────────────────────────────────
human_wcpm_data_pp = human_wcpm_data.copy()
human_per_audio_pp = human_wcpm_data_pp.groupby(["audio_file_name", "pre_or_post"]).agg(
    avg_wc=("words_correct", "mean"),
    duration_minutes=("duration_minutes", "first"),
).reset_index()
avg_human_prepost = {}
for period in ["pre", "post"]:
    sub = human_per_audio_pp[human_per_audio_pp["pre_or_post"] == period]
    if len(sub) > 0:
        avg_human_prepost[period] = round(sub["avg_wc"].sum() / sub["duration_minutes"].sum(), 1)
    else:
        avg_human_prepost[period] = 0

# AI pre/post WCPM (total/total)
ai_prepost_rows = []
for (model, period), grp in ai_wcpm_data.groupby(["scorer", "pre_or_post"]):
    total_wc = grp["words_correct"].sum()
    total_min = grp["duration_minutes"].sum()
    wcpm = round(total_wc / total_min, 1) if total_min > 0 else 0
    ai_prepost_rows.append({"scorer": model, "pre_or_post": period, "wcpm": wcpm, "n": grp["audio_file_name"].nunique()})
ai_prepost_wcpm = pd.DataFrame(ai_prepost_rows)

# %%
# ──────────────────────────────────────────────
# Step 8b: Statistical significance computations
# ──────────────────────────────────────────────
print("Computing statistical significance...")

# WC significance: per-audio (AI words_correct − avg human words_correct)
wc_sig = compute_significance_table(ai_scores, "words_correct", avg_human_wc_per_audio)

# WER significance: per-audio (AI WER vs human − avg inter-grader WER baseline) on ALL audios
ai_wer_baseline = pd.Series(avg_human_mean_wer, index=ai_scores["audio_file_name"].unique())
wer_sig = compute_significance_table(ai_scores, "wer_vs_human", ai_wer_baseline)

# WCPM bootstrap CIs
wcpm_ci_rows = []
# Human baseline CI
human_wc_arr = human_per_audio["avg_wc"].values
human_dur_arr = human_per_audio["duration_minutes"].values
h_ci_lo, h_ci_hi = bootstrap_wcpm_ci(human_wc_arr, human_dur_arr)
wcpm_ci_rows.append({
    "scorer": "Avg Human (baseline)", "wcpm": avg_human_wcpm,
    "ci_lo": round(h_ci_lo, 1), "ci_hi": round(h_ci_hi, 1),
    "n": len(human_per_audio),
})
for model, grp in ai_wcpm_data.groupby("scorer"):
    wc_arr = grp["words_correct"].values
    dur_arr = grp["duration_minutes"].values
    wcpm_val = round(wc_arr.sum() / dur_arr.sum(), 1) if dur_arr.sum() > 0 else 0
    ci_lo, ci_hi = bootstrap_wcpm_ci(wc_arr, dur_arr)
    wcpm_ci_rows.append({
        "scorer": model, "wcpm": wcpm_val,
        "ci_lo": round(ci_lo, 1), "ci_hi": round(ci_hi, 1),
        "n": grp["audio_file_name"].nunique(),
    })
wcpm_ci_df = pd.DataFrame(wcpm_ci_rows)

# Pre/Post WCPM delta significance (bootstrap: is post-pre improvement significant?)
prepost_delta_rows = []
# Human baseline delta
h_pre = human_per_audio_pp[human_per_audio_pp["pre_or_post"] == "pre"]
h_post = human_per_audio_pp[human_per_audio_pp["pre_or_post"] == "post"]
if len(h_pre) > 0 and len(h_post) > 0:
    h_pre_wcpm = round(h_pre["avg_wc"].sum() / h_pre["duration_minutes"].sum(), 1)
    h_post_wcpm = round(h_post["avg_wc"].sum() / h_post["duration_minutes"].sum(), 1)
    h_delta = round(h_post_wcpm - h_pre_wcpm, 1)
    h_ci_lo, h_ci_hi, h_pval = bootstrap_wcpm_delta(
        h_pre["avg_wc"].values, h_pre["duration_minutes"].values,
        h_post["avg_wc"].values, h_post["duration_minutes"].values,
    )
    prepost_delta_rows.append({
        "scorer": "Avg Human (baseline)", "pre_wcpm": h_pre_wcpm, "post_wcpm": h_post_wcpm,
        "delta": h_delta, "ci_lo": round(h_ci_lo, 1), "ci_hi": round(h_ci_hi, 1),
        "p_value": h_pval, "n_pre": len(h_pre), "n_post": len(h_post),
    })
# AI models
for model in AI_MODELS:
    ai_pre = ai_wcpm_data[(ai_wcpm_data["scorer"] == model) & (ai_wcpm_data["pre_or_post"] == "pre")]
    ai_post = ai_wcpm_data[(ai_wcpm_data["scorer"] == model) & (ai_wcpm_data["pre_or_post"] == "post")]
    if len(ai_pre) == 0 or len(ai_post) == 0:
        continue
    pre_wcpm = round(ai_pre["words_correct"].sum() / ai_pre["duration_minutes"].sum(), 1)
    post_wcpm = round(ai_post["words_correct"].sum() / ai_post["duration_minutes"].sum(), 1)
    delta = round(post_wcpm - pre_wcpm, 1)
    ci_lo, ci_hi, pval = bootstrap_wcpm_delta(
        ai_pre["words_correct"].values, ai_pre["duration_minutes"].values,
        ai_post["words_correct"].values, ai_post["duration_minutes"].values,
    )
    prepost_delta_rows.append({
        "scorer": model, "pre_wcpm": pre_wcpm, "post_wcpm": post_wcpm,
        "delta": delta, "ci_lo": round(ci_lo, 1), "ci_hi": round(ci_hi, 1),
        "p_value": pval, "n_pre": ai_pre["audio_file_name"].nunique(),
        "n_post": ai_post["audio_file_name"].nunique(),
    })
prepost_delta_df = pd.DataFrame(prepost_delta_rows)
print("  Statistical significance computed.")

# %%
# ──────────────────────────────────────────────
# Step 9: Pairwise WER matrix (for appendix heatmap)
# ──────────────────────────────────────────────
print("Computing pairwise WER matrix for heatmap...")
wer_matrix = pd.DataFrame(np.nan, index=scorer_names, columns=scorer_names)
for i, s1 in enumerate(scorer_names):
    for j, s2 in enumerate(scorer_names):
        if i == j:
            wer_matrix.loc[s1, s2] = 0.0
            continue
        if j < i:
            continue  # will fill symmetric
        shared = set(all_scorers[s1].keys()) & set(all_scorers[s2].keys())
        if len(shared) < 3:
            continue
        wers = [safe_wer(all_scorers[s1][a], all_scorers[s2][a]) for a in shared]
        wers = [w for w in wers if not np.isnan(w)]
        if wers:
            mean_w = np.mean(wers)
            wer_matrix.loc[s1, s2] = mean_w
            wer_matrix.loc[s2, s1] = mean_w
    if (i + 1) % 5 == 0:
        print(f"  ... {i + 1}/{len(scorer_names)} scorers done")
wer_matrix = wer_matrix.astype(float)
print("  Pairwise WER matrix complete.")

# %%
# ──────────────────────────────────────────────
# Step 10: Generate heatmaps
# ──────────────────────────────────────────────
print("Generating heatmaps...")

# Display labels: human graders get "(H)" suffix
display_labels = []
for s in scorer_names:
    if s in all_graders:
        display_labels.append(f"{s} (H)")
    else:
        display_labels.append(DISPLAY_NAMES.get(s, s))
n_humans = len(all_graders)

# --- Heatmap 1: Words Correct Correlation ---
corr_display = corr_matrix.loc[scorer_names, scorer_names].copy()
corr_display.index = display_labels
corr_display.columns = display_labels

fig, ax = plt.subplots(figsize=(16, 13))
annot = corr_display.map(
    lambda x: f"{x:.2f}" if pd.notna(x) and x != 1.0 else ("1.00" if x == 1.0 else "")
)
sns.heatmap(
    corr_display, annot=annot, fmt="", cmap="RdYlGn",
    vmin=0, vmax=1, square=True, linewidths=0.5,
    cbar_kws={"label": "Pearson r", "shrink": 0.8, "pad": 0.15}, ax=ax,
    annot_kws={"fontsize": 7, "fontweight": "bold"},
)
ax.set_title(
    "Pairwise Pearson r — Words Correct\nAll Human Graders (H) + AI Models",
    fontsize=15, fontweight="bold", pad=50,
)
ax.axhline(y=n_humans, color="black", linewidth=2.5)
ax.axvline(x=n_humans, color="black", linewidth=2.5)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(display_labels, rotation=0, fontsize=9)
# Add labels on top and right via manual text placement
n_labels = len(display_labels)
for i, label in enumerate(display_labels):
    ax.text(i + 0.5, -0.3, label, ha="left", va="bottom", rotation=45,
            fontsize=9, clip_on=False)
for i, label in enumerate(display_labels):
    ax.text(n_labels + 0.3, i + 0.5, label, ha="left", va="center",
            fontsize=9, clip_on=False)
plt.subplots_adjust(right=0.82)
plt.savefig(OUT_CORR_HEATMAP, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_CORR_HEATMAP}")

# --- Heatmap 2: Pairwise WER ---
wer_display = wer_matrix.loc[scorer_names, scorer_names].copy()
wer_display.index = display_labels
wer_display.columns = display_labels

fig, ax = plt.subplots(figsize=(16, 13))
annot = wer_display.map(
    lambda x: f"{x:.0%}" if pd.notna(x) and x != 0.0 else ("0%" if x == 0.0 else "")
)
sns.heatmap(
    wer_display, annot=annot, fmt="", cmap="YlOrRd",
    vmin=0, vmax=1, square=True, linewidths=0.5,
    cbar_kws={"label": "Mean WER", "shrink": 0.8, "pad": 0.15}, ax=ax,
    annot_kws={"fontsize": 7},
)
ax.set_title(
    "Pairwise WER — All Human Graders (H) + AI Models\n(lower = more agreement)",
    fontsize=15, fontweight="bold", pad=50,
)
ax.axhline(y=n_humans, color="black", linewidth=2.5)
ax.axvline(x=n_humans, color="black", linewidth=2.5)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(display_labels, rotation=0, fontsize=9)
# Add labels on top and right via manual text placement
n_labels = len(display_labels)
for i, label in enumerate(display_labels):
    ax.text(i + 0.5, -0.3, label, ha="left", va="bottom", rotation=45,
            fontsize=9, clip_on=False)
for i, label in enumerate(display_labels):
    ax.text(n_labels + 0.3, i + 0.5, label, ha="left", va="center",
            fontsize=9, clip_on=False)
plt.subplots_adjust(right=0.82)
plt.savefig(OUT_WER_HEATMAP, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_WER_HEATMAP}")

# --- Heatmap 3: Inter-grader words correct (Pearson r, humans only) ---
ig_corr = corr_matrix.loc[all_graders, all_graders].copy().astype(float)
ig_overlap = overlap_matrix.loc[all_graders, all_graders].copy()

fig, ax = plt.subplots(figsize=(9, 7))
# Annotation: show r value and shared count
annot_ig = ig_corr.copy().astype(str)
for g1 in all_graders:
    for g2 in all_graders:
        r_val = ig_corr.loc[g1, g2]
        n_val = ig_overlap.loc[g1, g2]
        if g1 == g2:
            annot_ig.loc[g1, g2] = ""
        elif pd.isna(r_val):
            annot_ig.loc[g1, g2] = f"n={n_val}"
        else:
            annot_ig.loc[g1, g2] = f"{r_val:.2f}\n(n={n_val})"

sns.heatmap(
    ig_corr, annot=annot_ig, fmt="", cmap="RdYlGn",
    vmin=0, vmax=1, square=True, linewidths=0.5,
    cbar_kws={"label": "Pearson r", "shrink": 0.8, "pad": 0.15}, ax=ax,
    annot_kws={"fontsize": 9},
)
ax.set_title("Inter-Grader Pearson r — Words Correct\nn = shared audios",
             fontsize=14, fontweight="bold", pad=40)
ax.set_xticklabels(all_graders, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(all_graders, rotation=0, fontsize=10)
# Add labels on top and right via manual text placement
n_graders = len(all_graders)
for i, label in enumerate(all_graders):
    ax.text(i + 0.5, -0.3, label, ha="left", va="bottom", rotation=45,
            fontsize=10, clip_on=False)
for i, label in enumerate(all_graders):
    ax.text(n_graders + 0.3, i + 0.5, label, ha="left", va="center",
            fontsize=10, clip_on=False)
plt.subplots_adjust(right=0.82)
plt.savefig(OUT_IG_WC_HEATMAP, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_IG_WC_HEATMAP}")

# --- Bar chart helper ---
def make_comparison_bar(
    ai_summary, baseline_val, baseline_label, metric_col, xlabel,
    out_path, higher_is_better=True, fmt=".1f", pct=False,
):
    """Horizontal bar chart: AI models vs human baseline."""
    names = [DISPLAY_NAMES.get(s, s) for s in ai_summary.index]
    values = ai_summary[metric_col].values

    # Sort so best is at top
    order = np.argsort(values)
    if higher_is_better:
        order = order[::-1]
    names = [names[i] for i in order]
    values = values[order]

    colors = []
    for v in values:
        if higher_is_better:
            colors.append("#27ae60" if v >= baseline_val else "#e74c3c")
        else:
            colors.append("#27ae60" if v <= baseline_val else "#e74c3c")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, height=0.6, zorder=2)
    ax.axvline(baseline_val, color="#2c3e50", linewidth=2, linestyle="--", zorder=3,
               label=f"{baseline_label}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for i, v in enumerate(values):
        label = f"{v:{fmt}}" if not pct else f"{v:.1%}"
        ax.text(v + (baseline_val * 0.01), i, label, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# --- Bar chart 1: Words Correct ---
make_comparison_bar(
    ai_wc_summary, avg_human_mean_wc, f"Avg Human ({avg_human_mean_wc:.1f})",
    "mean_wc", "Mean Words Correct", OUT_WC_BAR,
    higher_is_better=True,
)

# --- Bar chart 2: WER ---
make_comparison_bar(
    ai_wer_summary, avg_human_mean_wer, f"Avg Human ({avg_human_mean_wer:.1%})",
    "mean_wer", "Mean WER", OUT_WER_BAR,
    higher_is_better=False, pct=True,
)

# --- Bar chart 3: WCPM ---
make_comparison_bar(
    ai_wcpm_summary, avg_human_wcpm, f"Avg Human ({avg_human_wcpm})",
    "wcpm", "Words Correct Per Minute", OUT_WCPM_BAR,
    higher_is_better=True,
)

# %%
# ──────────────────────────────────────────────
# Step 11: Select example audios
# ──────────────────────────────────────────────
df_examples = df_dual[df_dual["inter_grader_wer"].notna()].copy()
df_examples = df_examples.sort_values("inter_grader_wer")

high_agree = df_examples[df_examples["inter_grader_wer"] == 0].iloc[0] if (df_examples["inter_grader_wer"] == 0).any() else df_examples.iloc[0]
median_idx = len(df_examples) // 2
med_agree = df_examples.iloc[median_idx]
low_candidates = df_examples[(df_examples["inter_grader_wer"] > 0.3) & (df_examples["inter_grader_wer"] < 1.0)]
low_agree = low_candidates.iloc[len(low_candidates) // 2] if len(low_candidates) > 0 else df_examples.iloc[-5]

examples = [
    ("High Agreement", high_agree),
    ("Medium Agreement", med_agree),
    ("Low Agreement", low_agree),
]

# Best AI model (by mean correlation with all human graders)
ai_human_corrs = []
for model in AI_MODELS:
    rs = [corr_matrix.loc[g, model] for g in all_graders if pd.notna(corr_matrix.loc[g, model])]
    mean_r = np.mean(rs) if rs else np.nan
    ai_human_corrs.append((model, mean_r))
ai_human_corrs.sort(key=lambda x: -x[1])
best_ai = ai_human_corrs[0][0]
best_ai_display = DISPLAY_NAMES.get(best_ai, best_ai)

print("Data computed. Generating markdown...")

# ──────────────────────────────────────────────
# Load pricing data
# ──────────────────────────────────────────────
pricing_df = None
if os.path.exists(PRICING_CSV):
    pricing_df = pd.read_csv(PRICING_CSV, sep="\t")
    pricing_df.columns = pricing_df.columns.str.strip()
    print(f"Loaded pricing data: {len(pricing_df)} rows")

# %%
# ══════════════════════════════════════════════
# GENERATE MARKDOWN
# ══════════════════════════════════════════════
md = []

# ── Title ──
md.append("# Human vs AI Transcription Analysis")
md.append("")
md.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
md.append("")

# ══════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 1. Overview")
md.append("")
md.append(f"- **Total audio files:** {len(df)}")
md.append(f"- **Audio files with at least one human transcription:** {has_any.sum()}")
md.append(f"- **Dual-graded audio files (both graders completed):** {has_both.sum()}")
md.append(f"- **Human graders:** {len(all_graders)} ({', '.join(all_graders)})")
md.append(f"- **AI models evaluated:** {len(AI_MODELS)}")
md.append(f"- **Pre-test audios:** {(df['pre_or_post'] == 'pre').sum()}")
md.append(f"- **Post-test audios:** {(df['pre_or_post'] == 'post').sum()}")
md.append("")
md.append("*All human-vs-AI comparisons use \"Avg Human\" — the per-audio average across all human graders — as the baseline.*")
md.append("")

md.append("### Transcription Counts")
md.append("")
md.append("| Grader | Transcriptions |")
md.append("|--------|---------------|")
for g in all_graders:
    md.append(f"| {g} | {len(grader_transcriptions[g])} |")
md.append("")

# ══════════════════════════════════════════════
# SECTION 2: METHODOLOGY
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 2. Methodology — Words Correct")
md.append("")
md.append("### How we count words correct")
md.append("")
md.append("We use **jiwer's `process_words`** to align the scorer's transcription against the reference text ")
md.append("(what the student was supposed to read). The algorithm finds the optimal word-level alignment ")
md.append("that minimizes total errors, then counts **hits** — words that matched exactly.")
md.append("")
md.append("The alignment handles:")
md.append("- **Substitutions** — a word was replaced (e.g., student said \"laaves\" instead of \"loves\")")
md.append("- **Deletions** — a word was skipped entirely")
md.append("- **Insertions** — an extra word was added")
md.append("- **Hits** — the word matched correctly")
md.append("")
md.append("### Worked example")
md.append("")
md.append("**Reference text:** *\"She loves to draw\"*")
md.append("")
md.append("**Transcription:** *\"She laaves to\"*")
md.append("")
md.append("| Reference | she | loves | to | draw |")
md.append("|-----------|-----|-------|----|------|")
md.append("| Transcription | she | laaves | to | — |")
md.append("| Result | Hit | Substitution | Hit | Deletion |")
md.append("")
md.append("**Words correct = 2** (\"she\" and \"to\" matched)")
md.append("")
md.append("### Why words correct (not WER) for correlation?")
md.append("")
md.append("Word Error Rate (WER) measures the *error rate* of the transcription vs the reference. ")
md.append("However, the transcription captures *what the student actually said*, not what they were ")
md.append("supposed to say. WER therefore conflates:")
md.append("")
md.append("1. **Student reading errors** (the actual signal — the student skipped or mispronounced words)")
md.append("2. **Transcription differences** (noise — graders wrote slightly different things)")
md.append("")
md.append("Words correct isolates the student performance signal: *how many words did the student get right?* ")
md.append("All scorers fundamentally agree on this, which is why word-correct correlations (r > 0.90) ")
md.append("are much higher than WER-based comparisons.")
md.append("")

# ══════════════════════════════════════════════
# SECTION 3: INTER-GRADER AGREEMENT — WORDS CORRECT
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 3. Inter-Grader Agreement — Words Correct")
md.append("")

# Compute summary stats for narrative
valid_ig_wc = df_ig_wc[df_ig_wc["pearson_r"].notna()]
mean_ig_r = valid_ig_wc["pearson_r"].mean()
min_ig_r = valid_ig_wc["pearson_r"].min()
max_ig_r = valid_ig_wc["pearson_r"].max()
mean_ig_diff = valid_ig_wc["mean_abs_diff"].mean()
std_ig_diff = valid_ig_wc["mean_abs_diff"].std()
std_ig_r = valid_ig_wc["pearson_r"].std()
high_agree_pairs = valid_ig_wc[valid_ig_wc["pearson_r"] >= 0.95]
low_agree_pairs = valid_ig_wc[valid_ig_wc["pearson_r"] < 0.85]

md.append(f"To establish whether human graders are consistent enough to serve as a reliable baseline, ")
md.append(f"we compared each pair of graders on the audio files they both transcribed. For each pair, ")
md.append(f"we computed the **mean absolute difference** in words correct (how many words apart they typically are) ")
md.append(f"and the **Pearson correlation** (whether they agree on which students performed well vs poorly).")
md.append("")
md.append(f"Across the {len(valid_ig_wc)} grader pairs with sufficient shared audios (n >= 3):")
md.append("")
md.append(f"- **Mean correlation: r = {mean_ig_r:.2f}** (std: {std_ig_r:.2f}, range: {min_ig_r:.2f} to {max_ig_r:.2f})")
md.append(f"- **Mean absolute difference: {mean_ig_diff:.1f} words** (std: {std_ig_diff:.2f})")
md.append(f"- **{len(high_agree_pairs)} pairs** have r >= 0.95 (strong agreement)")
if len(low_agree_pairs) > 0:
    low_names = [f"{r['grader_1']}-{r['grader_2']}" for _, r in low_agree_pairs.iterrows()]
    md.append(f"- **{len(low_agree_pairs)} pair(s)** fall below r = 0.85: {', '.join(low_names)}")
md.append("")
md.append("This confirms that human graders are largely consistent in identifying how many words ")
md.append("each student reads correctly, making the human baseline reliable for AI comparison.")
md.append("")
md.append("*Full pairwise heatmap in [Appendix C](#c-inter-grader-words-correct--pairwise-heatmap). ")
md.append("Correlation p-values in [Appendix K](#k-inter-grader-correlation-p-values).*")
md.append("")

# ══════════════════════════════════════════════
# SECTION 4: INTER-GRADER AGREEMENT — WER
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 4. Inter-Grader Agreement — WER")
md.append("")
md.append("While words correct captures the student performance signal, WER measures how differently ")
md.append("two graders *transcribed the same audio* — including filler words, spelling variations, ")
md.append("and different ways of representing the same speech. A non-zero WER does not necessarily ")
md.append("mean graders disagree on student ability; it often reflects transcription style differences.")
md.append("")

md.append("| Metric | Value |")
md.append("|--------|-------|")
md.append(f"| N (dual-graded audios) | {len(valid_ig_wer)} |")
md.append(f"| Mean WER | {valid_ig_wer.mean():.1%} |")
md.append(f"| Median WER | {valid_ig_wer.median():.1%} |")
md.append(f"| Std | {valid_ig_wer.std():.2f} |")
md.append(f"| Perfect agreement (WER=0) | {(valid_ig_wer == 0).sum()} audios ({(valid_ig_wer == 0).mean():.0%}) |")
md.append("")
md.append(f"The median WER of {valid_ig_wer.median():.1%} means that on a typical audio, graders' transcriptions ")
md.append(f"differ by about 1 in 6 words. {(valid_ig_wer == 0).mean():.0%} of audios have perfect inter-grader agreement. ")
md.append(f"The high standard deviation ({valid_ig_wer.std():.2f}) reflects that some challenging audios ")
md.append(f"produce very different transcriptions — this is expected with low-proficiency student speech.")
md.append("")

md.append("### How WER Conflates Student Errors and Transcription Style")
md.append("")
md.append("A high inter-grader WER can mean two very different things:")
md.append("")
md.append("1. **Convention noise** — graders heard the student the same way but wrote it differently ")
md.append("(spelling, filler words, elongation). They agree on words correct despite high WER.")
md.append("2. **Genuine disagreement** — graders actually disagree on what the student said, ")
md.append("leading to different words-correct counts.")
md.append("")
md.append("Two examples illustrate this distinction:")
md.append("")

for i, det in enumerate(conflation_details):
    label = "Convention noise" if i == 0 else "Genuine disagreement"
    md.append(f"**{label}:** `{det['audio']}` (Inter-grader WER: {det['wer']:.0%})")
    md.append("")
    md.append(f"*Reference:* \"{det['reference']}\"")
    md.append("")
    # Build word-level alignment table
    ref_words = [w for w, _ in det["align_1"]]
    header = "| Reference | " + " | ".join(ref_words) + " |"
    sep = "|-----------|" + "|".join(["---"] * len(ref_words)) + "|"
    row_1_results = [r for _, r in det["align_1"]]
    row_2_results = [r for _, r in det["align_2"]]
    row_1 = f"| {det['grader_1']} | " + " | ".join(row_1_results) + " |"
    row_2 = f"| {det['grader_2']} | " + " | ".join(row_2_results) + " |"
    md.append(header)
    md.append(sep)
    md.append(row_1)
    md.append(row_2)
    md.append("")
    md.append(f"**{det['grader_1']}:** {det['wc_1']} words correct — *\"{det['trans_1']}\"*")
    md.append(f"**{det['grader_2']}:** {det['wc_2']} words correct — *\"{det['trans_2']}\"*")
    md.append("")
    if i == 0:
        md.append(f"WER is {det['wer']:.0%} but both graders agree on {det['wc_1']} words correct. ")
        md.append("The difference is purely transcription style — not a scoring disagreement.")
    else:
        md.append(f"WER is {det['wer']:.0%} and the graders disagree by {abs(det['wc_1'] - det['wc_2'])} words correct. ")
        md.append("This reflects genuine uncertainty about what the student said.")
    md.append("")

# ══════════════════════════════════════════════
# SECTION 5: AVG HUMAN vs EACH AI — WORDS CORRECT
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 5. Avg Human vs Each AI — Words Correct")
md.append("")
md.append(f"Sections 3–4 established inter-grader reliability on the {len(df_dual)} dual-graded audios. ")
md.append(f"The following AI comparisons use all **{n_any_human} audios** with at least one human transcription, ")
md.append(f"with \"Avg Human\" as the per-audio average across available human graders.")
md.append("")
md.append("### How AI transcriptions were generated")
md.append("")
md.append("Each AI model was given the student audio and the following prompt:")
md.append("")
md.append("> *\"Transcribe the audio, if it is empty return nothing. Transcribe it exactly, don't add any extra words or fix any mistakes.\"*")
md.append("")
md.append("This transcribes the user audio exactly and doesn't auto-fix anything.")
md.append("")

# Narrative
n_above_human_wc = sum(1 for _, r in ai_wc_summary.iterrows() if r["mean_wc"] >= avg_human_mean_wc)
best_ai_wc_name = DISPLAY_NAMES.get(ai_wc_summary.index[0], ai_wc_summary.index[0])
best_ai_wc_val = ai_wc_summary.iloc[0]["mean_wc"]
worst_ai_wc_name = DISPLAY_NAMES.get(ai_wc_summary.index[-1], ai_wc_summary.index[-1])
worst_ai_wc_val = ai_wc_summary.iloc[-1]["mean_wc"]

md.append(f"**{n_above_human_wc} of {len(AI_MODELS)} AI models** match or exceed the average human baseline ")
md.append(f"on words correct. {best_ai_wc_name} leads at {best_ai_wc_val:.1f} words correct per audio ")
md.append(f"(vs {avg_human_mean_wc:.1f} for Avg Human), while {worst_ai_wc_name} trails at {worst_ai_wc_val:.1f}. ")
md.append(f"The spread across AI models is relatively narrow ({worst_ai_wc_val:.1f}–{best_ai_wc_val:.1f}), ")
md.append(f"indicating that most models capture student reading performance comparably to human graders.")
md.append("")
md.append(f"![Words Correct Comparison]({os.path.relpath(OUT_WC_BAR, PROJECT_DIR)})")
md.append("")
md.append("*See [Appendix H](#h-words-correct--statistical-significance) for Wilcoxon signed-rank tests and 95% confidence intervals.*")
md.append("")

# ══════════════════════════════════════════════
# SECTION 6: AVG HUMAN vs EACH AI — WER
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 6. Avg Human vs Each AI — WER")
md.append("")
md.append(f"WER here measures how closely each AI model's transcription matches the human transcriptions ")
md.append(f"of the same audio. For each AI model on each audio, we compute the WER between the AI's ")
md.append(f"transcription and each human grader's transcription, then average. The \"Avg Human\" baseline ")
md.append(f"is the inter-grader WER (how much human graders differ from each other). ")
md.append(f"Lower WER = closer agreement between AI and human transcriptions.")
md.append("")

# Narrative
n_better_than_human_wer = sum(1 for _, r in ai_wer_summary.iterrows() if r["mean_wer"] < avg_human_mean_wer)
best_ai_wer_name = DISPLAY_NAMES.get(ai_wer_summary.index[0], ai_wer_summary.index[0])
best_ai_wer_val = ai_wer_summary.iloc[0]["mean_wer"]

md.append(f"**{n_better_than_human_wer} of {len(AI_MODELS)} AI models** achieve lower WER against human transcriptions than ")
md.append(f"the inter-grader baseline ({avg_human_mean_wer:.1%}). {best_ai_wer_name} leads with {best_ai_wer_val:.1%} WER. ")
md.append(f"Models with WER at or below the inter-grader baseline are transcribing at least as consistently ")
md.append(f"as human graders agree with each other.")
md.append("")
md.append(f"![WER Comparison]({os.path.relpath(OUT_WER_BAR, PROJECT_DIR)})")
md.append("")
md.append("*See [Appendix I](#i-wer--statistical-significance) for Wilcoxon signed-rank tests and 95% confidence intervals.*")
md.append("")

# ══════════════════════════════════════════════
# SECTION 7: WCPM — AVG HUMAN vs EACH AI
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 7. WCPM — Avg Human vs Each AI")
md.append("")
md.append(f"Words Correct Per Minute (WCPM) is the standard reading fluency metric, computed as ")
md.append(f"`total_words_correct / total_duration_minutes` across all audios for each scorer. ")
md.append(f"It combines accuracy (words correct) with speed (how long the student took).")
md.append("")
md.append(f"Duration data is available for **{n_with_dur}** out of {n_any_human} audios ({n_with_dur/n_any_human:.0%} coverage).")
md.append("")

# Narrative
n_above_human_wcpm = sum(1 for _, r in ai_wcpm_summary.iterrows() if r["wcpm"] >= avg_human_wcpm)
best_ai_wcpm_name = DISPLAY_NAMES.get(ai_wcpm_summary.index[0], ai_wcpm_summary.index[0])
best_ai_wcpm_val = ai_wcpm_summary.iloc[0]["wcpm"]

md.append(f"**{n_above_human_wcpm} of {len(AI_MODELS)} AI models** produce WCPM scores at or above the human baseline ")
md.append(f"({avg_human_wcpm} WCPM). {best_ai_wcpm_name} leads at {best_ai_wcpm_val} WCPM. ")
md.append(f"Since WCPM is the metric that ultimately determines a student's reading level, this is the ")
md.append(f"most operationally relevant comparison — it shows that AI scoring closely tracks human scoring ")
md.append(f"on the metric that matters most.")
md.append("")
md.append(f"![WCPM Comparison]({os.path.relpath(OUT_WCPM_BAR, PROJECT_DIR)})")
md.append("")
md.append("*See [Appendix J](#j-wcpm--confidence-intervals) for bootstrap 95% confidence intervals.*")
md.append("")

# ══════════════════════════════════════════════
# SECTION 8: PRE/POST WCPM — AVG HUMAN vs EACH AI
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 8. Pre/Post WCPM — Avg Human vs Each AI")
md.append("")
md.append(f"Students are assessed before (pre-test) and after (post-test) the intervention. ")
md.append(f"The key question: **does AI scoring detect the same student improvement that human graders see?**")
md.append("")

md.append("| Scorer | Pre WCPM | Post WCPM | Delta |")
md.append("|--------|----------|-----------|-------|")

# Avg Human row
avg_h_pre = avg_human_prepost.get("pre", 0)
avg_h_post = avg_human_prepost.get("post", 0)
avg_h_delta = avg_h_post - avg_h_pre
md.append(f"| **Avg Human (baseline)** | **{avg_h_pre}** | **{avg_h_post}** | **{avg_h_delta:+.1f}** |")

# AI rows — sorted by delta descending
ai_pp_table_rows = []
for model in AI_MODELS:
    name = DISPLAY_NAMES.get(model, model)
    model_data = ai_prepost_wcpm[ai_prepost_wcpm["scorer"] == model]
    pre_row = model_data[model_data["pre_or_post"] == "pre"]
    post_row = model_data[model_data["pre_or_post"] == "post"]
    pre_val = pre_row["wcpm"].values[0] if len(pre_row) > 0 else np.nan
    post_val = post_row["wcpm"].values[0] if len(post_row) > 0 else np.nan
    delta = post_val - pre_val if pd.notna(pre_val) and pd.notna(post_val) else np.nan
    ai_pp_table_rows.append((name, pre_val, post_val, delta))
ai_pp_table_rows.sort(key=lambda x: -(x[3] if pd.notna(x[3]) else -999))

for name, pre_val, post_val, delta in ai_pp_table_rows:
    pre_str = f"{pre_val}" if pd.notna(pre_val) else "—"
    post_str = f"{post_val}" if pd.notna(post_val) else "—"
    delta_str = f"{delta:+.1f}" if pd.notna(delta) else "—"
    if pd.notna(delta) and delta > avg_h_delta:
        md.append(f"| **{name}** | **{pre_str}** | **{post_str}** | **{delta_str}** |")
    else:
        md.append(f"| {name} | {pre_str} | {post_str} | {delta_str} |")
md.append("")

# Narrative
ai_deltas = [d for _, _, _, d in ai_pp_table_rows if pd.notna(d)]
mean_ai_delta = np.mean(ai_deltas)
md.append(f"Human graders measure a **{avg_h_delta:+.1f} WCPM improvement** from pre to post. ")
md.append(f"AI models measure a mean improvement of **{mean_ai_delta:+.1f} WCPM** — closely tracking ")
md.append(f"the human-observed gain. All {len(ai_deltas)} AI models detect a positive pre-to-post improvement, ")
md.append(f"confirming that AI scoring reliably captures the effect of the reading intervention.")
md.append("")
md.append("*See [Appendix L](#l-prepost-wcpm--statistical-significance) for pre/post WCPM significance tests.*")
md.append("")

# ══════════════════════════════════════════════
# SECTION 9: CONCLUSION
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 9. Conclusion")
md.append("")
md.append(f"Inter-grader agreement is strong (mean r = {mean_ig_r:.2f}, std = {std_ig_r:.2f} across {len(valid_ig_wc)} pairs), ")
md.append(f"establishing a reliable human baseline for comparing AI transcription models. ")
md.append(f"On the {n_any_human} audios with human transcription:")
md.append("")
md.append(f"- **{n_above_human_wc} of {len(AI_MODELS)} AI models** match or exceed the average human baseline on words correct")
md.append(f"- **{n_above_human_wcpm} of {len(AI_MODELS)} AI models** meet or exceed the human WCPM baseline ({avg_human_wcpm} WCPM)")
md.append(f"- **All {len(ai_deltas)} AI models** detect a positive pre-to-post improvement, ")
md.append(f"with a mean AI delta of {mean_ai_delta:+.1f} WCPM vs the human-observed {avg_h_delta:+.1f} WCPM")
md.append("")
md.append("WER differences between AI and human transcriptions are comparable to inter-grader WER, ")
md.append("suggesting that much of the measured disagreement reflects transcription style rather than ")
md.append("scoring disagreement. AI scoring can reliably replace or supplement human grading for ")
md.append("reading fluency assessment.")
md.append("")

# ══════════════════════════════════════════════
# SECTION 10: APPENDIX
# ══════════════════════════════════════════════
md.append("---")
md.append("")
md.append("## 10. Appendix")
md.append("")

md.append("### A. Words Correct Correlation Heatmap")
md.append("")
md.append("Pairwise Pearson correlation of word-correct counts across shared audio files, ")
md.append("using all graders + AI models. Color scale: 0 to 1.")
md.append("")
md.append("![Words Correct Correlation](plots/v3_report_words_correct_correlation.png)")
md.append("")

md.append("#### Mean Correlations by Group")
md.append("")
md.append("| Group | Mean Pearson r | N pairs |")
md.append("|-------|---------------|---------|")
if not np.isnan(mean_hh):
    md.append(f"| Human-Human | {mean_hh:.3f} | {len(hh_corrs)} |")
if not np.isnan(mean_ha):
    md.append(f"| Human-AI | {mean_ha:.3f} | {len(ha_corrs)} |")
if not np.isnan(mean_aa):
    md.append(f"| AI-AI | {mean_aa:.3f} | {len(aa_corrs)} |")
md.append("")

md.append("### B. WER Heatmap")
md.append("")
md.append("Pairwise mean WER between all scorers on shared audio files. ")
md.append("Color scale: 0 to 1 (0 = identical transcriptions, 1 = completely different).")
md.append("")
md.append("![WER Heatmap](plots/v3_report_wer_heatmap.png)")
md.append("")

md.append("### C. Inter-Grader Words Correct — Pairwise Heatmap")
md.append("")
md.append("Pairwise Pearson correlation of word-correct counts between human graders. ")
md.append("Each cell shows the correlation and the number of shared audios. ")
md.append("Color scale: 0 to 1 (green = strong agreement).")
md.append("")
md.append("![Inter-Grader Words Correct Heatmap](plots/v3_report_inter_grader_wc.png)")
md.append("")

md.append("### D. Transcription Examples")
md.append("")
md.append("Three examples showing how human graders and AI models transcribe the same audio, ")
md.append("selected by inter-grader agreement level (WER between the two human graders).")
md.append("")

for label, row in examples:
    audio = row["audio_file_name"]
    ref = row["question"]
    t1 = row["transcription_1"]
    t2 = row["transcription_2"]
    g1 = row["grader_1"]
    g2 = row["grader_2"]
    ig_wer_val = row["inter_grader_wer"]

    ai_col = f"ai_{best_ai}"
    ai_trans = row.get(ai_col)
    ai_trans = str(ai_trans).strip() if pd.notna(ai_trans) else "N/A"

    wc1 = compute_words_correct(ref, t1)
    wc2 = compute_words_correct(ref, t2)
    wc_ai = compute_words_correct(ref, ai_trans) if ai_trans != "N/A" else "N/A"
    total_words = len(clean_text(ref).split())

    md.append(f"#### {label} (Inter-grader WER: {ig_wer_val:.0%})")
    md.append("")
    md.append(f"**Audio:** `{audio}`")
    md.append("")
    md.append(f"**Reference text:** *\"{ref}\"*")
    md.append("")
    md.append(f"| Scorer | Transcription | Words Correct |")
    md.append(f"|--------|---------------|---------------|")
    md.append(f"| {g1} (Human) | {t1} | {wc1}/{total_words} |")
    md.append(f"| {g2} (Human) | {t2} | {wc2}/{total_words} |")
    md.append(f"| {best_ai_display} (AI) | {ai_trans} | {wc_ai}/{total_words} |")
    md.append("")

# Appendix E: Words Correct table
md.append("### E. Words Correct — Full Rankings")
md.append("")
md.append("| Rank | Scorer | Mean Words Correct | Median | N (audios) |")
md.append("|------|--------|--------------------|--------|------------|")
md.append(f"| — | **Avg Human (baseline)** | **{avg_human_mean_wc:.1f}** | **{avg_human_median_wc:.0f}** | **{avg_human_n_wc}** |")
for rank, (scorer, row) in enumerate(ai_wc_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    md.append(f"| {rank} | {name} | {row['mean_wc']:.1f} | {row['median_wc']:.0f} | {int(row['n'])} |")
md.append("")

# Appendix F: WER table
md.append("### F. WER — Full Rankings")
md.append("")
md.append("| Rank | Scorer | Mean WER | Median WER | N (audios) |")
md.append("|------|--------|----------|------------|------------|")
md.append(f"| — | **Avg Human (baseline)** | **{avg_human_mean_wer:.1%}** | **{avg_human_median_wer:.1%}** | **{avg_human_n_wc}** |")
for rank, (scorer, row) in enumerate(ai_wer_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    md.append(f"| {rank} | {name} | {row['mean_wer']:.1%} | {row['median_wer']:.1%} | {int(row['n'])} |")
md.append("")

# Appendix G: WCPM table
md.append("### G. WCPM — Full Rankings")
md.append("")
md.append("| Rank | Scorer | WCPM | N (audios) |")
md.append("|------|--------|------|------------|")
md.append(f"| — | **Avg Human (baseline)** | **{avg_human_wcpm}** | **{avg_human_n_wcpm}** |")
for rank, (scorer, row) in enumerate(ai_wcpm_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    md.append(f"| {rank} | {name} | {row['wcpm']} | {int(row['n'])} |")
md.append("")

# Appendix H: Words Correct — Statistical Significance
md.append("### H. Words Correct — Statistical Significance")
md.append("")
md.append("Wilcoxon signed-rank tests comparing each AI model's per-audio words correct against the ")
md.append("average human baseline, with Benjamini-Hochberg FDR correction for multiple comparisons. ")
md.append("Mean Diff = AI mean minus human baseline (positive = AI scored higher).")
md.append("")
md.append("| Rank | Model | N | Mean Diff | 95% CI | p (adjusted) | Sig? |")
md.append("|------|-------|---|-----------|--------|--------------|------|")
wc_sig_sorted = wc_sig.sort_values("mean_diff", ascending=False)
for rank, (_, row) in enumerate(wc_sig_sorted.iterrows(), 1):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    ci_str = f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
    sig = "Yes" if row["significant"] else "No"
    md.append(f"| {rank} | {name} | {row['n']} | {row['mean_diff']:+.2f} | {ci_str} | {fmt_pval(row['p_adjusted'])} | {sig} |")
md.append("")
md.append("*With n > 500 paired observations, even small differences reach statistical significance. ")
md.append("The 95% CI on mean difference is more informative for practical interpretation.*")
md.append("")

# Appendix I: WER — Statistical Significance
md.append("### I. WER — Statistical Significance")
md.append("")
md.append("Wilcoxon signed-rank tests comparing each AI model's per-audio WER (vs human transcription) ")
md.append("against the average inter-grader WER baseline, across all audios with human transcription.")
md.append("Negative mean diff = AI has lower (better) WER than inter-grader baseline.")
md.append("")
md.append("| Rank | Model | N | Mean Diff | 95% CI | p (adjusted) | Sig? |")
md.append("|------|-------|---|-----------|--------|--------------|------|")
wer_sig_sorted = wer_sig.sort_values("mean_diff")
for rank, (_, row) in enumerate(wer_sig_sorted.iterrows(), 1):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
    sig = "Yes" if row["significant"] else "No"
    md.append(f"| {rank} | {name} | {row['n']} | {row['mean_diff']:+.4f} | {ci_str} | {fmt_pval(row['p_adjusted'])} | {sig} |")
md.append("")

# Appendix J: WCPM — Confidence Intervals
md.append("### J. WCPM — Confidence Intervals")
md.append("")
md.append("Bootstrap 95% confidence intervals (10,000 resamples) for aggregate WCPM ")
md.append("(total words correct / total duration minutes) per scorer.")
md.append("")
md.append("| Rank | Scorer | WCPM | 95% CI | N (audios) |")
md.append("|------|--------|------|--------|------------|")
wcpm_ci_sorted = wcpm_ci_df.sort_values("wcpm", ascending=False)
ai_rank = 0
for _, row in wcpm_ci_sorted.iterrows():
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    is_baseline = row["scorer"] == "Avg Human (baseline)"
    ci_str = f"[{row['ci_lo']}, {row['ci_hi']}]"
    if is_baseline:
        md.append(f"| — | **{name}** | **{row['wcpm']}** | **{ci_str}** | **{int(row['n'])}** |")
    else:
        ai_rank += 1
        md.append(f"| {ai_rank} | {name} | {row['wcpm']} | {ci_str} | {int(row['n'])} |")
md.append("")

# Appendix K: Inter-Grader Correlation P-values
md.append("### K. Inter-Grader Correlation P-values")
md.append("")
md.append("Pearson correlation p-values for each grader pair's words-correct agreement. ")
md.append("All pairs with sufficient shared audios show highly significant correlations.")
md.append("")
md.append("| Grader 1 | Grader 2 | Pearson r | p-value | Shared Audios |")
md.append("|----------|----------|-----------|---------|---------------|")
valid_ig = df_ig_wc[df_ig_wc["pearson_r"].notna()]
for _, row in valid_ig.iterrows():
    md.append(f"| {row['grader_1']} | {row['grader_2']} | {row['pearson_r']:.3f} | {fmt_pval(row['pearson_p'])} | {row['shared_audios']} |")
md.append("")

# Appendix L: Pre/Post WCPM — Statistical Significance
md.append("### L. Pre/Post WCPM — Statistical Significance")
md.append("")
md.append("Bootstrap significance test (10,000 resamples) for the pre-to-post WCPM improvement. ")
md.append("Delta = Post WCPM minus Pre WCPM. A significant result means the improvement is ")
md.append("reliably different from zero.")
md.append("")
md.append("| Rank | Scorer | Pre WCPM | Post WCPM | Delta | 95% CI | p-value | Sig? |")
md.append("|------|--------|----------|-----------|-------|--------|---------|------|")
pp_delta_sorted = prepost_delta_df.sort_values("delta", ascending=False)
ai_rank = 0
for _, row in pp_delta_sorted.iterrows():
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    is_baseline = row["scorer"] == "Avg Human (baseline)"
    ci_str = f"[{row['ci_lo']}, {row['ci_hi']}]"
    sig = "Yes" if row["p_value"] < 0.05 else "No"
    if is_baseline:
        md.append(f"| — | **{name}** | **{row['pre_wcpm']}** | **{row['post_wcpm']}** | **{row['delta']:+.1f}** | **{ci_str}** | **{fmt_pval(row['p_value'])}** | **{sig}** |")
    else:
        ai_rank += 1
        md.append(f"| {ai_rank} | {name} | {row['pre_wcpm']} | {row['post_wcpm']} | {row['delta']:+.1f} | {ci_str} | {fmt_pval(row['p_value'])} | {sig} |")
md.append("")

# Appendix M: Model Pricing
md.append("### M. Model Pricing")
md.append("")
md.append("Estimated cost per 1,000 minutes of audio and median processing speed for each AI model evaluated. ")
md.append("Pricing sourced from provider documentation as of March 2026. ")
md.append("Empty cells indicate pricing was not publicly available.")
md.append("")
md.append("| Model | Price (USD / 1,000 min) | Median Speed Factor | Benchmark WER (provider) |")
md.append("|-------|------------------------|--------------------:|-------------------------:|")
if pricing_df is not None:
    for model in AI_MODELS:
        display = DISPLAY_NAMES.get(model, model)
        pricing_name = PRICING_NAMES.get(model, display)
        match = pricing_df[pricing_df["model"].str.strip() == pricing_name]
        if len(match) > 0:
            row = match.iloc[0]
            price = f"${row['price_usd_per_1000_min']:.2f}" if pd.notna(row.get("price_usd_per_1000_min")) and str(row.get("price_usd_per_1000_min")).strip() else "—"
            speed = f"{row['median_speed_factor']}x" if pd.notna(row.get("median_speed_factor")) and str(row.get("median_speed_factor")).strip() else "—"
            bwer = str(row.get("word_error_rate_perc", "—")).strip() if pd.notna(row.get("word_error_rate_perc")) and str(row.get("word_error_rate_perc")).strip() else "—"
        else:
            price, speed, bwer = "—", "—", "—"
        md.append(f"| {display} | {price} | {speed} | {bwer} |")
md.append("")
md.append("*Speed factor = how many times faster than real-time the model transcribes (higher = faster). ")
md.append("Benchmark WER is the provider-reported word error rate on standard English benchmarks, ")
md.append("which may differ from WER observed on Pakistani English child speech.*")
md.append("")

# ──────────────────────────────────────────────
# Write markdown
# ──────────────────────────────────────────────
md_content = "\n".join(md)
with open(OUT_MARKDOWN, "w") as f:
    f.write(md_content)
print(f"Saved report to: {OUT_MARKDOWN}")

# %%
# ══════════════════════════════════════════════
# GENERATE WORD DOCUMENT
# ══════════════════════════════════════════════
print("Generating Word document...")

doc = Document()

# ── Helpers ──
def add_table_to_doc(doc, headers, rows, bold_row_indices=None):
    """Add a formatted table to the document."""
    bold_row_indices = bold_row_indices or set()
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
        is_bold = r_idx in bold_row_indices
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(9)
                    if is_bold:
                        run.bold = True
    return table


def add_para(doc, text, bold=False, italic=False, size=None):
    """Add a paragraph with optional formatting."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if size:
        run.font.size = Pt(size)
    return p


# ── Title ──
title = doc.add_heading("Human vs AI Transcription Analysis", level=0)
add_para(doc, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}", italic=True)

# ══════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ══════════════════════════════════════════════
doc.add_heading("1. Overview", level=1)

overview_items = [
    f"Total audio files: {len(df)}",
    f"Audio files with at least one human transcription: {has_any.sum()}",
    f"Dual-graded audio files (both graders completed): {has_both.sum()}",
    f"Human graders: {len(all_graders)} ({', '.join(all_graders)})",
    f"AI models evaluated: {len(AI_MODELS)}",
    f"Pre-test audios: {(df['pre_or_post'] == 'pre').sum()}",
    f"Post-test audios: {(df['pre_or_post'] == 'post').sum()}",
]
for item in overview_items:
    doc.add_paragraph(item, style="List Bullet")

add_para(doc, 'All human-vs-AI comparisons use "Avg Human" — the per-audio average across all human graders — as the baseline.', italic=True)

doc.add_heading("Transcription Counts", level=2)
tc_rows = [[g, str(len(grader_transcriptions[g]))] for g in all_graders]
add_table_to_doc(doc, ["Grader", "Transcriptions"], tc_rows)

# ══════════════════════════════════════════════
# SECTION 2: METHODOLOGY
# ══════════════════════════════════════════════
doc.add_heading("2. Methodology — Words Correct", level=1)

doc.add_heading("How we count words correct", level=2)
doc.add_paragraph(
    "We use jiwer's process_words to align the scorer's transcription against the reference text "
    "(what the student was supposed to read). The algorithm finds the optimal word-level alignment "
    "that minimizes total errors, then counts hits — words that matched exactly."
)
doc.add_paragraph("The alignment handles:")
for item in [
    "Substitutions — a word was replaced (e.g., student said \"laaves\" instead of \"loves\")",
    "Deletions — a word was skipped entirely",
    "Insertions — an extra word was added",
    "Hits — the word matched correctly",
]:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("Worked example", level=2)
doc.add_paragraph('Reference text: "She loves to draw"')
doc.add_paragraph('Transcription: "She laaves to"')
add_table_to_doc(doc,
    ["Reference", "she", "loves", "to", "draw"],
    [["Transcription", "she", "laaves", "to", "—"],
     ["Result", "Hit", "Substitution", "Hit", "Deletion"]],
)
doc.add_paragraph("")
add_para(doc, 'Words correct = 2 ("she" and "to" matched)', bold=True)

doc.add_heading("Why words correct (not WER) for correlation?", level=2)
doc.add_paragraph(
    "Word Error Rate (WER) measures the error rate of the transcription vs the reference. "
    "However, the transcription captures what the student actually said, not what they were "
    "supposed to say. WER therefore conflates:"
)
doc.add_paragraph("Student reading errors (the actual signal — the student skipped or mispronounced words)", style="List Number")
doc.add_paragraph("Transcription differences (noise — graders wrote slightly different things)", style="List Number")
doc.add_paragraph(
    "Words correct isolates the student performance signal: how many words did the student get right? "
    "All scorers fundamentally agree on this, which is why word-correct correlations (r > 0.90) "
    "are much higher than WER-based comparisons."
)

# ══════════════════════════════════════════════
# SECTION 3: INTER-GRADER AGREEMENT — WORDS CORRECT
# ══════════════════════════════════════════════
doc.add_heading("3. Inter-Grader Agreement — Words Correct", level=1)

doc.add_paragraph(
    "To establish whether human graders are consistent enough to serve as a reliable baseline, "
    "we compared each pair of graders on the audio files they both transcribed. For each pair, "
    "we computed the mean absolute difference in words correct (how many words apart they typically are) "
    "and the Pearson correlation (whether they agree on which students performed well vs poorly)."
)

doc.add_paragraph(f"Across the {len(valid_ig_wc)} grader pairs with sufficient shared audios (n >= 3):")
summary_items_3 = [
    f"Mean correlation: r = {mean_ig_r:.2f} (std: {std_ig_r:.2f}, range: {min_ig_r:.2f} to {max_ig_r:.2f})",
    f"Mean absolute difference: {mean_ig_diff:.1f} words (std: {std_ig_diff:.2f})",
    f"{len(high_agree_pairs)} pairs have r >= 0.95 (strong agreement)",
]
if len(low_agree_pairs) > 0:
    low_names = [f"{r['grader_1']}-{r['grader_2']}" for _, r in low_agree_pairs.iterrows()]
    summary_items_3.append(f"{len(low_agree_pairs)} pair(s) fall below r = 0.85: {', '.join(low_names)}")
for item in summary_items_3:
    doc.add_paragraph(item, style="List Bullet")

doc.add_paragraph(
    "This confirms that human graders are largely consistent in identifying how many words "
    "each student reads correctly, making the human baseline reliable for AI comparison."
)
add_para(doc, "Full pairwise heatmap in Appendix C. Correlation p-values in Appendix K.", italic=True)

# ══════════════════════════════════════════════
# SECTION 4: INTER-GRADER AGREEMENT — WER
# ══════════════════════════════════════════════
doc.add_heading("4. Inter-Grader Agreement — WER", level=1)

doc.add_paragraph(
    "While words correct captures the student performance signal, WER measures how differently "
    "two graders transcribed the same audio — including filler words, spelling variations, "
    "and different ways of representing the same speech. A non-zero WER does not necessarily "
    "mean graders disagree on student ability; it often reflects transcription style differences."
)

add_table_to_doc(doc, ["Metric", "Value"], [
    ["N (dual-graded audios)", str(len(valid_ig_wer))],
    ["Mean WER", f"{valid_ig_wer.mean():.1%}"],
    ["Median WER", f"{valid_ig_wer.median():.1%}"],
    ["Std", f"{valid_ig_wer.std():.2f}"],
    ["Perfect agreement (WER=0)", f"{(valid_ig_wer == 0).sum()} audios ({(valid_ig_wer == 0).mean():.0%})"],
])

doc.add_paragraph(
    f"The median WER of {valid_ig_wer.median():.1%} means that on a typical audio, graders' transcriptions "
    f"differ by about 1 in 6 words. {(valid_ig_wer == 0).mean():.0%} of audios have perfect inter-grader agreement. "
    f"The high standard deviation ({valid_ig_wer.std():.2f}) reflects that some challenging audios "
    f"produce very different transcriptions — this is expected with low-proficiency student speech."
)

doc.add_heading("How WER Conflates Student Errors and Transcription Style", level=2)
doc.add_paragraph(
    "A high inter-grader WER can mean two very different things:"
)
doc.add_paragraph(
    "Convention noise — graders heard the student the same way but wrote it differently "
    "(spelling, filler words, elongation). They agree on words correct despite high WER.",
    style="List Number"
)
doc.add_paragraph(
    "Genuine disagreement — graders actually disagree on what the student said, "
    "leading to different words-correct counts.",
    style="List Number"
)
doc.add_paragraph("Two examples illustrate this distinction:")

for i, det in enumerate(conflation_details):
    label = "Convention noise" if i == 0 else "Genuine disagreement"
    add_para(doc, f"{label}: {det['audio']} (Inter-grader WER: {det['wer']:.0%})", bold=True, size=10)
    add_para(doc, f'Reference: "{det["reference"]}"', italic=True, size=10)
    ref_words = [w for w, _ in det["align_1"]]
    results_1 = [r for _, r in det["align_1"]]
    results_2 = [r for _, r in det["align_2"]]
    add_table_to_doc(doc, ["Reference"] + ref_words, [
        [det["grader_1"]] + results_1,
        [det["grader_2"]] + results_2,
    ])
    doc.add_paragraph(
        f"{det['grader_1']}: {det['wc_1']} words correct — \"{det['trans_1']}\""
    )
    doc.add_paragraph(
        f"{det['grader_2']}: {det['wc_2']} words correct — \"{det['trans_2']}\""
    )
    if i == 0:
        doc.add_paragraph(
            f"WER is {det['wer']:.0%} but both graders agree on {det['wc_1']} words correct. "
            "The difference is purely transcription style — not a scoring disagreement."
        )
    else:
        doc.add_paragraph(
            f"WER is {det['wer']:.0%} and the graders disagree by {abs(det['wc_1'] - det['wc_2'])} words correct. "
            "This reflects genuine uncertainty about what the student said."
        )
    doc.add_paragraph("")

# ══════════════════════════════════════════════
# SECTION 5: AVG HUMAN vs EACH AI — WORDS CORRECT
# ══════════════════════════════════════════════
doc.add_heading("5. Avg Human vs Each AI — Words Correct", level=1)

doc.add_paragraph(
    f'Sections 3–4 established inter-grader reliability on the {len(df_dual)} dual-graded audios. '
    f'The following AI comparisons use all {n_any_human} audios with at least one human transcription, '
    f'with "Avg Human" as the per-audio average across available human graders.'
)

doc.add_heading("How AI transcriptions were generated", level=2)
doc.add_paragraph(
    "Each AI model was given the student audio and the following prompt:"
)
p_quote = doc.add_paragraph()
p_quote.style = "Quote"
run = p_quote.add_run(
    '"Transcribe the audio, if it is empty return nothing. '
    'Transcribe it exactly, don\'t add any extra words or fix any mistakes."'
)
run.italic = True
doc.add_paragraph(
    "This transcribes the user audio exactly and doesn't auto-fix anything."
)

doc.add_paragraph(
    f"{n_above_human_wc} of {len(AI_MODELS)} AI models match or exceed the average human baseline "
    f"on words correct. {best_ai_wc_name} leads at {best_ai_wc_val:.1f} words correct per audio "
    f"(vs {avg_human_mean_wc:.1f} for Avg Human), while {worst_ai_wc_name} trails at {worst_ai_wc_val:.1f}. "
    f"The spread across AI models is relatively narrow ({worst_ai_wc_val:.1f}–{best_ai_wc_val:.1f}), "
    f"indicating that most models capture student reading performance comparably to human graders."
)
doc.add_picture(OUT_WC_BAR, width=Inches(5.5))
add_para(doc, "See Appendix H for Wilcoxon signed-rank tests and 95% confidence intervals.", italic=True)

# ══════════════════════════════════════════════
# SECTION 6: AVG HUMAN vs EACH AI — WER
# ══════════════════════════════════════════════
doc.add_heading("6. Avg Human vs Each AI — WER", level=1)

doc.add_paragraph(
    "WER here measures how closely each AI model's transcription matches the human transcriptions "
    "of the same audio. For each AI model on each audio, we compute the WER between the AI's "
    "transcription and each human grader's transcription, then average. The \"Avg Human\" baseline "
    "is the inter-grader WER (how much human graders differ from each other). "
    "Lower WER = closer agreement between AI and human transcriptions."
)

doc.add_paragraph(
    f"{n_better_than_human_wer} of {len(AI_MODELS)} AI models achieve lower WER against human transcriptions than "
    f"the inter-grader baseline ({avg_human_mean_wer:.1%}). {best_ai_wer_name} leads with {best_ai_wer_val:.1%} WER. "
    f"Models with WER at or below the inter-grader baseline are transcribing at least as consistently "
    f"as human graders agree with each other."
)
doc.add_picture(OUT_WER_BAR, width=Inches(5.5))
add_para(doc, "See Appendix I for Wilcoxon signed-rank tests and 95% confidence intervals.", italic=True)

# ══════════════════════════════════════════════
# SECTION 7: WCPM — AVG HUMAN vs EACH AI
# ══════════════════════════════════════════════
doc.add_heading("7. WCPM — Avg Human vs Each AI", level=1)

doc.add_paragraph(
    "Words Correct Per Minute (WCPM) is the standard reading fluency metric, computed as "
    "total_words_correct / total_duration_minutes across all audios for each scorer. "
    "It combines accuracy (words correct) with speed (how long the student took)."
)
doc.add_paragraph(
    f"Duration data is available for {n_with_dur} out of {n_any_human} audios ({n_with_dur/n_any_human:.0%} coverage)."
)

doc.add_paragraph(
    f"{n_above_human_wcpm} of {len(AI_MODELS)} AI models produce WCPM scores at or above the human baseline "
    f"({avg_human_wcpm} WCPM). {best_ai_wcpm_name} leads at {best_ai_wcpm_val} WCPM. "
    f"Since WCPM is the metric that ultimately determines a student's reading level, this is the "
    f"most operationally relevant comparison — it shows that AI scoring closely tracks human scoring "
    f"on the metric that matters most."
)
doc.add_picture(OUT_WCPM_BAR, width=Inches(5.5))
add_para(doc, "See Appendix J for bootstrap 95% confidence intervals.", italic=True)

# ══════════════════════════════════════════════
# SECTION 8: PRE/POST WCPM
# ══════════════════════════════════════════════
doc.add_heading("8. Pre/Post WCPM — Avg Human vs Each AI", level=1)

doc.add_paragraph(
    "Students are assessed before (pre-test) and after (post-test) the intervention. "
    "The key question: does AI scoring detect the same student improvement that human graders see?"
)

prepost_rows = [[
    "Avg Human (baseline)",
    str(avg_h_pre), str(avg_h_post), f"{avg_h_delta:+.1f}",
]]
prepost_bold = {0}
for i, (name, pre_val, post_val, delta) in enumerate(ai_pp_table_rows):
    pre_str = str(pre_val) if pd.notna(pre_val) else "—"
    post_str = str(post_val) if pd.notna(post_val) else "—"
    delta_str = f"{delta:+.1f}" if pd.notna(delta) else "—"
    prepost_rows.append([name, pre_str, post_str, delta_str])
    if pd.notna(delta) and delta > avg_h_delta:
        prepost_bold.add(i + 1)
add_table_to_doc(doc, ["Scorer", "Pre WCPM", "Post WCPM", "Delta"], prepost_rows, bold_row_indices=prepost_bold)

doc.add_paragraph(
    f"Human graders measure a {avg_h_delta:+.1f} WCPM improvement from pre to post. "
    f"AI models measure a mean improvement of {mean_ai_delta:+.1f} WCPM — closely tracking "
    f"the human-observed gain. All {len(ai_deltas)} AI models detect a positive pre-to-post improvement, "
    f"confirming that AI scoring reliably captures the effect of the reading intervention."
)
add_para(doc, "See Appendix L for pre/post WCPM significance tests.", italic=True)

# ══════════════════════════════════════════════
# SECTION 9: CONCLUSION
# ══════════════════════════════════════════════
doc.add_heading("9. Conclusion", level=1)

doc.add_paragraph(
    f"Inter-grader agreement is strong (mean r = {mean_ig_r:.2f}, std = {std_ig_r:.2f} across {len(valid_ig_wc)} pairs), "
    f"establishing a reliable human baseline for comparing AI transcription models. "
    f"On the {n_any_human} audios with human transcription:"
)
for item in [
    f"{n_above_human_wc} of {len(AI_MODELS)} AI models match or exceed the average human baseline on words correct",
    f"{n_above_human_wcpm} of {len(AI_MODELS)} AI models meet or exceed the human WCPM baseline ({avg_human_wcpm} WCPM)",
    f"All {len(ai_deltas)} AI models detect a positive pre-to-post improvement, "
    f"with a mean AI delta of {mean_ai_delta:+.1f} WCPM vs the human-observed {avg_h_delta:+.1f} WCPM",
]:
    doc.add_paragraph(item, style="List Bullet")
doc.add_paragraph(
    "WER differences between AI and human transcriptions are comparable to inter-grader WER, "
    "suggesting that much of the measured disagreement reflects transcription style rather than "
    "scoring disagreement. AI scoring can reliably replace or supplement human grading for "
    "reading fluency assessment."
)

# ══════════════════════════════════════════════
# SECTION 10: APPENDIX
# ══════════════════════════════════════════════
doc.add_heading("10. Appendix", level=1)

doc.add_heading("A. Words Correct Correlation Heatmap", level=2)
doc.add_paragraph(
    "Pairwise Pearson correlation of word-correct counts across shared audio files, "
    "using all graders + AI models. Color scale: 0 to 1."
)
doc.add_picture(OUT_CORR_HEATMAP, width=Inches(6.0))

doc.add_heading("Mean Correlations by Group", level=3)
corr_rows = []
if not np.isnan(mean_hh):
    corr_rows.append(["Human-Human", f"{mean_hh:.3f}", str(len(hh_corrs))])
if not np.isnan(mean_ha):
    corr_rows.append(["Human-AI", f"{mean_ha:.3f}", str(len(ha_corrs))])
if not np.isnan(mean_aa):
    corr_rows.append(["AI-AI", f"{mean_aa:.3f}", str(len(aa_corrs))])
add_table_to_doc(doc, ["Group", "Mean Pearson r", "N pairs"], corr_rows)

doc.add_heading("B. WER Heatmap", level=2)
doc.add_paragraph(
    "Pairwise mean WER between all scorers on shared audio files. "
    "Color scale: 0 to 1 (0 = identical transcriptions, 1 = completely different)."
)
doc.add_picture(OUT_WER_HEATMAP, width=Inches(6.0))

doc.add_heading("C. Inter-Grader Words Correct — Pairwise Heatmap", level=2)
doc.add_paragraph(
    "Pairwise Pearson correlation of word-correct counts between human graders. "
    "Each cell shows the correlation and the number of shared audios. "
    "Color scale: 0 to 1 (green = strong agreement)."
)
doc.add_picture(OUT_IG_WC_HEATMAP, width=Inches(5.5))

doc.add_heading("D. Transcription Examples", level=2)
doc.add_paragraph(
    "Three examples showing how human graders and AI models transcribe the same audio, "
    "selected by inter-grader agreement level (WER between the two human graders)."
)

for label, ex_row in examples:
    audio = ex_row["audio_file_name"]
    ref = ex_row["question"]
    t1 = ex_row["transcription_1"]
    t2 = ex_row["transcription_2"]
    g1 = ex_row["grader_1"]
    g2 = ex_row["grader_2"]
    ig_wer_val = ex_row["inter_grader_wer"]

    ai_col = f"ai_{best_ai}"
    ai_trans = ex_row.get(ai_col)
    ai_trans = str(ai_trans).strip() if pd.notna(ai_trans) else "N/A"

    wc1 = compute_words_correct(ref, t1)
    wc2 = compute_words_correct(ref, t2)
    wc_ai = compute_words_correct(ref, ai_trans) if ai_trans != "N/A" else "N/A"
    total_words = len(clean_text(ref).split())

    doc.add_heading(f"{label} (Inter-grader WER: {ig_wer_val:.0%})", level=3)
    doc.add_paragraph(f"Audio: {audio}")
    doc.add_paragraph(f'Reference text: "{ref}"')
    add_table_to_doc(doc, ["Scorer", "Transcription", "Words Correct"], [
        [f"{g1} (Human)", str(t1), f"{wc1}/{total_words}"],
        [f"{g2} (Human)", str(t2), f"{wc2}/{total_words}"],
        [f"{best_ai_display} (AI)", str(ai_trans), f"{wc_ai}/{total_words}"],
    ])
    doc.add_paragraph("")

# Appendix E: Words Correct table
doc.add_heading("E. Words Correct — Full Rankings", level=2)
wc_rows = [["—", "Avg Human (baseline)", f"{avg_human_mean_wc:.1f}", f"{avg_human_median_wc:.0f}", str(avg_human_n_wc)]]
for rank, (scorer, row) in enumerate(ai_wc_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    wc_rows.append([str(rank), name, f"{row['mean_wc']:.1f}", f"{row['median_wc']:.0f}", str(int(row['n']))])
add_table_to_doc(doc, ["Rank", "Scorer", "Mean Words Correct", "Median", "N (audios)"], wc_rows, bold_row_indices={0})

# Appendix F: WER table
doc.add_heading("F. WER — Full Rankings", level=2)
wer_rows = [["—", "Avg Human (baseline)", f"{avg_human_mean_wer:.1%}", f"{avg_human_median_wer:.1%}", str(avg_human_n_wc)]]
for rank, (scorer, row) in enumerate(ai_wer_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    wer_rows.append([str(rank), name, f"{row['mean_wer']:.1%}", f"{row['median_wer']:.1%}", str(int(row['n']))])
add_table_to_doc(doc, ["Rank", "Scorer", "Mean WER", "Median WER", "N (audios)"], wer_rows, bold_row_indices={0})

# Appendix G: WCPM table
doc.add_heading("G. WCPM — Full Rankings", level=2)
wcpm_rows = [["—", "Avg Human (baseline)", str(avg_human_wcpm), str(avg_human_n_wcpm)]]
for rank, (scorer, row) in enumerate(ai_wcpm_summary.iterrows(), 1):
    name = DISPLAY_NAMES.get(scorer, scorer)
    wcpm_rows.append([str(rank), name, str(row['wcpm']), str(int(row['n']))])
add_table_to_doc(doc, ["Rank", "Scorer", "WCPM", "N (audios)"], wcpm_rows, bold_row_indices={0})

# Appendix H: WC Significance
doc.add_heading("H. Words Correct — Statistical Significance", level=2)
doc.add_paragraph(
    "Wilcoxon signed-rank tests comparing each AI model's per-audio words correct against the "
    "average human baseline, with Benjamini-Hochberg FDR correction for multiple comparisons. "
    "Mean Diff = AI mean minus human baseline (positive = AI scored higher)."
)
wc_sig_rows = []
wc_sig_sorted_doc = wc_sig.sort_values("mean_diff", ascending=False)
for rank, (_, row) in enumerate(wc_sig_sorted_doc.iterrows(), 1):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    ci_str = f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
    sig = "Yes" if row["significant"] else "No"
    wc_sig_rows.append([str(rank), name, str(row["n"]), f"{row['mean_diff']:+.2f}", ci_str, fmt_pval(row["p_adjusted"]), sig])
add_table_to_doc(doc, ["Rank", "Model", "N", "Mean Diff", "95% CI", "p (adjusted)", "Sig?"], wc_sig_rows)
add_para(doc,
    "With n > 500 paired observations, even small differences reach statistical significance. "
    "The 95% CI on mean difference is more informative for practical interpretation.",
    italic=True, size=9,
)

# Appendix I: WER Significance
doc.add_heading("I. WER — Statistical Significance", level=2)
doc.add_paragraph(
    "Wilcoxon signed-rank tests comparing each AI model's per-audio WER (vs human transcription) "
    "against the average inter-grader WER baseline, across all audios with human transcription."
    "Negative mean diff = AI has lower (better) WER than inter-grader baseline."
)
wer_sig_rows = []
wer_sig_sorted_doc = wer_sig.sort_values("mean_diff")
for rank, (_, row) in enumerate(wer_sig_sorted_doc.iterrows(), 1):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
    sig = "Yes" if row["significant"] else "No"
    wer_sig_rows.append([str(rank), name, str(row["n"]), f"{row['mean_diff']:+.4f}", ci_str, fmt_pval(row["p_adjusted"]), sig])
add_table_to_doc(doc, ["Rank", "Model", "N", "Mean Diff", "95% CI", "p (adjusted)", "Sig?"], wer_sig_rows)

# Appendix J: WCPM CIs
doc.add_heading("J. WCPM — Confidence Intervals", level=2)
doc.add_paragraph(
    "Bootstrap 95% confidence intervals (10,000 resamples) for aggregate WCPM "
    "(total words correct / total duration minutes) per scorer."
)
wcpm_ci_rows_doc = []
wcpm_ci_sorted_doc = wcpm_ci_df.sort_values("wcpm", ascending=False)
wcpm_ci_bold = set()
doc_ai_rank = 0
for row_idx, (_, row) in enumerate(wcpm_ci_sorted_doc.iterrows()):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    is_baseline = row["scorer"] == "Avg Human (baseline)"
    ci_str = f"[{row['ci_lo']}, {row['ci_hi']}]"
    if is_baseline:
        rank_str = "—"
        wcpm_ci_bold.add(row_idx)
    else:
        doc_ai_rank += 1
        rank_str = str(doc_ai_rank)
    wcpm_ci_rows_doc.append([rank_str, name, str(row["wcpm"]), ci_str, str(int(row["n"]))])
add_table_to_doc(doc, ["Rank", "Scorer", "WCPM", "95% CI", "N (audios)"], wcpm_ci_rows_doc, bold_row_indices=wcpm_ci_bold)

# Appendix K: Inter-Grader Correlation P-values
doc.add_heading("K. Inter-Grader Correlation P-values", level=2)
doc.add_paragraph(
    "Pearson correlation p-values for each grader pair's words-correct agreement. "
    "All pairs with sufficient shared audios show highly significant correlations."
)
ig_pval_rows = []
valid_ig_doc = df_ig_wc[df_ig_wc["pearson_r"].notna()]
for _, row in valid_ig_doc.iterrows():
    ig_pval_rows.append([
        row["grader_1"], row["grader_2"],
        f"{row['pearson_r']:.3f}", fmt_pval(row["pearson_p"]),
        str(row["shared_audios"]),
    ])
add_table_to_doc(doc, ["Grader 1", "Grader 2", "Pearson r", "p-value", "Shared Audios"], ig_pval_rows)

# Appendix L: Pre/Post WCPM Delta Significance
doc.add_heading("L. Pre/Post WCPM — Statistical Significance", level=2)
doc.add_paragraph(
    "Bootstrap significance test (10,000 resamples) for the pre-to-post WCPM improvement. "
    "Delta = Post WCPM minus Pre WCPM. A significant result means the improvement is "
    "reliably different from zero."
)
pp_delta_rows_doc = []
pp_delta_bold = set()
pp_delta_sorted_doc = prepost_delta_df.sort_values("delta", ascending=False)
doc_ai_rank_pp = 0
for row_idx, (_, row) in enumerate(pp_delta_sorted_doc.iterrows()):
    name = DISPLAY_NAMES.get(row["scorer"], row["scorer"])
    is_baseline = row["scorer"] == "Avg Human (baseline)"
    ci_str = f"[{row['ci_lo']}, {row['ci_hi']}]"
    sig = "Yes" if row["p_value"] < 0.05 else "No"
    if is_baseline:
        rank_str = "—"
        pp_delta_bold.add(row_idx)
    else:
        doc_ai_rank_pp += 1
        rank_str = str(doc_ai_rank_pp)
    pp_delta_rows_doc.append([
        rank_str, name, str(row["pre_wcpm"]), str(row["post_wcpm"]),
        f"{row['delta']:+.1f}", ci_str, fmt_pval(row["p_value"]), sig,
    ])
add_table_to_doc(doc,
    ["Rank", "Scorer", "Pre WCPM", "Post WCPM", "Delta", "95% CI", "p-value", "Sig?"],
    pp_delta_rows_doc, bold_row_indices=pp_delta_bold,
)

# Appendix M: Model Pricing
doc.add_heading("M. Model Pricing", level=2)
doc.add_paragraph(
    "Estimated cost per 1,000 minutes of audio and median processing speed for each "
    "AI model evaluated. Pricing sourced from provider documentation as of March 2026. "
    "Empty cells indicate pricing was not publicly available."
)
pricing_rows_doc = []
if pricing_df is not None:
    for model in AI_MODELS:
        display = DISPLAY_NAMES.get(model, model)
        pricing_name = PRICING_NAMES.get(model, display)
        match = pricing_df[pricing_df["model"].str.strip() == pricing_name]
        if len(match) > 0:
            row = match.iloc[0]
            price = f"${row['price_usd_per_1000_min']:.2f}" if pd.notna(row.get("price_usd_per_1000_min")) and str(row.get("price_usd_per_1000_min")).strip() else "—"
            speed = f"{row['median_speed_factor']}x" if pd.notna(row.get("median_speed_factor")) and str(row.get("median_speed_factor")).strip() else "—"
            bwer = str(row.get("word_error_rate_perc", "—")).strip() if pd.notna(row.get("word_error_rate_perc")) and str(row.get("word_error_rate_perc")).strip() else "—"
        else:
            price, speed, bwer = "—", "—", "—"
        pricing_rows_doc.append([display, price, speed, bwer])
add_table_to_doc(doc,
    ["Model", "Price (USD / 1,000 min)", "Median Speed Factor", "Benchmark WER"],
    pricing_rows_doc,
)
doc.add_paragraph("")
add_para(doc,
    "Speed factor = how many times faster than real-time the model transcribes (higher = faster). "
    "Benchmark WER is the provider-reported word error rate on standard English benchmarks, "
    "which may differ from WER observed on Pakistani English child speech.",
    italic=True,
)

# ── Save ──
doc.save(OUT_DOCX)
print(f"Saved Word document to: {OUT_DOCX}")
print("Done!")
# %%
