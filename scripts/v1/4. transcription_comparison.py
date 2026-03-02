# %%
import pandas as pd
import numpy as np
import json

# Levenshtein distance implementation (no external dependency needed)
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if s1 is None or s2 is None:
        return np.nan
    s1, s2 = str(s1).lower(), str(s2).lower()
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_levenshtein(s1, s2):
    """Calculate normalized Levenshtein distance (0-1 scale, 0 = identical)."""
    if s1 is None or s2 is None:
        return np.nan
    s1, s2 = str(s1), str(s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len

def word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    if reference is None or hypothesis is None:
        return np.nan
    ref_words = str(reference).lower().split()
    hyp_words = str(hypothesis).lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # Levenshtein at word level
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j] + 1,    # deletion
                             d[i, j-1] + 1,     # insertion
                             d[i-1, j-1] + 1)   # substitution

    return d[len(ref_words), len(hyp_words)] / len(ref_words)

# %%
# Load data
df = pd.read_csv("../../data/v1/clean/merged_for_analysis.csv")

print(f"Total records: {len(df)}")
print(f"\nColumns available:")
for col in df.columns:
    print(f"  - {col}")

# %%
# Clean transcriptions for comparison
df['transcription_human_clean'] = df['transcription_human'].str.lower().str.strip()
df['transcription_ai_clean'] = df['transcription_ai'].str.lower().str.strip()

# Remove punctuation for fairer comparison
import re
def clean_text(text):
    if pd.isna(text):
        return None
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    return text.strip()

df['transcription_human_normalized'] = df['transcription_human'].apply(clean_text)
df['transcription_ai_normalized'] = df['transcription_ai'].apply(clean_text)

# %%
# Calculate transcription similarity metrics
print("Calculating transcription similarity metrics...")

df['levenshtein_distance'] = df.apply(
    lambda row: levenshtein_distance(row['transcription_human_normalized'],
                                      row['transcription_ai_normalized']), axis=1)

df['levenshtein_normalized'] = df.apply(
    lambda row: normalized_levenshtein(row['transcription_human_normalized'],
                                        row['transcription_ai_normalized']), axis=1)

df['word_error_rate'] = df.apply(
    lambda row: word_error_rate(row['transcription_human_normalized'],
                                 row['transcription_ai_normalized']), axis=1)

# Character-level similarity (1 - normalized_levenshtein)
df['char_similarity'] = 1 - df['levenshtein_normalized']

# Exact match indicator
df['exact_match'] = df['transcription_human_normalized'] == df['transcription_ai_normalized']

# %%
# Parse scores from JSON columns
def parse_json_col(val):
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}

def get_ai_total_score(val):
    """Extract total score from AI feedback JSON."""
    parsed = parse_json_col(val)
    total = 0
    for word, score_data in parsed.items():
        if isinstance(score_data, dict):
            total += score_data.get('normalized_rounded', 0)
        elif isinstance(score_data, (int, float)):
            total += score_data
    return total

def get_human_total_score(val):
    """Extract total score from human feedback JSON."""
    parsed = parse_json_col(val)
    return sum(v for v in parsed.values() if isinstance(v, (int, float)))

# Calculate AI total scores
df['ai_score_total'] = df['feedback_ai_json'].apply(get_ai_total_score)

# Human scores should already be in score_human_total
# Calculate score difference
df['score_diff'] = df['ai_score_total'] - df['score_human_total']
df['score_diff_abs'] = df['score_diff'].abs()

# %%
# Print summary statistics
print("=" * 80)
print("TRANSCRIPTION COMPARISON: AI vs HUMAN")
print("=" * 80)

print(f"\nDataset Overview:")
print(f"  Total records: {len(df)}")
print(f"  Records with both transcriptions: {df[['transcription_human', 'transcription_ai']].notna().all(axis=1).sum()}")

# %%
print(f"\n{'='*80}")
print("TRANSCRIPTION SIMILARITY METRICS")
print(f"{'='*80}")

# Filter to rows with both transcriptions
df_valid = df[df['transcription_human_normalized'].notna() & df['transcription_ai_normalized'].notna()]

print(f"\nCharacter-Level Metrics (n={len(df_valid)}):")
print(f"  Mean Levenshtein distance: {df_valid['levenshtein_distance'].mean():.2f} (SD: {df_valid['levenshtein_distance'].std():.2f})")
print(f"  Mean normalized Levenshtein: {df_valid['levenshtein_normalized'].mean():.3f} (SD: {df_valid['levenshtein_normalized'].std():.3f})")
print(f"  Mean character similarity: {df_valid['char_similarity'].mean():.1%} (SD: {df_valid['char_similarity'].std():.1%})")

print(f"\nWord-Level Metrics:")
print(f"  Mean Word Error Rate (WER): {df_valid['word_error_rate'].mean():.1%} (SD: {df_valid['word_error_rate'].std():.1%})")
print(f"  Exact match rate: {df_valid['exact_match'].mean():.1%}")

# %%
print(f"\n{'='*80}")
print("SCORE COMPARISON: AI vs HUMAN")
print(f"{'='*80}")

# Filter to rows with valid scores
df_scores = df[(df['score_human_total'].notna()) & (df['ai_score_total'].notna())]

print(f"\nScore Summary (n={len(df_scores)}):")
print(f"\n  Human Scores:")
print(f"    Mean: {df_scores['score_human_total'].mean():.2f}")
print(f"    SD: {df_scores['score_human_total'].std():.2f}")
print(f"    Min: {df_scores['score_human_total'].min():.0f}")
print(f"    Max: {df_scores['score_human_total'].max():.0f}")

print(f"\n  AI Scores:")
print(f"    Mean: {df_scores['ai_score_total'].mean():.2f}")
print(f"    SD: {df_scores['ai_score_total'].std():.2f}")
print(f"    Min: {df_scores['ai_score_total'].min():.0f}")
print(f"    Max: {df_scores['ai_score_total'].max():.0f}")

print(f"\n  Score Difference (AI - Human):")
print(f"    Mean: {df_scores['score_diff'].mean():.2f}")
print(f"    SD: {df_scores['score_diff'].std():.2f}")
print(f"    Mean absolute diff: {df_scores['score_diff_abs'].mean():.2f}")

# Direction of bias
if df_scores['score_diff'].mean() > 0:
    print(f"    Bias: AI tends to score HIGHER than humans")
else:
    print(f"    Bias: AI tends to score LOWER than humans")

# %%
# Correlation between transcription similarity and score agreement
print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print(f"{'='*80}")

df_corr = df_valid[df_valid['score_diff_abs'].notna()]

if len(df_corr) > 2:
    corr_lev_score = df_corr['levenshtein_normalized'].corr(df_corr['score_diff_abs'])
    corr_wer_score = df_corr['word_error_rate'].corr(df_corr['score_diff_abs'])
    corr_sim_score = df_corr['char_similarity'].corr(df_corr['score_diff_abs'])

    print(f"\nCorrelation between transcription metrics and score disagreement:")
    print(f"  Levenshtein (normalized) vs |score diff|: r = {corr_lev_score:.3f}")
    print(f"  WER vs |score diff|: r = {corr_wer_score:.3f}")
    print(f"  Character similarity vs |score diff|: r = {corr_sim_score:.3f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if abs(corr_lev_score) < 0.3:
        print(f"    Weak correlation - transcription differences don't strongly predict score differences")
    elif abs(corr_lev_score) < 0.6:
        print(f"    Moderate correlation - some relationship between transcription and score differences")
    else:
        print(f"    Strong correlation - transcription differences predict score differences")

# %%
# Breakdown by grade
print(f"\n{'='*80}")
print("BREAKDOWN BY GRADE")
print(f"{'='*80}")

if 'grade' in df.columns:
    grade_stats = df_valid.groupby('grade').agg({
        'char_similarity': ['mean', 'std', 'count'],
        'word_error_rate': ['mean', 'std'],
        'score_human_total': 'mean',
        'ai_score_total': 'mean',
        'score_diff': ['mean', 'std'],
        'score_diff_abs': 'mean',
        'exact_match': 'mean'
    }).round(3)

    grade_stats.columns = ['char_sim_mean', 'char_sim_std', 'n', 'wer_mean', 'wer_std',
                           'human_score_mean', 'ai_score_mean', 'score_diff_mean', 'score_diff_std',
                           'score_diff_abs_mean', 'exact_match_rate']

    print(f"\nBy Grade:")
    print("-" * 100)
    print(f"{'Grade':<8} {'n':>6} {'Char Sim':>12} {'WER':>12} {'Human Score':>12} {'AI Score':>10} {'Score Diff':>12} {'Exact Match':>12}")
    print("-" * 100)
    for idx, row in grade_stats.iterrows():
        print(f"{idx:<8} {row['n']:>6.0f} {row['char_sim_mean']:>10.1%} {row['wer_mean']:>10.1%} {row['human_score_mean']:>12.2f} {row['ai_score_mean']:>10.2f} {row['score_diff_mean']:>+10.2f} {row['exact_match_rate']:>12.1%}")

    print(f"\nDetailed by Grade:")
    print("-" * 100)
    for idx, row in grade_stats.iterrows():
        print(f"\n  Grade {idx} (n={row['n']:.0f}):")
        print(f"    Transcription: char_sim={row['char_sim_mean']:.1%} (SD: {row['char_sim_std']:.1%}), WER={row['wer_mean']:.1%} (SD: {row['wer_std']:.1%}), exact_match={row['exact_match_rate']:.1%}")
        print(f"    Scores: human={row['human_score_mean']:.2f}, AI={row['ai_score_mean']:.2f}, diff={row['score_diff_mean']:+.2f} (SD: {row['score_diff_std']:.2f}), |diff|={row['score_diff_abs_mean']:.2f}")

# %%
# Breakdown by question
print(f"\n{'='*80}")
print("BREAKDOWN BY QUESTION")
print(f"{'='*80}")

if 'question_text' in df.columns:
    question_stats = df_valid.groupby('question_text').agg({
        'char_similarity': ['mean', 'std', 'count'],
        'word_error_rate': ['mean', 'std'],
        'score_human_total': 'mean',
        'ai_score_total': 'mean',
        'score_diff': ['mean', 'std'],
        'score_diff_abs': 'mean',
        'exact_match': 'mean'
    }).round(3)

    question_stats.columns = ['char_sim_mean', 'char_sim_std', 'n', 'wer_mean', 'wer_std',
                              'human_score_mean', 'ai_score_mean', 'score_diff_mean', 'score_diff_std',
                              'score_diff_abs_mean', 'exact_match_rate']
    question_stats = question_stats.sort_values('char_sim_mean', ascending=True)

    print(f"\nBy Question (sorted by character similarity):")
    print("-" * 100)
    for idx, row in question_stats.iterrows():
        q_text = idx[:60] + "..." if len(idx) > 60 else idx
        print(f"\n  \"{q_text}\"")
        print(f"    n={row['n']:.0f}, exact_match={row['exact_match_rate']:.1%}")
        print(f"    Transcription: char_sim={row['char_sim_mean']:.1%} (SD: {row['char_sim_std']:.1%}), WER={row['wer_mean']:.1%} (SD: {row['wer_std']:.1%})")
        print(f"    Scores: human={row['human_score_mean']:.2f}, AI={row['ai_score_mean']:.2f}, diff={row['score_diff_mean']:+.2f} (SD: {row['score_diff_std']:.2f})")

# %%
# Cross-tabulation: Grade x Question
print(f"\n{'='*80}")
print("BREAKDOWN BY GRADE x QUESTION")
print(f"{'='*80}")

if 'grade' in df.columns and 'question_text' in df.columns:
    grade_question_stats = df_valid.groupby(['grade', 'question_text']).agg({
        'char_similarity': ['mean', 'count'],
        'word_error_rate': 'mean',
        'score_diff': 'mean',
        'score_diff_abs': 'mean'
    }).round(3)

    grade_question_stats.columns = ['char_sim_mean', 'n', 'wer_mean', 'score_diff_mean', 'score_diff_abs_mean']
    grade_question_stats = grade_question_stats.reset_index()

    print(f"\nGrade x Question Summary:")
    print("-" * 120)
    print(f"{'Grade':<8} {'Question':<55} {'n':>5} {'Char Sim':>10} {'WER':>8} {'Score Diff':>12}")
    print("-" * 120)
    for _, row in grade_question_stats.iterrows():
        q_text = row['question_text'][:50] + "..." if len(row['question_text']) > 50 else row['question_text']
        print(f"{row['grade']:<8} {q_text:<55} {row['n']:>5.0f} {row['char_sim_mean']:>9.1%} {row['wer_mean']:>7.1%} {row['score_diff_mean']:>+11.2f}")

# %%
# Pre vs Post comparison
print(f"\n{'='*80}")
print("BREAKDOWN BY PRE/POST")
print(f"{'='*80}")

if 'pre_or_post' in df.columns:
    pre_post_stats = df_valid.groupby('pre_or_post').agg({
        'char_similarity': ['mean', 'std', 'count'],
        'word_error_rate': 'mean',
        'score_diff': 'mean',
        'score_diff_abs': 'mean'
    }).round(3)

    pre_post_stats.columns = ['char_sim_mean', 'char_sim_std', 'n', 'wer_mean', 'score_diff_mean', 'score_diff_abs_mean']

    print(f"\nPre/Post Comparison:")
    print("-" * 80)
    for idx, row in pre_post_stats.iterrows():
        print(f"  {idx}: n={row['n']:.0f}, char_sim={row['char_sim_mean']:.1%} (SD: {row['char_sim_std']:.1%}), WER={row['wer_mean']:.1%}, score_diff={row['score_diff_mean']:.2f}")

# %%
# Show examples of high and low similarity
print(f"\n{'='*80}")
print("EXAMPLE COMPARISONS")
print(f"{'='*80}")

# Highest similarity (best matches)
print(f"\nHighest Similarity (Top 5):")
print("-" * 80)
top_matches = df_valid.nsmallest(5, 'levenshtein_distance')[['transcription_human_normalized', 'transcription_ai_normalized', 'char_similarity', 'score_human_total', 'ai_score_total']]
for i, (_, row) in enumerate(top_matches.iterrows(), 1):
    print(f"\n  {i}. Similarity: {row['char_similarity']:.1%}")
    print(f"     Human: \"{row['transcription_human_normalized']}\"")
    print(f"     AI:    \"{row['transcription_ai_normalized']}\"")
    print(f"     Scores: Human={row['score_human_total']:.0f}, AI={row['ai_score_total']:.0f}")

# Lowest similarity (worst matches)
print(f"\n\nLowest Similarity (Bottom 5):")
print("-" * 80)
bottom_matches = df_valid.nlargest(5, 'levenshtein_distance')[['transcription_human_normalized', 'transcription_ai_normalized', 'char_similarity', 'score_human_total', 'ai_score_total']]
for i, (_, row) in enumerate(bottom_matches.iterrows(), 1):
    print(f"\n  {i}. Similarity: {row['char_similarity']:.1%}")
    print(f"     Human: \"{row['transcription_human_normalized']}\"")
    print(f"     AI:    \"{row['transcription_ai_normalized']}\"")
    print(f"     Scores: Human={row['score_human_total']:.0f}, AI={row['ai_score_total']:.0f}")

# %%
# Save detailed comparison to CSV
output_cols = [
    'student_profile_id', 'question_text', 'pre_or_post',
    'transcription_human', 'transcription_ai',
    'transcription_human_normalized', 'transcription_ai_normalized',
    'levenshtein_distance', 'levenshtein_normalized', 'char_similarity',
    'word_error_rate', 'exact_match',
    'score_human_total', 'ai_score_total', 'score_diff', 'score_diff_abs',
    'grader_name'
]

# Keep only columns that exist
output_cols = [c for c in output_cols if c in df.columns]
df_output = df[output_cols].copy()

output_path = '../../data/v1/clean/transcription_comparison.csv'
df_output.to_csv(output_path, index=False)
print(f"\n\nSaved detailed comparison to: {output_path}")

# %%
# Summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")

summary_data = {
    'Metric': [
        'Total records',
        'Character similarity (mean)',
        'Character similarity (SD)',
        'Word Error Rate (mean)',
        'Word Error Rate (SD)',
        'Exact match rate',
        'Human score (mean)',
        'AI score (mean)',
        'Score difference (mean)',
        'Score difference (SD)',
        'Abs score difference (mean)'
    ],
    'Value': [
        f"{len(df_valid)}",
        f"{df_valid['char_similarity'].mean():.1%}",
        f"{df_valid['char_similarity'].std():.1%}",
        f"{df_valid['word_error_rate'].mean():.1%}",
        f"{df_valid['word_error_rate'].std():.1%}",
        f"{df_valid['exact_match'].mean():.1%}",
        f"{df_scores['score_human_total'].mean():.2f}",
        f"{df_scores['ai_score_total'].mean():.2f}",
        f"{df_scores['score_diff'].mean():.2f}",
        f"{df_scores['score_diff'].std():.2f}",
        f"{df_scores['score_diff_abs'].mean():.2f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# %%
