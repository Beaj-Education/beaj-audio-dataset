# %%
import pandas as pd
import json
import numpy as np

df_comb = pd.read_csv("../../data/v1/clean/merged_for_analysis.csv")

# Find audio files that were graded by multiple graders
graders_per_file = df_comb.groupby('audio_filename')['grader_name'].nunique()
multi_grader_files = graders_per_file[graders_per_file > 1].index.tolist()

print(f"Audio files graded by multiple graders: {len(multi_grader_files)}")

# Filter to only files with multiple graders
df_multi_grader = df_comb[df_comb['audio_filename'].isin(multi_grader_files)]

# Show which graders rated each file
multi_grader_summary = df_multi_grader.groupby('audio_filename').agg({
    'grader_name': lambda x: list(x.unique()),
    'score_human_total': list
}).reset_index()
multi_grader_summary.columns = ['audio_filename', 'graders', 'scores']

print(f"\nSummary of multi-grader files:")
print(multi_grader_summary)

# %%
# Helper function to parse JSON columns
def parse_json_col(val):
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}

def parse_ai_scores(val):
    """Parse AI scores, extracting normalized_rounded (0-2 scale) from nested structure."""
    raw = parse_json_col(val)
    result = {}
    for word, score_data in raw.items():
        if isinstance(score_data, dict):
            # Nested structure: {"word": {"actual_score": X, "normalized_rounded": Y}}
            result[word] = score_data.get('normalized_rounded', np.nan)
        else:
            # Simple structure: {"word": score}
            result[word] = score_data
    return result

# Parse feedback columns
df_multi_grader = df_multi_grader.copy()
df_multi_grader['human_scores'] = df_multi_grader['feedback_human_json'].apply(parse_json_col)
df_multi_grader['ai_scores'] = df_multi_grader['feedback_ai_json'].apply(parse_ai_scores)

# %%
# Word-level comparison for pair-graded files
comparison_results = []

for audio_file in multi_grader_files:
    file_data = df_multi_grader[df_multi_grader['audio_filename'] == audio_file]

    # Get all words across all graders
    all_words = set()
    for _, row in file_data.iterrows():
        all_words.update(row['human_scores'].keys())
    all_words = sorted(all_words)

    if not all_words:
        continue

    # Build comparison table for this file
    rows = []
    for _, row in file_data.iterrows():
        grader = row['grader_name']
        human_scores = row['human_scores']
        word_scores = {w: human_scores.get(w, np.nan) for w in all_words}
        word_scores['grader'] = grader
        word_scores['total'] = row['score_human_total']
        word_scores['audio_filename'] = audio_file
        rows.append(word_scores)

    # Add AI scorer row (use first row's AI scores since they should be same)
    ai_scores = file_data.iloc[0]['ai_scores']
    ai_row = {w: ai_scores.get(w, np.nan) for w in all_words}
    ai_row['grader'] = 'AI'
    ai_row['total'] = sum(v for v in ai_scores.values() if isinstance(v, (int, float)))
    ai_row['audio_filename'] = audio_file
    rows.append(ai_row)

    comparison_results.extend(rows)

df_comparison = pd.DataFrame(comparison_results)

# %%
# Build comparison tables matching the screenshot structure
word_cols = [c for c in df_comparison.columns if c not in ['grader', 'total', 'audio_filename']]

def build_comparison_table(audio_file):
    """Build a comparison table for a single audio file matching the screenshot format."""
    file_df = df_comparison[df_comparison['audio_filename'] == audio_file].copy()
    human_df = file_df[file_df['grader'] != 'AI']

    if human_df.empty:
        return None, None

    # Calculate human average total
    human_avg_total = human_df['total'].mean()

    # Add Dev and Abs dev columns to main table
    file_df['Dev'] = file_df['total'] - human_avg_total
    file_df['Abs dev'] = file_df['Dev'].abs()

    # Create Human avg row
    human_avg_row = {'grader': 'Human avg'}
    for col in word_cols:
        human_avg_row[col] = human_df[col].mean()
    human_avg_row['total'] = human_avg_total
    human_avg_row['Dev'] = np.nan
    human_avg_row['Abs dev'] = np.nan

    # Main scores table
    main_table = pd.concat([file_df, pd.DataFrame([human_avg_row])], ignore_index=True)

    # Per-word deviations table (human scorers only)
    dev_rows = []
    for _, row in human_df.iterrows():
        dev_row = {'grader': row['grader']}
        for col in word_cols:
            dev_row[col] = row[col] - human_avg_row[col] if pd.notna(row[col]) else np.nan
        dev_rows.append(dev_row)

    dev_table = pd.DataFrame(dev_rows)

    return main_table, dev_table

# %%
# Build tables for all pair-graded files
all_main_tables = {}
all_dev_tables = {}

for audio_file in multi_grader_files:
    main_table, dev_table = build_comparison_table(audio_file)
    if main_table is not None:
        all_main_tables[audio_file] = main_table
        all_dev_tables[audio_file] = dev_table

# %%
# Display example for first file (like the screenshot)
if multi_grader_files:
    example_file = multi_grader_files[0]
    print(f"\n{'='*80}")
    print(f"EXAMPLE: {example_file}")
    print(f"{'='*80}")

    main_table = all_main_tables[example_file]
    dev_table = all_dev_tables[example_file]

    # Get word columns for this file
    file_word_cols = [c for c in main_table.columns if c not in ['grader', 'total', 'Dev', 'Abs dev', 'audio_filename']]

    print("\nTable 1: Scores by grader")
    print("-" * 60)
    display_cols = ['grader'] + file_word_cols + ['total', 'Dev', 'Abs dev']
    print(main_table[display_cols].to_string(index=False))

    print("\n\nTable 2: Per-word deviations from Human avg")
    print("-" * 60)
    print(dev_table[['grader'] + file_word_cols].to_string(index=False))

# %%
# Summary statistics across all pair-graded files
print(f"\n{'='*80}")
print("INTER-RATER RELIABILITY SUMMARY")
print(f"{'='*80}")

# Collect stats from all files
all_human_std = []
all_human_range = []
all_ai_dev = []

for audio_file, main_table in all_main_tables.items():
    human_rows = main_table[(main_table['grader'] != 'AI') & (main_table['grader'] != 'Human avg')]
    ai_row = main_table[main_table['grader'] == 'AI']

    if len(human_rows) > 1:
        all_human_std.append(human_rows['total'].std())
        all_human_range.append(human_rows['total'].max() - human_rows['total'].min())

    if not ai_row.empty:
        all_ai_dev.append(ai_row['Dev'].values[0])

print(f"\nHuman inter-rater agreement:")
print(f"  Avg std deviation in total scores: {np.mean(all_human_std):.2f} (SD: {np.std(all_human_std):.2f})")
print(f"  Avg score range (max-min): {np.mean(all_human_range):.2f} (SD: {np.std(all_human_range):.2f})")

print(f"\nAI vs Human comparison:")
print(f"  Mean AI deviation from human avg: {np.mean(all_ai_dev):.2f}")
print(f"  Std of AI deviation: {np.std(all_ai_dev):.2f}")
print(f"  AI tends to score {'higher' if np.mean(all_ai_dev) > 0 else 'lower'} than humans")

# %%
# Cohen's Kappa and Percent Agreement at word level
try:
    from sklearn.metrics import cohen_kappa_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: Install scikit-learn for Cohen's Kappa: pip install scikit-learn")

def calculate_agreement_metrics(scores1, scores2):
    """Calculate percent agreement and Cohen's Kappa between two score arrays."""
    # Filter out NaN pairs
    valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))
    s1 = scores1[valid_mask].astype(int)
    s2 = scores2[valid_mask].astype(int)

    if len(s1) == 0:
        return np.nan, np.nan, np.nan, 0

    # Exact percent agreement
    exact_agreement = np.mean(s1 == s2) * 100

    # Adjacent agreement (within 1 point)
    adjacent_agreement = np.mean(np.abs(s1 - s2) <= 1) * 100

    # Cohen's Kappa (weighted for ordinal data)
    if SKLEARN_AVAILABLE:
        try:
            kappa = cohen_kappa_score(s1, s2, weights='quadratic')
        except:
            kappa = np.nan
    else:
        kappa = np.nan

    return exact_agreement, adjacent_agreement, kappa, len(s1)

# Collect all word-level scores for human graders
human_word_scores = []
ai_word_scores = []

for audio_file in multi_grader_files:
    file_df = df_comparison[df_comparison['audio_filename'] == audio_file]
    human_df = file_df[file_df['grader'] != 'AI']
    ai_df = file_df[file_df['grader'] == 'AI']

    if len(human_df) < 2:
        continue

    graders = human_df['grader'].tolist()
    file_word_cols = [c for c in human_df.columns if c not in ['grader', 'total', 'audio_filename']]

    # Collect pairwise human scores
    for i in range(len(graders)):
        for j in range(i + 1, len(graders)):
            g1_scores = human_df[human_df['grader'] == graders[i]][file_word_cols].values.flatten()
            g2_scores = human_df[human_df['grader'] == graders[j]][file_word_cols].values.flatten()
            human_word_scores.append((graders[i], graders[j], g1_scores, g2_scores))

    # Collect human avg vs AI scores
    if not ai_df.empty:
        human_avg_scores = human_df[file_word_cols].mean().values
        ai_scores_arr = ai_df[file_word_cols].values.flatten()
        ai_word_scores.append((human_avg_scores, ai_scores_arr))

# Calculate human inter-rater metrics
print(f"\n{'='*80}")
print("WORD-LEVEL AGREEMENT METRICS")
print(f"{'='*80}")

all_exact = []
all_adjacent = []
all_kappa = []
total_words = 0

for g1, g2, s1, s2 in human_word_scores:
    exact, adjacent, kappa, n = calculate_agreement_metrics(s1, s2)
    if not np.isnan(exact):
        all_exact.append(exact)
        all_adjacent.append(adjacent)
        if not np.isnan(kappa):
            all_kappa.append(kappa)
        total_words += n

print(f"\nHuman Inter-Rater Agreement (word level):")
print(f"  Total word comparisons: {total_words}")
print(f"  Exact agreement: {np.mean(all_exact):.1f}% (SD: {np.std(all_exact):.1f}%)")
print(f"  Adjacent agreement (within 1 point): {np.mean(all_adjacent):.1f}% (SD: {np.std(all_adjacent):.1f}%)")
if all_kappa:
    print(f"  Cohen's Kappa (quadratic weighted): {np.mean(all_kappa):.3f} (SD: {np.std(all_kappa):.3f})")
    kappa_val = np.mean(all_kappa)
    if kappa_val < 0:
        interpretation = "Poor (less than chance)"
    elif kappa_val < 0.20:
        interpretation = "Slight"
    elif kappa_val < 0.40:
        interpretation = "Fair"
    elif kappa_val < 0.60:
        interpretation = "Moderate"
    elif kappa_val < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost perfect"
    print(f"  Kappa interpretation: {interpretation}")

# AI vs Human agreement
if ai_word_scores:
    ai_exact = []
    ai_adjacent = []
    ai_kappa = []
    ai_total = 0

    for human_avg, ai_arr in ai_word_scores:
        # Round human avg for comparison
        human_rounded = np.round(human_avg)
        exact, adjacent, kappa, n = calculate_agreement_metrics(human_rounded, ai_arr)
        if not np.isnan(exact):
            ai_exact.append(exact)
            ai_adjacent.append(adjacent)
            if not np.isnan(kappa):
                ai_kappa.append(kappa)
            ai_total += n

    print(f"\nAI vs Human Agreement (word level):")
    print(f"  Total word comparisons: {ai_total}")
    print(f"  Exact agreement: {np.mean(ai_exact):.1f}% (SD: {np.std(ai_exact):.1f}%)")
    print(f"  Adjacent agreement (within 1 point): {np.mean(ai_adjacent):.1f}% (SD: {np.std(ai_adjacent):.1f}%)")
    if ai_kappa:
        print(f"  Cohen's Kappa (quadratic weighted): {np.mean(ai_kappa):.3f} (SD: {np.std(ai_kappa):.3f})")

# %%
# Create detailed word-by-word comparison table for all pair-graded files
print(f"\n{'='*80}")
print("DETAILED WORD-BY-WORD COMPARISON")
print(f"{'='*80}")

all_comparisons = []

for audio_file in multi_grader_files:
    file_df = df_comparison[df_comparison['audio_filename'] == audio_file]
    human_df = file_df[file_df['grader'] != 'AI']
    ai_df = file_df[file_df['grader'] == 'AI']

    if len(human_df) < 2 or ai_df.empty:
        continue

    graders = human_df['grader'].tolist()
    file_word_cols = [c for c in human_df.columns if c not in ['grader', 'total', 'audio_filename']]

    # For each word, get scores from all graders
    for word in file_word_cols:
        row_data = {
            'audio_filename': audio_file,
            'word': word,
        }

        # Get each human grader's score
        for i, grader in enumerate(graders):
            score = human_df[human_df['grader'] == grader][word].values[0]
            row_data[f'human_{i+1}'] = score
            row_data[f'human_{i+1}_name'] = grader

        # Human average
        human_scores = [row_data.get(f'human_{i+1}') for i in range(len(graders))]
        human_scores_clean = [s for s in human_scores if pd.notna(s)]
        row_data['human_avg'] = np.mean(human_scores_clean) if human_scores_clean else np.nan

        # AI score
        ai_score = ai_df[word].values[0]
        row_data['ai_score'] = ai_score

        # Difference between AI and human avg
        if pd.notna(row_data['human_avg']) and pd.notna(ai_score):
            row_data['ai_minus_human_avg'] = ai_score - row_data['human_avg']
        else:
            row_data['ai_minus_human_avg'] = np.nan

        all_comparisons.append(row_data)

df_word_comparison = pd.DataFrame(all_comparisons)

# Add totals row for each audio file
totals_list = []
for audio_file in multi_grader_files:
    file_df = df_comparison[df_comparison['audio_filename'] == audio_file]
    human_df = file_df[file_df['grader'] != 'AI']
    ai_df = file_df[file_df['grader'] == 'AI']

    if len(human_df) < 2 or ai_df.empty:
        continue

    graders = human_df['grader'].tolist()
    totals_row = {
        'audio_filename': audio_file,
        'word': 'TOTAL',
    }

    for i, grader in enumerate(graders):
        totals_row[f'human_{i+1}'] = human_df[human_df['grader'] == grader]['total'].values[0]
        totals_row[f'human_{i+1}_name'] = grader

    human_totals = [totals_row.get(f'human_{i+1}') for i in range(len(graders))]
    totals_row['human_avg'] = np.mean([t for t in human_totals if pd.notna(t)])
    totals_row['ai_score'] = ai_df['total'].values[0]
    totals_row['ai_minus_human_avg'] = totals_row['ai_score'] - totals_row['human_avg']

    totals_list.append(totals_row)

df_totals = pd.DataFrame(totals_list)

# Display summary
print("\nTotals by audio file:")
print("-" * 80)
display_cols = ['audio_filename', 'human_1', 'human_2', 'human_avg', 'ai_score', 'ai_minus_human_avg']
available_cols = [c for c in display_cols if c in df_totals.columns]
print(df_totals[available_cols].to_string(index=False))

# Save detailed comparison to CSV
df_word_comparison.to_csv('../../data/v1/clean/human_interrater_word_mismatches.csv', index=False)
df_totals.to_csv('../../data/v1/clean/human_interrater_disagreements.csv', index=False)
print(f"\nSaved word-level comparison to: ../../data/v1/clean/human_interrater_word_mismatches.csv")
print(f"Saved totals comparison to: ../../data/v1/clean/human_interrater_disagreements.csv")

# Show cases where humans disagreed significantly (diff > 1 point on a word)
if 'human_1' in df_word_comparison.columns and 'human_2' in df_word_comparison.columns:
    df_word_comparison['human_diff'] = abs(df_word_comparison['human_1'] - df_word_comparison['human_2'])
    high_disagreement = df_word_comparison[df_word_comparison['human_diff'] > 1]

    if not high_disagreement.empty:
        print(f"\n\nHigh disagreement cases (human diff > 1 point):")
        print("-" * 80)
        print(high_disagreement[['audio_filename', 'word', 'human_1', 'human_2', 'human_diff', 'ai_score']].to_string(index=False))
        high_disagreement.to_csv('../../data/v1/clean/high_disagreement_cases.csv', index=False)
        print(f"\nSaved high disagreement cases to: ../../data/v1/clean/high_disagreement_cases.csv")
