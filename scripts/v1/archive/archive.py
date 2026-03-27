# import json, ast
# from collections import defaultdict, Counter
# import pandas as pd
# import numpy as np
# from scipy import stats
# from statsmodels.stats.multitest import multipletests

# def _parse_dict(x):
#     if pd.isna(x) or x is None or x == '':
#         return {}
#     if isinstance(x, dict):
#         return x
#     if isinstance(x, str):
#         try:
#             return json.loads(x)
#         except:
#             try:
#                 return ast.literal_eval(x)
#             except:
#                 return {}
#     return {}

# rows = []

# for _, r in df_merged.iterrows():
#     h = _parse_dict(r.get('feedback_human_json'))
#     a = _parse_dict(r.get('feedback_ai_json'))

#     if not h or not a:
#         continue

#     for w, hscore in h.items():
#         if w not in a:
#             continue

#         ai_obj = a[w]
#         if not isinstance(ai_obj, dict):
#             continue

#         ai_score = ai_obj.get('normalized_rounded')  # discrete 0/1/2
#         if ai_score is None:
#             continue

#         try:
#             hval = float(hscore)
#             aval = float(ai_score)
#         except:
#             continue

#         rows.append({
#             "word": w,
#             "human_score": hval,
#             "ai_score": aval,
#             "diff_h_minus_ai": hval - aval,
#             "abs_diff": abs(hval - aval),
#             "mismatch": int(hval != aval),
#         })

# word_df = pd.DataFrame(rows)

# # Helper function to compute statistical tests for each word
# def compute_word_stats(group, min_n=3):
#     """Compute paired statistical tests for a word group."""
#     n = len(group)
    
#     stats_dict = {
#         'n': n,
#         'mean_human': group['human_score'].mean(),
#         'mean_ai': group['ai_score'].mean(),
#         'mean_diff': group['diff_h_minus_ai'].mean(),
#         'mae': group['abs_diff'].mean(),
#         'mismatch_rate': group['mismatch'].mean(),
#         'std_human': group['human_score'].std(),
#         'std_ai': group['ai_score'].std(),
#         'std_diff': group['diff_h_minus_ai'].std(),
#     }
    
#     # Only compute statistical tests if we have enough samples
#     if n >= min_n:
#         human_scores = group['human_score'].values
#         ai_scores = group['ai_score'].values
        
#         # Paired t-test
#         try:
#             t_stat, p_ttest = stats.ttest_rel(human_scores, ai_scores)
#             stats_dict['t_statistic'] = t_stat
#             stats_dict['p_ttest'] = p_ttest
#         except:
#             stats_dict['t_statistic'] = np.nan
#             stats_dict['p_ttest'] = np.nan
        
#         # Wilcoxon signed-rank test (non-parametric alternative)
#         try:
#             w_stat, p_wilcoxon = stats.wilcoxon(human_scores, ai_scores, zero_method='zsplit')
#             stats_dict['wilcoxon_statistic'] = w_stat
#             stats_dict['p_wilcoxon'] = p_wilcoxon
#         except:
#             stats_dict['wilcoxon_statistic'] = np.nan
#             stats_dict['p_wilcoxon'] = np.nan
        
#         # Effect size (Cohen's d for paired samples)
#         try:
#             diff = human_scores - ai_scores
#             cohens_d = diff.mean() / diff.std()
#             stats_dict['cohens_d'] = cohens_d
#         except:
#             stats_dict['cohens_d'] = np.nan
#     else:
#         stats_dict['t_statistic'] = np.nan
#         stats_dict['p_ttest'] = np.nan
#         stats_dict['wilcoxon_statistic'] = np.nan
#         stats_dict['p_wilcoxon'] = np.nan
#         stats_dict['cohens_d'] = np.nan
    
#     return pd.Series(stats_dict)

# # Aggregate per word with statistical tests
# print("Computing word-level statistics and significance tests...")
# word_summary = word_df.groupby("word").apply(compute_word_stats).reset_index()

# # Apply multiple comparison correction (both Bonferroni and FDR)
# # Only apply to words with valid p-values
# valid_p_mask = word_summary['p_ttest'].notna()

# if valid_p_mask.sum() > 0:
#     # Bonferroni correction
#     word_summary['p_ttest_bonferroni'] = np.nan
#     word_summary.loc[valid_p_mask, 'p_ttest_bonferroni'] = word_summary.loc[valid_p_mask, 'p_ttest'] * valid_p_mask.sum()
#     word_summary['p_ttest_bonferroni'] = word_summary['p_ttest_bonferroni'].clip(upper=1.0)  # cap at 1.0
    
#     # FDR correction (Benjamini-Hochberg)
#     _, p_fdr, _, _ = multipletests(word_summary.loc[valid_p_mask, 'p_ttest'], method='fdr_bh')
#     word_summary['p_ttest_fdr'] = np.nan
#     word_summary.loc[valid_p_mask, 'p_ttest_fdr'] = p_fdr
    
#     # Same for Wilcoxon
#     valid_w_mask = word_summary['p_wilcoxon'].notna()
#     if valid_w_mask.sum() > 0:
#         word_summary['p_wilcoxon_bonferroni'] = np.nan
#         word_summary.loc[valid_w_mask, 'p_wilcoxon_bonferroni'] = word_summary.loc[valid_w_mask, 'p_wilcoxon'] * valid_w_mask.sum()
#         word_summary['p_wilcoxon_bonferroni'] = word_summary['p_wilcoxon_bonferroni'].clip(upper=1.0)
        
#         _, p_w_fdr, _, _ = multipletests(word_summary.loc[valid_w_mask, 'p_wilcoxon'], method='fdr_bh')
#         word_summary['p_wilcoxon_fdr'] = np.nan
#         word_summary.loc[valid_w_mask, 'p_wilcoxon_fdr'] = p_w_fdr
# else:
#     word_summary['p_ttest_bonferroni'] = np.nan
#     word_summary['p_ttest_fdr'] = np.nan
#     word_summary['p_wilcoxon_bonferroni'] = np.nan
#     word_summary['p_wilcoxon_fdr'] = np.nan

# # Add significance flags (using FDR-corrected p-values at α=0.05)
# word_summary['significant_fdr_0.05'] = word_summary['p_ttest_fdr'] < 0.05
# word_summary['significant_bonferroni_0.05'] = word_summary['p_ttest_bonferroni'] < 0.05

# # Sort by significance and effect size
# word_summary = word_summary.sort_values(['significant_fdr_0.05', 'mae', 'n'], ascending=[False, False, False])

# # Filter to words that appear often enough (n>=10) for meaningful analysis
# word_summary_filtered = word_summary[word_summary['n'] >= 10].copy()

# out_path = "../data/clean/word_level_summary.csv"
# word_summary_filtered.to_csv(out_path, index=False)

# print(f"\nSaved: {out_path}")
# print(f"Total unique words analyzed: {len(word_summary)}")
# print(f"Words with n>=10: {len(word_summary_filtered)}")
# print(f"Statistically significant (FDR p<0.05, n>=10): {word_summary_filtered['significant_fdr_0.05'].sum()}")
# print(f"Statistically significant (Bonferroni p<0.05, n>=10): {word_summary_filtered['significant_bonferroni_0.05'].sum()}")

# print("\n" + "="*80)
# print("Top 20 words by MAE (n>=10):")
# print("="*80)
# display_cols = ['word', 'n', 'mean_human', 'mean_ai', 'mean_diff', 'mae', 'cohens_d', 
#                 'p_ttest', 'p_ttest_fdr', 'significant_fdr_0.05']
# print(word_summary_filtered[display_cols].head(20).to_string(index=False))

# print("\n" + "="*80)
# print("Statistically significant words (FDR p<0.05, sorted by effect size):")
# print("="*80)
# sig_words = word_summary_filtered[word_summary_filtered['significant_fdr_0.05']].copy()
# if len(sig_words) > 0:
#     sig_words = sig_words.sort_values('cohens_d', key=abs, ascending=False)
#     print(sig_words[display_cols].head(30).to_string(index=False))
# else:
#     print("No statistically significant words found at FDR p<0.05 threshold.")

# # %%
# # Visualization: Top statistically significant words
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Get top significant words by absolute effect size
# sig_words = word_summary_filtered[word_summary_filtered['significant_fdr_0.05']].copy()

# if len(sig_words) > 0:
#     # Sort by absolute Cohen's d (effect size) and take top 15
#     sig_words_plot = sig_words.sort_values('cohens_d', key=abs, ascending=False).head(15)
    
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
#     # 1. Bar chart: Mean difference for top significant words
#     ax1 = axes[0, 0]
#     words_ordered = sig_words_plot.sort_values('mean_diff', ascending=True)['word'].values
#     y_pos = np.arange(len(words_ordered))
#     colors = ['red' if x < 0 else 'green' for x in sig_words_plot.sort_values('mean_diff', ascending=True)['mean_diff']]
    
#     ax1.barh(y_pos, sig_words_plot.sort_values('mean_diff', ascending=True)['mean_diff'], color=colors, alpha=0.7)
#     ax1.set_yticks(y_pos)
#     ax1.set_yticklabels(words_ordered)
#     ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
#     ax1.set_xlabel('Mean Difference (Human - AI)', fontsize=12)
#     ax1.set_title('Top Significant Words: Mean Score Difference\n(Green=Human>AI, Red=AI>Human)', fontsize=14, fontweight='bold')
#     ax1.grid(axis='x', alpha=0.3)
    
#     # 2. Scatter: Human vs AI for significant words
#     ax2 = axes[0, 1]
#     ax2.scatter(sig_words_plot['mean_ai'], sig_words_plot['mean_human'], 
#                 s=sig_words_plot['n']*10, alpha=0.6, c=sig_words_plot['cohens_d'], cmap='RdYlGn')
#     ax2.plot([0, 2], [0, 2], 'k--', alpha=0.5, label='Perfect Agreement')
#     ax2.set_xlabel('Mean AI Score', fontsize=12)
#     ax2.set_ylabel('Mean Human Score', fontsize=12)
#     ax2.set_title('Human vs AI Scores for Significant Words\n(Size=Count, Color=Effect Size)', fontsize=14, fontweight='bold')
#     ax2.legend()
#     ax2.grid(alpha=0.3)
    
#     # Add annotations for top 5
#     for _, row in sig_words_plot.head(5).iterrows():
#         ax2.annotate(row['word'], (row['mean_ai'], row['mean_human']), 
#                     fontsize=9, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
#     # 3. Effect size (Cohen's d) bar chart
#     ax3 = axes[1, 0]
#     words_effect = sig_words_plot.sort_values('cohens_d', ascending=True)['word'].values
#     y_pos_effect = np.arange(len(words_effect))
#     colors_effect = ['red' if x < 0 else 'green' for x in sig_words_plot.sort_values('cohens_d', ascending=True)['cohens_d']]
    
#     ax3.barh(y_pos_effect, sig_words_plot.sort_values('cohens_d', ascending=True)['cohens_d'], 
#              color=colors_effect, alpha=0.7)
#     ax3.set_yticks(y_pos_effect)
#     ax3.set_yticklabels(words_effect)
#     ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
#     ax3.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
#     ax3.set_title('Effect Size for Significant Words', fontsize=14, fontweight='bold')
#     ax3.grid(axis='x', alpha=0.3)
    
#     # 4. Mismatch rate for significant words
#     ax4 = axes[1, 1]
#     words_mismatch = sig_words_plot.sort_values('mismatch_rate', ascending=True)['word'].values
#     y_pos_mismatch = np.arange(len(words_mismatch))
    
#     ax4.barh(y_pos_mismatch, sig_words_plot.sort_values('mismatch_rate', ascending=True)['mismatch_rate'], 
#              color='coral', alpha=0.7)
#     ax4.set_yticks(y_pos_mismatch)
#     ax4.set_yticklabels(words_mismatch)
#     ax4.set_xlabel('Mismatch Rate (Human ≠ AI)', fontsize=12)
#     ax4.set_title('Classification Disagreement Rate\n(FDR p<0.05 words)', fontsize=14, fontweight='bold')
#     ax4.set_xlim([0, 1])
#     ax4.grid(axis='x', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('../plots/word_significance_analysis.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"\nVisualized {len(sig_words_plot)} most significant words (out of {len(sig_words)} total significant)")
# else:
#     print("No statistically significant words to visualize at FDR p<0.05 threshold.")
#     print("Consider using a less stringent threshold or checking if sample sizes are sufficient.")
# # %%
# # Summary: Words AI consistently over/underscores
# import pandas as pd

# print("="*100)
# print("WORD-LEVEL ANALYSIS SUMMARY")
# print("="*100)

# # 1. Words AI significantly UNDERSCORES (human > AI)
# underscore = word_summary_filtered[
#     (word_summary_filtered['significant_fdr_0.05']) & 
#     (word_summary_filtered['mean_diff'] > 0)
# ].sort_values('mean_diff', ascending=False)

# print("\n" + "="*100)
# print("WORDS AI SIGNIFICANTLY UNDERSCORES (Human scores > AI scores, FDR p<0.05)")
# print("="*100)
# if len(underscore) > 0:
#     print(f"\nFound {len(underscore)} words where AI underscores compared to humans:\n")
#     display_cols = ['word', 'n', 'mean_human', 'mean_ai', 'mean_diff', 'mae', 
#                     'mismatch_rate', 'cohens_d', 'p_ttest_fdr']
#     print(underscore[display_cols].head(15).to_string(index=False))
    
#     # Export for manual review
#     underscore[display_cols].to_csv('../data/clean/words_ai_underscores.csv', index=False)
#     print("\n→ Exported to: ../data/clean/words_ai_underscores.csv")
# else:
#     print("No words found where AI significantly underscores.")

# # 2. Words AI significantly OVERSCORES (AI > human)
# overscore = word_summary_filtered[
#     (word_summary_filtered['significant_fdr_0.05']) & 
#     (word_summary_filtered['mean_diff'] < 0)
# ].sort_values('mean_diff', ascending=True)

# print("\n" + "="*100)
# print("WORDS AI SIGNIFICANTLY OVERSCORES (AI scores > Human scores, FDR p<0.05)")
# print("="*100)
# if len(overscore) > 0:
#     print(f"\nFound {len(overscore)} words where AI overscores compared to humans:\n")
#     display_cols = ['word', 'n', 'mean_human', 'mean_ai', 'mean_diff', 'mae', 
#                     'mismatch_rate', 'cohens_d', 'p_ttest_fdr']
#     print(overscore[display_cols].head(15).to_string(index=False))
    
#     # Export for manual review
#     overscore[display_cols].to_csv('../data/clean/words_ai_overscores.csv', index=False)
#     print("\n→ Exported to: ../data/clean/words_ai_overscores.csv")
# else:
#     print("No words found where AI significantly overscores.")

# # 3. Highest mismatch rate (regardless of direction)
# high_mismatch = word_summary_filtered[
#     word_summary_filtered['significant_fdr_0.05']
# ].sort_values('mismatch_rate', ascending=False)

# print("\n" + "="*100)
# print("HIGHEST DISAGREEMENT WORDS (sorted by mismatch rate, FDR p<0.05)")
# print("="*100)
# if len(high_mismatch) > 0:
#     print(f"\nWords with highest human-AI classification disagreement:\n")
#     display_cols = ['word', 'n', 'mismatch_rate', 'mean_human', 'mean_ai', 
#                     'mae', 'cohens_d', 'p_ttest_fdr']
#     print(high_mismatch[display_cols].head(15).to_string(index=False))
    
#     high_mismatch[display_cols].to_csv('../data/clean/words_high_mismatch.csv', index=False)
#     print("\n→ Exported to: ../data/clean/words_high_mismatch.csv")
# else:
#     print("No significant high-mismatch words found.")

# # 4. Overall summary statistics
# print("\n" + "="*100)
# print("OVERALL STATISTICAL SUMMARY")
# print("="*100)
# print(f"\nTotal unique words analyzed: {len(word_summary)}")
# print(f"Words with n>=10: {len(word_summary_filtered)}")
# print(f"Statistically significant (FDR p<0.05): {word_summary_filtered['significant_fdr_0.05'].sum()}")
# print(f"  • AI underscores: {len(underscore)}")
# print(f"  • AI overscores: {len(overscore)}")
# print(f"  • Mean absolute effect size (significant words): {word_summary_filtered[word_summary_filtered['significant_fdr_0.05']]['cohens_d'].abs().mean():.3f}")
# print(f"\nAverage mismatch rate (all words n>=10): {word_summary_filtered['mismatch_rate'].mean():.2%}")
# print(f"Average mismatch rate (significant words): {word_summary_filtered[word_summary_filtered['significant_fdr_0.05']]['mismatch_rate'].mean():.2%}")

# print("\n" + "="*100)