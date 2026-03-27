# %%
import pandas as pd
from helper_functions import compute_utterance_duration_seconds, normalize_word, parse_human_cell, parse_ai_cell
import json
import numpy as np

df_ai = pd.read_csv("../../data/v1/clean/ai_responses_extracted.csv").drop_duplicates()

df_h = pd.read_csv("../../data/v1/clean/graded_combined.csv")
df_h.rename(columns={'Human Transcription': 'human_transcription'}, inplace=True)

df_h['question_clean'] = df_h['Question'].str.strip()
df_ai['question_clean'] = df_ai['question_clean'].str.strip()

df_ai['question_clean'] = df_ai['question_clean'].str.replace("Come on. We can do it  together.", "Come on. We can do it together.")
df_h['question_clean'] = df_h['question_clean'].str.replace("Come on. We can do it  together.", "Come on. We can do it together.")

df_h['human_transcription'] = df_h['human_transcription'].str.strip().str.replace(".", "", regex=False).str.replace("?", "", regex=False).str.replace("!", "", regex=False)
df_h = df_h[~df_h['question_clean'].str.contains("It is Faiz")]
tmp = df_h['audio_file_name'].str.extract(r'(?P<date>\d{8})[_-]?(?P<time>\d{6})')

df_h['pre_or_post'] = np.where(df_h['audio_file_name'].str.contains('_LP_'), 'pre', 'post')
df_h['grade'] = df_h['audio_file_name'].str.split('_').str[0].str.replace("grade", "", regex=False)

# ...existing code...

df_merged = df_ai.merge(df_h, 
                        left_on=['profile_id', 'question_clean', 'pre_or_post'], right_on=['profile_id', 'question_clean', 'pre_or_post'], 
                        how='inner')
# %%
# Rename and reorder columns for clarity
# Create a mapping of old names to new user-friendly names
column_mapping = {
    # Core identifiers
    'profile_id': 'student_profile_id',
    'id': 'ai_submission_id',
    'lesson_id': 'lesson_id',
    'question_id': 'question_id',
    'question_number': 'question_number',
    'difficulty_level': 'difficulty_level',

    # Question & transcriptions
    'question_clean': 'question_text',
    'Question': 'question_text_original',
    'human_transcription': 'transcription_human',
    'ai_transcription': 'transcription_ai',

    # Audio & submission metadata
    'audio_file_name': 'audio_filename',
    'Submitted Audio Link': 'audio_link',
    'submission_date': 'submission_date',

    # Human grading
    'grader': 'grader_name',
    'score': 'score_human_total',
    'total_possible_score': 'score_human_max',
    'score_json': 'score_human_per_word_json',
    'Notes': 'grader_notes',

    # AI feedback
    'submitted_feedback_json': 'feedback_ai_json',

    # pre or post
    'pre_or_post': 'pre_or_post'
}
    
df_merged['ai_utterance_duration_seconds'] = df_merged['submitted_feedback_json'].apply(compute_utterance_duration_seconds)
# Rename columns
df_merged.rename(columns=column_mapping, inplace=True)
# %%df
# Create unified per-word feedback columns so analysis notebook can read a single, consistent CSV


# Parse human per-word feedback into {word: score}
human_col_candidates = ['score_human_per_word_json', 'score_json']
ai_col_candidates = ['feedback_ai_json', 'submitted_feedback_json']

# Build the new columns
# Human
human_source = next((c for c in human_col_candidates if c in df_merged.columns), None)
ai_source = next((c for c in ai_col_candidates if c in df_merged.columns), None)

if human_source is not None:
    df_merged['feedback_human_json'] = df_merged[human_source].apply(parse_human_cell)
else:
    df_merged['feedback_human_json'] = [{} for _ in range(len(df_merged))]

if ai_source is not None:
    df_merged['feedback_ai_json'] = df_merged[ai_source].apply(parse_ai_cell)
else:
    df_merged['feedback_ai_json'] = [{} for _ in range(len(df_merged))]

# %%
# Save a CSV for analysis notebook to consume (serialize dicts as JSON strings)
out_path = '../../data/v1/clean/merged_for_analysis.csv'
df_save = df_merged.copy()
for c in ['feedback_human_json', 'feedback_ai_json']:
    df_save[c] = df_save[c].apply(lambda x: json.dumps(x, ensure_ascii=False))

# Define column order (organized by category)
column_order = [
    # Student & question identifiers
    'student_profile_id',
    'question_text',
    'question_number',
    'difficulty_level',
    'pre_or_post',
    
    # Transcriptions (side by side for easy comparison)
    'transcription_human',
    'transcription_ai',
    
    # Human grading & AI feedback (aligned for comparison)
    'score_human_total',
    'score_human_max',
    'feedback_human_json',
    'feedback_ai_json',
    'ai_utterance_duration_seconds',
    
    # Human grading metadata
    'grader_name',
    'grader_notes',
    
    # IDs & metadata
    'ai_submission_id',
    'lesson_id',
    'question_id',
    'submission_date',
    
    # Audio files
    'audio_filename',
    'audio_link',
    'ai_utterance_duration_seconds'
]

# Reorder columns (include any columns not in column_order at the end)
existing_cols = [c for c in column_order if c in df_save.columns]
other_cols = [c for c in df_save.columns if c not in column_order]
df_final = df_save[existing_cols + other_cols].drop(columns=['score_human_per_word_json'], errors='ignore')

print(f"\nRenamed and reordered {len(df_final.columns)} columns")
print("\nColumn order:")
for i, col in enumerate(df_final.columns, 1):
    print(f"{i:2d}. {col}")

out_path = '../../data/v1/clean/merged_for_analysis.csv'
df_final.head()
df_final.to_csv(out_path, index=False)
# %%
