# %%
import pandas as pd
import ast

df_comb = pd.read_csv("../../data/v1/graded/ai_responses_raw.csv")

def extract_first_element(val):
    """Extract first element from a list or string representation of a list. Always returns a hashable type."""
    if pd.isna(val):
        return None
    # Already a list - recursively extract until we get a non-list
    if isinstance(val, list):
        if len(val) == 0:
            return None
        return extract_first_element(val[0])
    # String representation of a list
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return extract_first_element(parsed)
        return str(parsed) if parsed is not None else None
    except (ValueError, SyntaxError):
        return str(val)

df_comb['question_clean'] = df_comb['answer'].apply(extract_first_element)
df_comb['ai_transcription'] = df_comb['submitted_answer_text'].apply(extract_first_element)


df_comb['submission_datetime_ai'] = df_comb['submission_date'].astype(str).str.split(".").str[0]

# ...existing code...
df_comb['submission_datetime_ai'] = pd.to_datetime(df_comb['submission_datetime_ai'], errors='coerce')

# ---- simple group-by min/max -> assign pre/post/Other ----
# create a normalized submission_datetime column (coerce errors)
df_comb['submission_datetime'] = pd.to_datetime(df_comb['submission_date'].astype(str).str.split(".").str[0], errors='coerce')

# compute min and max per profile_id + question_clean
minmax = df_comb.groupby(['profile_id', 'question_clean'])['submission_datetime'].agg(dt_min='min', dt_max='max').reset_index()

# merge min/max back onto df_comb
df_comb = df_comb.merge(minmax, on=['profile_id', 'question_clean'], how='left')

# label rows: earliest -> pre, latest -> post, others -> Other; single timestamp -> post; missing timestamp -> Other
def _label_pre_post(row):
    sd = row['submission_datetime']
    if pd.isna(sd):
        return 'Other'
    if pd.isna(row['dt_min']) or pd.isna(row['dt_max']):
        return 'Other'
    if row['dt_min'] == row['dt_max']:
        return 'post'
    if sd == row['dt_min']:
        return 'pre'
    if sd == row['dt_max']:
        return 'post'
    return 'Other'

df_comb['pre_or_post'] = df_comb.apply(_label_pre_post, axis=1)

# cleanup helper columns
df_comb.drop(columns=['submission_datetime', 'dt_min', 'dt_max'], inplace=True)

# %%
cols_to_keep = ['id', 'lesson_id', 'question_id', 'profile_id', 'submission_date', 'question_clean', 'ai_transcription',
                'submitted_feedback_json', 'question_number', 'difficulty_level',
                'pre_or_post', 'city', 'school_name']
df_final = df_comb[cols_to_keep]

# %%
df_final.to_csv("../../data/v1/clean/ai_responses_extracted.csv", index=False)
