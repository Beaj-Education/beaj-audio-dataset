# %%
"""
Extract AI utterance duration for all v2 audios.

Uses ai_responses_extracted.csv (which has submitted_feedback_json for all 105 profiles)
and maps to audio filenames from main.csv via profile_id + question_number + pre_or_post.

Output: data/clean/v2_audio_durations.csv
  Columns: audio_file_name, ai_utterance_duration_seconds
"""
import os
import re
import sys

import pandas as pd

# Add parent scripts dir so we can import helper_functions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "v1"))

from helper_functions import compute_utterance_duration_seconds

PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

AI_RESPONSES = os.path.join(DATA_DIR, "v1", "clean", "ai_responses_extracted.csv")
MAIN_CSV = os.path.join(DATA_DIR, "v3", "ai_transcribed", "main.csv")
OUT_CSV = os.path.join(DATA_DIR, "v2", "clean", "v2_audio_durations.csv")

# %%
# Load AI responses (all profiles, pre/post only)
df_ai = pd.read_csv(AI_RESPONSES)
df_ai = df_ai[df_ai["pre_or_post"].isin(["pre", "post"])].copy()
print(f"AI responses (pre/post): {len(df_ai)} rows, {df_ai['profile_id'].nunique()} profiles")

# Compute duration from submitted_feedback_json
df_ai["ai_utterance_duration_seconds"] = df_ai["submitted_feedback_json"].apply(
    compute_utterance_duration_seconds
)

# Build match key: profile_id (str) + question_number (int) + pre_or_post
df_ai["match_key"] = (
    df_ai["profile_id"].astype(str) + "_q" +
    df_ai["question_number"].astype(str) + "_" +
    df_ai["pre_or_post"]
)

# Keep only the columns we need; drop duplicates on match_key (take first)
df_ai_dur = df_ai[["match_key", "ai_utterance_duration_seconds"]].copy()
df_ai_dur = df_ai_dur[df_ai_dur["ai_utterance_duration_seconds"] > 0]
df_ai_dur = df_ai_dur.drop_duplicates(subset="match_key", keep="first")
print(f"AI durations (valid, deduplicated): {len(df_ai_dur)}")

# %%
# Load main.csv to get audio filenames
df_main = pd.read_csv(MAIN_CSV)
audio_files = df_main["audio_file_name"].unique()
print(f"Total audio files in main.csv: {len(audio_files)}")

# Parse match key from audio filenames
# Pattern: grade1_profile31203_q2_LP_20251122_100201.mp3
records = []
for fn in audio_files:
    m = re.match(r"grade\d+_profile(\d+)_q(\d+)_(LP|QK)_", fn)
    if m:
        profile_id = m.group(1)
        q_num = m.group(2)
        pre_or_post = "pre" if m.group(3) == "LP" else "post"
        match_key = f"{profile_id}_q{q_num}_{pre_or_post}"
        records.append({"audio_file_name": fn, "match_key": match_key})

df_audio = pd.DataFrame(records)
print(f"Parsed audio filenames: {len(df_audio)}")

# %%
# Join: audio filenames → durations
df_result = df_audio.merge(df_ai_dur, on="match_key", how="left")

matched = df_result["ai_utterance_duration_seconds"].notna().sum()
print(f"Matched with duration: {matched}/{len(df_result)}")

# Drop match_key, keep only the output columns
df_out = df_result[["audio_file_name", "ai_utterance_duration_seconds"]].dropna()
df_out.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df_out)} rows to {OUT_CSV}")
