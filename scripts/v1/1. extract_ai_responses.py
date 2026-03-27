# %%
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Get database credentials from environment variables
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
        
# Create connection string
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# Connect to database
engine = create_engine(connection_string)

# %%
graded_df = pd.read_csv("../../data/v1/clean/graded_combined.csv")

# %%
filt_df = graded_df[['audio_file_name', 'Question', 'Human Transcription', 'profile_id']].drop_duplicates()

# %%
df_list = []
for i, row in filt_df.iterrows():
    profile_id = row['profile_id']
    question_text = row['Question'].replace("'", "''")  # Escape single quotes
    query = f"""
    SELECT war.*, saq.answer, saq.question_number, saq.difficulty_level,
    wum.city, wum.school_name
    FROM public.wa_question_responses war
    LEFT JOIN public.speak_activity_questions saq ON saq.id = war.question_id
    LEFT JOIN wa_users_metadata wum ON wum.profile_id = war.profile_id

    WHERE activity_type IN ('assessmentWatchAndSpeak')
    AND war.profile_id = '{profile_id}'
    """
    df = pd.read_sql(query, engine)
    print("Processed: ", profile_id)

    df_list.append(df)

df_comb = pd.concat(df_list)

# %%
df_comb.to_csv("../../data/v1/graded/ai_responses_raw.csv", index=False)

# %%
