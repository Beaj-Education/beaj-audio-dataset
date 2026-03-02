import json
import numpy as np
import ast
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from jiwer import wer, cer
from Levenshtein import distance as levenshtein_distance
import json
import numpy as np
import ast
from scipy.stats import pearsonr, spearmanr
import ast, json, re, unicodedata


def _parse_dict(x):
    if pd.isna(x) or x is None or x == '':
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)
            except:
                return {}
    return {}


# ...existing code...
def calculate_ai_fluency(row, use_rounded=False):
    """Calculate AI overall fluency from word-level scores in feedback_ai_json.
    
    Args:
        row: DataFrame row
        use_rounded: If True, use normalized_rounded (discrete 0/1/2), 
                     else use normalized_score (continuous 0-2)
    """
    ai_feedback = row.get('feedback_ai_json')
    
    if pd.isna(ai_feedback) or not ai_feedback:
        return 0.0
    
    # Parse if string
    if isinstance(ai_feedback, str):
        try:
            ai_data = json.loads(ai_feedback)
        except:
            try:
                ai_data = ast.literal_eval(ai_feedback)
            except:
                return 0.0
    else:
        ai_data = ai_feedback
    
    # ai_data is dict {word: {'actual_score', 'normalized_score', 'normalized_rounded'}}
    # Calculate TOTAL scores (not average) and normalize at the end
    if isinstance(ai_data, dict) and ai_data:
        total_score = 0
        total_possible = 0
        score_key = 'normalized_rounded' if use_rounded else 'normalized_score'
        
        for word, word_data in ai_data.items():
            if isinstance(word_data, dict) and score_key in word_data:
                score_val = word_data[score_key]
                if score_val is not None:
                    total_score += score_val
                    total_possible += 2  # Each word max is 2 on the 0-2 scale
        
        if total_possible > 0:
            # Normalize: total_score / total_possible gives 0-1 scale
            return total_score / total_possible
    
    return 0.0



def calculate_transcription_metrics(row):
    """Calculate WER, CER, Levenshtein distance for human and AI transcriptions."""
    human_text = str(row['transcription_human']).lower().strip()
    ai_text = str(row['transcription_ai']).lower().strip()
    reference = str(row['question_text']).lower().strip()

    return {
        'wer_human': wer(reference, human_text),
        'wer_ai': wer(reference, ai_text),
        'wer_human_ai': wer(human_text, ai_text),
        'cer_human': cer(reference, human_text),
        'cer_ai': cer(reference, ai_text),
        'levenshtein_human': levenshtein_distance(reference, human_text),
        'levenshtein_ai': levenshtein_distance(reference, ai_text),
        'levenshtein_human_ai': levenshtein_distance(human_text, ai_text),
        'exact_match_human': int(human_text == reference),
        'exact_match_ai': int(ai_text == reference)
    }


# 2. Word-level fluency analysis
def compare_word_fluency(row):
    """
    Compare human word scores with AI fluency scores using new merged column names.
    Uses feedback_human_json and feedback_ai_json which are already normalized dicts.
    """
    try:
        # Parse human scores from feedback_human_json
        human_feedback = row.get('feedback_human_json')
        if pd.isna(human_feedback) or not human_feedback:
            return {
                'pearson_correlation': np.nan,
                'spearman_correlation': np.nan,
                'mae_word_level': np.nan,
                'num_words_compared': 0
            }
        
        # Parse as dict if it's a JSON string
        if isinstance(human_feedback, str):
            try:
                human_scores = json.loads(human_feedback)
            except:
                try:
                    human_scores = ast.literal_eval(human_feedback)
                except:
                    return {
                        'pearson_correlation': np.nan,
                        'spearman_correlation': np.nan,
                        'mae_word_level': np.nan,
                        'num_words_compared': 0
                    }
        else:
            human_scores = human_feedback
        
        # Parse AI scores from feedback_ai_json
        ai_feedback = row.get('feedback_ai_json')
        if pd.isna(ai_feedback) or not ai_feedback:
            return {
                'pearson_correlation': np.nan,
                'spearman_correlation': np.nan,
                'mae_word_level': np.nan,
                'num_words_compared': 0
            }
        
        # Parse as dict if it's a JSON string
        if isinstance(ai_feedback, str):
            try:
                ai_scores_raw = json.loads(ai_feedback)
            except:
                try:
                    ai_scores_raw = ast.literal_eval(ai_feedback)
                except:
                    return {
                        'pearson_correlation': np.nan,
                        'spearman_correlation': np.nan,
                        'mae_word_level': np.nan,
                        'num_words_compared': 0
                    }
        else:
            ai_scores_raw = ai_feedback
        
        # Match words between human and AI scores
        # Normalize all words first to ensure consistent matching (removes punctuation, etc.)
        human_word_scores = []
        ai_word_scores = []
        matched_words = []
        
        # Create normalized lookup for AI scores
        ai_normalized_lookup = {}
        for word, ai_data in ai_scores_raw.items():
            normalized_word = normalize_word(word)
            if normalized_word:  # Skip empty strings
                ai_normalized_lookup[normalized_word] = ai_data
        
        # Match using normalized words
        for word, human_score in human_scores.items():
            normalized_word = normalize_word(word)
            if not normalized_word:  # Skip empty strings
                continue
                
            if normalized_word in ai_normalized_lookup:
                ai_data = ai_normalized_lookup[normalized_word]
                # Use normalized_rounded (0/1/2) for comparison with human scores
                if isinstance(ai_data, dict) and 'normalized_rounded' in ai_data:
                    ai_score = ai_data['normalized_rounded']
                    if ai_score is not None:
                        matched_words.append(normalized_word)
                        human_word_scores.append(float(human_score))
                        ai_word_scores.append(float(ai_score))
        
        if len(human_word_scores) > 1:  # Need at least 2 points for correlation
            # Calculate correlations
            pearson_corr, _ = pearsonr(human_word_scores, ai_word_scores)
            spearman_corr, _ = spearmanr(human_word_scores, ai_word_scores)
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(np.array(human_word_scores) - np.array(ai_word_scores)))
            
            return {
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'mae_word_level': mae,
                'num_words_compared': len(matched_words)
            }
        else:
            return {
                'pearson_correlation': np.nan,
                'spearman_correlation': np.nan,
                'mae_word_level': np.nan,
                'num_words_compared': len(matched_words)
            }
    
    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            'pearson_correlation': np.nan,
            'spearman_correlation': np.nan,
            'mae_word_level': np.nan,
            'num_words_compared': 0
        }

# 3
def compare_sentence_fluency(row):
    """
    Compare overall human score with AI fluency score using new merged column names.
    Human: score_human_total / score_human_max
    AI: Extract overall accuracy from feedback_ai_json (calculated as average of word-level scores)
    """
    try:
        # Human fluency (normalized to 0-1)
        human_total = row.get('score_human_total', 0)
        human_max = row.get('score_human_max', 1)
        human_normalized = human_total / human_max if human_max and human_max > 0 else 0

        # AI overall fluency - calculate from word-level scores in feedback_ai_json
        ai_normalized = 0.0
        ai_feedback = row.get('feedback_ai_json')
        
        if not pd.isna(ai_feedback) and ai_feedback:
            # Parse if string
            if isinstance(ai_feedback, str):
                try:
                    ai_data = json.loads(ai_feedback)
                except:
                    try:
                        ai_data = ast.literal_eval(ai_feedback)
                    except:
                        ai_data = None
            else:
                ai_data = ai_feedback
            
            # ai_data is dict {word: {'actual_score', 'normalized_score', 'normalized_rounded'}}
            # Calculate average of normalized_score (0-2 scale) and convert to 0-1
            if isinstance(ai_data, dict) and ai_data:
                scores = []
                for word, word_data in ai_data.items():
                    if isinstance(word_data, dict) and 'normalized_score' in word_data:
                        scores.append(word_data['normalized_score'])
                
                if scores:
                    # Average normalized score (0-2 scale), convert to 0-1
                    ai_normalized = np.mean(scores) / 2.0
        
        # Clamp to [0,1]
        ai_normalized = max(0.0, min(1.0, ai_normalized))
        human_normalized = max(0.0, min(1.0, human_normalized))

        return {
            'human_fluency_normalized': human_normalized,
            'ai_fluency_normalized': ai_normalized,
            'fluency_difference': abs(human_normalized - ai_normalized)
        }

    except Exception as e:
        print(f"Error in sentence fluency: {e}")
        return {
            'human_fluency_normalized': np.nan,
            'ai_fluency_normalized': np.nan,
            'fluency_difference': np.nan
        }

def compute_utterance_duration_seconds(val):
    """
    Calculate utterance duration from AI feedback words list.
    Formula: (max(Offset + Duration) - min(Offset)) / 10_000_000
    """
    try:
        if pd.isna(val):
            return 0.0
        try:
            data = ast.literal_eval(val) if isinstance(val, str) else val
        except Exception:
            try:
                data = json.loads(val)
            except Exception:
                return 0.0
        
        fb = None
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            fb = data[0]
        elif isinstance(data, dict):
            fb = data
        
        if not isinstance(fb, dict):
            return 0.0
        
        # Get words list
        words_list = None

        if 'Words' in fb and isinstance(fb['Words'], list):
            words_list = fb['Words']
        elif 'words' in fb and isinstance(fb['words'], list):
            words_list = fb['words']
        
        if not words_list:
            return 0.0
        
        offsets = []
        end_times = []
        
        for wobj in words_list:
            if not isinstance(wobj, dict):
                continue
            
            offset = wobj.get('Offset') or wobj.get('offset')
            duration = wobj.get('Duration') or wobj.get('duration')
            
            if offset is not None and duration is not None:
                try:
                    offset_val = float(offset)
                    duration_val = float(duration)
                    offsets.append(offset_val)
                    end_times.append(offset_val + duration_val)
                except Exception:
                    continue
        
        if not offsets or not end_times:
            return 0.0
        
        utterance_start = min(offsets)
        utterance_end = max(end_times)
        utterance_duration_seconds = (utterance_end - utterance_start) / 10_000_000.0
        
        return utterance_duration_seconds
        
    except Exception as e:
        print(f"[compute_utterance_duration_seconds] Unexpected error: {e}", flush=True)
        return 0.0

def normalize_word(w):
    """Canonicalize a token for matching:
    - NFKC unicode normalization
    - lowercasing
    - replace smart quotes/apostrophes with ascii equivalents
    - strip surrounding quotes and common leading/trailing punctuation
    - collapse internal whitespace
    """
    if w is None:
        return ''
    s = str(w)
    # normalize unicode forms
    s = unicodedata.normalize('NFKC', s)
    # replace smart quotes/apostrophes with ASCII
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.strip()
    s = s.lower()
    # remove surrounding quotes
    s = s.strip("'\"")
    # strip common punctuation at ends
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s

def parse_human_cell(val):
    if pd.isna(val):
        return {}
    try:
        data = ast.literal_eval(val) if isinstance(val, str) else val
    except Exception:
        try:
            data = json.loads(val)
        except Exception:
            return {}
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            w = normalize_word(k)
            try:
                out[w] = int(v)
            except Exception:
                try:
                    out[w] = float(v)
                except Exception:
                    out[w] = v
    return out

# Parse AI feedback into {word: {'actual_score': <0-100>, 'normalized_score': <0-2>}}
def parse_ai_cell(val):
    if pd.isna(val):
        return {}
    try:
        data = ast.literal_eval(val) if isinstance(val, str) else val
    except Exception:
        try:
            data = json.loads(val)
        except Exception:
            return {}
    fb = None
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        fb = data[0]
    elif isinstance(data, dict):
        fb = data
    out = {}
    
    # Preferred path: explicit 'words' list
    if isinstance(fb, dict) and 'words' in fb:
        for wobj in fb['words']:
            try:
                if not isinstance(wobj, dict):
                    continue
                word_raw = wobj.get('Word', '')
                word = normalize_word(word_raw)
                pa = wobj.get('PronunciationAssessment', {}) or {}
                # try multiple possible keys for accuracy
                acc = pa.get('AccuracyScore', None)
                if acc is None:
                    acc = pa.get('accuracyScore', None)
                if acc is None:
                    # some formats embed scores differently
                    acc = wobj.get('AccuracyScore', None) or wobj.get('score', None)
                if acc is None:
                    continue
                try:
                    actual = float(acc)
                except Exception:
                    continue
                normalized = (actual / 100.0) * 2.0
                # Rounded discrete bin on 0/1/2 scale for easier comparison with human scores
                try:
                    normalized_rounded = int(round(normalized))
                except Exception:
                    normalized_rounded = None
                # clamp to [0,2] if present
                if normalized_rounded is not None:
                    normalized_rounded = max(0, min(2, normalized_rounded))
                out[word] = {'actual_score': actual, 'normalized_score': normalized, 'normalized_rounded': normalized_rounded}
            except Exception:
                continue
    # Fallback: feedback object where top-level keys are words
    elif isinstance(fb, dict):
        # decide if keys look like words (not metadata keys)
        keys = [k for k in fb.keys() if isinstance(k, str)]
        # if keys look like content words, attempt to parse them
        if keys:
            for k in keys:
                v = fb.get(k)
                word = normalize_word(k)
                actual = None
                # If value is a dict, look for accuracy keys inside
                if isinstance(v, dict):
                    pa = v.get('PronunciationAssessment', {}) or v.get('pronunciationAssessment', {}) or v.get('pa', {}) or {}
                    acc = None
                    if isinstance(pa, dict):
                        acc = pa.get('AccuracyScore') or pa.get('accuracyScore')
                    if acc is None:
                        acc = v.get('AccuracyScore') or v.get('accuracyScore') or v.get('score') or v.get('actual_score')
                    if acc is not None:
                        try:
                            actual = float(acc)
                        except Exception:
                            actual = None
                else:
                    # value might be numeric (score) or string numeric
                    try:
                        actual = float(v)
                    except Exception:
                        actual = None
                if actual is None:
                    # unable to get an accuracy score for this token, skip
                    continue
                normalized = (actual / 100.0) * 2.0
                try:
                    normalized_rounded = int(round(normalized))
                except Exception:
                    normalized_rounded = None
                if normalized_rounded is not None:
                    normalized_rounded = max(0, min(2, normalized_rounded))
                out[word] = {'actual_score': actual, 'normalized_score': normalized, 'normalized_rounded': normalized_rounded}
    return out


def calculate_words_correct_per_minute(df, score_threshold='partial', group_by_col='student_profile_id'):
    """
    Calculate words correct per minute for each student.
    
    This function aggregates all audio submissions for each student and computes:
    - Words correct per minute (WCPM) where correct = score >= threshold
    - Total words attempted
    - Total duration in minutes
    - Both for human and AI scores
    
    Args:
        df: DataFrame with columns:
            - student_profile_id (or group_by_col): student identifier
            - feedback_human_json: dict/string with {word: score} where score is 0/1/2
            - feedback_ai_json: dict/string with {word: {normalized_rounded: score}}
            - ai_utterance_duration_seconds: duration in seconds
        score_threshold: str, either 'partial' (score >= 1) or 'perfect' (score == 2)
            - 'partial': correct = 1 or 2 (partial or fluent)
            - 'perfect': correct = 2 only (fluent only)
        group_by_col: str, column name to group by (default: 'student_profile_id')
    
    Returns:
        DataFrame with columns:
            - student_profile_id (or group_by_col)
            - human_wcpm_partial: words correct per minute (human, score >= 1)
            - human_wcpm_perfect: words correct per minute (human, score == 2)
            - ai_wcpm_partial: words correct per minute (AI, score >= 1)
            - ai_wcpm_perfect: words correct per minute (AI, score == 2)
            - human_total_words: total words scored by human
            - ai_total_words: total words scored by AI
            - total_duration_seconds: total audio duration
            - total_duration_minutes: total audio duration in minutes
            - num_submissions: number of audio submissions
    """
    
    def _parse_dict(x):
        """Parse JSON/dict string to dict."""
        if pd.isna(x) or x is None or x == '':
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                try:
                    return ast.literal_eval(x)
                except:
                    return {}
        return {}
    
    # Aggregate by student
    results = []
    
    for student_id, group in df.groupby(group_by_col):
        # Initialize counters
        human_words_partial = 0  # score >= 1
        human_words_perfect = 0  # score == 2
        ai_words_partial = 0
        ai_words_perfect = 0
        human_total_words = 0
        ai_total_words = 0
        total_duration = 0
        num_submissions = 0
        
        for _, row in group.iterrows():
            # Get duration
            duration = row.get('ai_utterance_duration_seconds', 0)
            if pd.notna(duration):
                total_duration += float(duration)
            
            num_submissions += 1
            
            # Parse human feedback
            human_feedback = _parse_dict(row.get('feedback_human_json'))
            for word, score in human_feedback.items():
                normalized_word = normalize_word(word)
                if not normalized_word:  # Skip empty strings
                    continue
                try:
                    score_val = float(score)
                    human_total_words += 1
                    if score_val >= 1:
                        human_words_partial += 1
                    if score_val == 2:
                        human_words_perfect += 1
                except:
                    continue
            
            # Parse AI feedback
            ai_feedback = _parse_dict(row.get('feedback_ai_json'))
            for word, word_data in ai_feedback.items():
                normalized_word = normalize_word(word)
                if not normalized_word:  # Skip empty strings
                    continue
                    
                if not isinstance(word_data, dict):
                    continue
                
                ai_score = word_data.get('normalized_rounded')
                if ai_score is None:
                    continue
                
                try:
                    score_val = float(ai_score)
                    ai_total_words += 1
                    if score_val >= 1:
                        ai_words_partial += 1
                    if score_val == 2:
                        ai_words_perfect += 1
                except:
                    continue
        
        # Calculate per-minute rates
        total_duration_minutes = total_duration / 60.0 if total_duration > 0 else 0
        
        human_wcpm_partial = human_words_partial / total_duration_minutes if total_duration_minutes > 0 else 0
        human_wcpm_perfect = human_words_perfect / total_duration_minutes if total_duration_minutes > 0 else 0
        ai_wcpm_partial = ai_words_partial / total_duration_minutes if total_duration_minutes > 0 else 0
        ai_wcpm_perfect = ai_words_perfect / total_duration_minutes if total_duration_minutes > 0 else 0
        
        results.append({
            group_by_col: student_id,
            'human_wcpm_partial': human_wcpm_partial,
            'human_wcpm_perfect': human_wcpm_perfect,
            'ai_wcpm_partial': ai_wcpm_partial,
            'ai_wcpm_perfect': ai_wcpm_perfect,
            'human_total_words': human_total_words,
            'ai_total_words': ai_total_words,
            'human_words_correct_partial': human_words_partial,
            'human_words_correct_perfect': human_words_perfect,
            'ai_words_correct_partial': ai_words_partial,
            'ai_words_correct_perfect': ai_words_perfect,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration_minutes,
            'num_submissions': num_submissions,
        })
    
    return pd.DataFrame(results)
