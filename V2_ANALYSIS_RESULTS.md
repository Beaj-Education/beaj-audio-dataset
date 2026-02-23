# V2 Analysis Results
*Generated: 2026-02-20 | Script: `scripts/v2_analysis.py`*

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows across all graders | 1,260 |
| Rows with transcriptions | 770 |
| Unique audio files transcribed | 558 |
| Human graders | 7 (Abdullah, Ali, Amna, Dania, Rehma, Salman, Semal) |
| AI responses loaded | 4,128 rows across 105 profiles |
| AI match rate | 757 / 770 (98.3%) |

### Transcriptions per Grader

| Grader | Transcriptions | Mean Words Correct | SD |
|--------|---------------|-------------------|----|
| Abdullah | 18 | 6.44 | 3.93 |
| Ali | 69 | 8.30 | 3.99 |
| Amna | 130 | 8.06 | 4.15 |
| Dania | 125 | 7.28 | 4.87 |
| Rehma | 169 | 7.03 | 3.60 |
| Salman | 70 | 8.90 | 4.04 |
| Semal | 189 | 7.24 | 3.94 |

### Reference Word Counts

Three question types were assessed, with reference lengths of **6**, **7**, and **15** words:

| Reference Length | Count |
|-----------------|-------|
| 6 words | 264 |
| 7 words | 252 |
| 15 words | 254 |

---

## Score Distribution

Scores represent word-match counts (number of reference words correctly reproduced).

| Statistic | Human | AI |
|-----------|-------|----|
| Mean | 7.57 | 5.42 |
| SD | 4.12 | 2.30 |
| Min | 0 | 0 |
| 25th pct | 5 | 4 |
| Median | 6 | 5 |
| 75th pct | 11 | 7 |
| Max | 15 | 11 |

---

## AI vs Human Score Comparison

| Metric | Value |
|--------|-------|
| Pearson r (human vs AI) | 0.673 (p < 0.0001, n=770) |
| Mean difference (AI − Human) | −2.14 points |
| SD of difference | 3.08 |
| Mean absolute difference | 2.61 points |
| Bias direction | AI scores **lower** than humans |

---

## Pre vs Post Comparison

| Period | Human Mean | AI Mean |
|--------|-----------|---------|
| Pre | 7.33 | 4.97 |
| Post | 7.80 | 5.86 |
| Change | +0.47 | +0.89 |

Both human and AI scores show improvement from pre to post assessments.

---

## Pair-wise Pearson Correlations

Computed on shared audio files only. NaN = fewer than 3 shared audio files (insufficient for meaningful correlation). Diagonal = self-correlation (1.0).

|          | Abdullah | Ali   | Amna  | Dania | Rehma | Salman | Semal | AI    |
|----------|----------|-------|-------|-------|-------|--------|-------|-------|
| **Abdullah** | 1.000 | — | 1.000 | — | — | — | 0.756 | 0.893 |
| **Ali** | — | 1.000 | 0.998 | 0.905 | 0.987 | — | 0.986 | 0.712 |
| **Amna** | 1.000 | 0.998 | 1.000 | 0.580 | 0.971 | 0.982 | 0.905 | 0.751 |
| **Dania** | — | 0.905 | 0.580 | 1.000 | 0.899 | 0.986 | 0.833 | 0.606 |
| **Rehma** | — | 0.987 | 0.971 | 0.899 | 1.000 | 0.794 | 0.947 | 0.656 |
| **Salman** | — | — | 0.982 | 0.986 | 0.794 | 1.000 | 0.971 | 0.660 |
| **Semal** | 0.756 | 0.986 | 0.905 | 0.833 | 0.947 | 0.971 | 1.000 | 0.677 |
| **AI** | 0.893 | 0.712 | 0.751 | 0.606 | 0.656 | 0.660 | 0.677 | 1.000 |

### Shared Audio File Overlap (n per pair)

|          | Abdullah | Ali | Amna | Dania | Rehma | Salman | Semal | AI  |
|----------|----------|-----|------|-------|-------|--------|-------|-----|
| **Abdullah** | 18 | 0 | 3 | 2 | 2 | 0 | 3 | 18 |
| **Ali** | 0 | 69 | 10 | 10 | 9 | 2 | 8 | 69 |
| **Amna** | 3 | 10 | 130 | 13 | 20 | 8 | 24 | 130 |
| **Dania** | 2 | 10 | 13 | 125 | 19 | 9 | 22 | 125 |
| **Rehma** | 2 | 9 | 20 | 19 | 169 | 11 | 27 | 169 |
| **Salman** | 0 | 2 | 8 | 9 | 11 | 70 | 10 | 70 |
| **Semal** | 3 | 8 | 24 | 22 | 27 | 10 | 189 | 189 |
| **AI** | 18 | 69 | 130 | 125 | 169 | 70 | 189 | 558 |

---

## Key Findings

1. **Human inter-rater agreement is generally high** — most human-human pairs correlate at r > 0.90, suggesting graders apply similar standards. Notable exception: Amna vs Dania (r = 0.58, n=13), though the small overlap warrants caution.

2. **AI scores systematically lower than humans** — AI averages 2.14 points below human scores (SD 3.08), with a moderate overall correlation of r = 0.673. The AI also has a narrower score range (max 11 vs 15 for humans).

3. **Human-AI correlations are weaker than human-human** — all human-AI pairs fall in the 0.61–0.89 range, compared to most human-human pairs at 0.90+. This suggests the AI and human graders use somewhat different criteria despite correlating positively.

4. **Pre/post improvement detected** — both human (+0.47) and AI (+0.89) scores are higher in post-assessments, consistent with learning gains.

5. **Limited overlap between many grader pairs** — several pairs (e.g., Abdullah-Ali, Abdullah-Dania) share zero or very few audio files, making those correlations unreliable or uncomputable. Expanding shared audio assignments would strengthen future reliability estimates.

---

---

## Words Correct Per Minute (WCPM)

WCPM is computed from `data/clean/wcpm_by_student.csv` (89 students; 88 with valid human data). It reflects total words correctly scored divided by total audio duration in minutes, aggregated per student across all submissions. Per-question timing is not available.

| Statistic | Human WCPM | AI WCPM | Difference (Human − AI) |
|-----------|-----------|---------|------------------------|
| Mean | 90.8 | 79.5 | 11.3 |
| Median | 89.5 | 71.1 | 12.5 |
| SD | 42.4 | 39.8 | 19.6 |
| Min | 0.0 | 19.5 | −113.2 |
| Max | 277.8 | 240.0 | 48.4 |

On average, human-scored WCPM is **11.3 points higher** than AI-scored WCPM, consistent with the word-level scoring gap observed in the main analysis. The large SD in the difference (19.6) reflects student-level variability: for most students human WCPM exceeds AI WCPM, but in a small number of cases AI rates students higher (negative differences).

### WCPM by Pre / Post

Computed by joining v2 word scores with per-question audio durations from the v1 dataset (322 / 558 audio files matched; unmatched files lack duration data in the v1 records).

| Period | Human WCPM (mean) | Human WCPM (median) | AI WCPM (mean) | AI WCPM (median) | n |
|--------|-------------------|---------------------|----------------|------------------|---|
| Pre | 73.9 | 69.1 | 58.4 | 54.1 | 164 |
| Post | 109.2 | 100.6 | 90.4 | 85.6 | 158 |
| **Change (Δ)** | **+35.3** | | **+32.0** | | |

Both human and AI WCPM increase substantially from pre to post (~48% and ~55% respectively), consistent with genuine learning gains across the assessment period. The human-AI gap persists at roughly the same magnitude in both periods (pre: ~15.5 WCPM, post: ~18.8 WCPM), suggesting the AI underscoring is not specific to one assessment stage.

---

## Word-Level Disagreement (AI vs Human)

Sourced from `data/clean/word_level_summary.csv`. Each row is one word from the assessment passages; scores are on a 0–2 scale. *Mismatch rate* is the proportion of instances where AI and human scores differ.

| Word | n | Mean Human | Mean AI | Mean Diff (H−AI) | Mismatch Rate |
|------|---|-----------|---------|-----------------|---------------|
| faiz | 48 | 1.75 | 1.00 | +0.75 | 83.3% |
| together | 47 | 1.77 | 1.04 | +0.72 | 63.8% |
| locked | 53 | 1.38 | 1.17 | +0.21 | 58.5% |
| we | 51 | 1.92 | 1.53 | +0.39 | 45.1% |
| on | 51 | 1.92 | 1.57 | +0.35 | 45.1% |
| come | 47 | 1.98 | 1.34 | +0.64 | 44.7% |
| zara | 56 | 1.93 | 1.50 | +0.43 | 44.6% |
| build | 49 | 1.86 | 1.22 | +0.63 | 42.9% |
| things | 49 | 1.84 | 1.39 | +0.45 | 38.8% |
| draw | 54 | 1.85 | 1.63 | +0.22 | 33.3% |
| loves | 57 | 1.70 | 1.86 | **−0.16** | 28.1% |

**Notable patterns:**
- **South Asian proper nouns dominate the top of the list.** "Faiz" (a common South Asian name) has the highest mismatch rate (83%) and the largest mean score gap (+0.75). "Zara" ranks 7th (44.6% mismatch, +0.43 gap). These are words an AI system trained primarily on Western English corpora is least likely to recognize reliably.
- **Multisyllabic and contextually complex words** also show high mismatch: "together" (63.8%), "locked" (58.5%), "build" (42.9%), "things" (38.8%).
- **"Loves" is the one word where AI scores higher than humans** (mean diff −0.16), suggesting AI can over-credit phonetically simpler words in some contexts.

---

## Motivation for Calibrating AI Scoring to Local Standards

The data provides converging evidence that off-the-shelf AI scoring tools require calibration before deployment in South Asian educational contexts:

**1. The bias is systematic, not random — making it correctable.**
AI scores are consistently 2.14 points lower than human scores (SD = 3.08) at the word-count level, and 11.3 WCPM lower in fluency estimates. A consistent downward offset is the hallmark of a calibration problem rather than random noise. A simple linear recalibration (or per-word adjustment) could substantially close this gap.

**2. South Asian proper nouns reveal a specific cultural failure mode.**
The two highest-mismatch words — "faiz" and "zara" — are South Asian names embedded in the assessment passages. These names are phonologically and lexically underrepresented in the datasets used to train general-purpose AI speech and scoring models, which skew toward American and British English. The result is systematic score suppression for culturally-specific vocabulary that human graders handle with no difficulty.

**3. The gap is non-uniform across words, so flat scaling is insufficient.**
AI severely underscores "together" (+0.72) and "come" (+0.64), yet slightly overscores "loves" (−0.16). A single scaling factor applied uniformly would over-correct some words while under-correcting others. Effective calibration likely requires word-level or phoneme-class-level adjustments anchored to local human grader data.

**4. High human-AI correlation shows the AI understands the task — but not the context.**
The overall Pearson r of 0.673 between human and AI scores means the AI's relative ordering of students is meaningful. The problem is not that the AI is scoring randomly; it is applying a scoring standard miscalibrated to this population. This is encouraging: it suggests that targeted fine-tuning or post-hoc calibration on a modest local dataset could bring AI scores into alignment without retraining from scratch.

**5. Deployment risk without calibration.**
AI reading assessments deployed at scale in Pakistan (or similar South Asian contexts) without local calibration would systematically underestimate student fluency. This could misdirect remediation resources, under-identify students who are performing adequately, and erode trust in AI-assisted grading among educators. The ≈11 WCPM gap in this dataset translates to a meaningful difference in fluency classification thresholds used in early literacy programs.

---

## Output Files

| File | Description |
|------|-------------|
| `data/clean/v2_merged_scores.csv` | Merged dataset (human + AI scores per transcription) |
| `plots/v2_pairwise_correlation_heatmap.png` | Pair-wise correlation heatmap |
