# Analysis Results

## Inter-Rater Reliability

**Files with multiple human graders:** 13

### Human Agreement
| Metric | Value |
|--------|-------|
| Mean score range (max - min) | 1.54 points |
| Score range SD | 1.15 |

### AI vs Human Comparison
| Metric | Value |
|--------|-------|
| Mean deviation from human avg | -2.77 points |
| Deviation SD | 4.32 |
| Bias direction | AI scores **lower** than humans |

### High Disagreement Cases (word-level diff > 1 point)
4 words across 3 files showed significant human grader disagreement (2-point difference).

---

## Transcription Comparison (AI vs Human)

**Total records:** 344
**Valid comparisons:** 334

### Transcription Similarity
| Metric | Mean | SD |
|--------|------|-----|
| Character similarity | 89.9% | 13.9% |
| Word Error Rate (WER) | 17.6% | 20.8% |
| Exact match rate | 39.8% | - |

### Score Comparison
| Metric | Human | AI |
|--------|-------|-----|
| Mean score | 14.19 | 11.91 |

| Score Difference (AI - Human) | Value |
|-------------------------------|-------|
| Mean | -2.27 |
| SD | 6.09 |
| Mean absolute diff | 4.26 |
| Bias direction | AI scores **lower** than humans |

---

## Key Findings

1. **AI transcription quality is high** - 89.9% character-level similarity with human transcriptions
2. **AI scoring bias** - AI consistently scores ~2-3 points lower than human graders
3. **Human inter-rater agreement is reasonable** - average range of 1.54 points between graders
4. **Exact transcription matches are moderate** - ~40% of transcriptions match exactly after normalization
