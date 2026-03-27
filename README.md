# audio-dataset

Short: reproducible pipeline for cleaning human-graded CSVs, extracting AI responses, merging human+AI judgments, and running analysis/visualizations.

## Quick start (macOS, zsh)

1) Clone this repo

   git clone <repo-url>
   cd <repo-dir>

2) Install Homebrew (if needed):

   Follow https://docs.brew.sh/Installation

3) Install Python 3.12 — two options

- Option A — Homebrew (installs latest 3.12.x):

   brew install python@3.12
   # ensure brew python is on your PATH (zsh)
   echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   python3.12 --version

- Option B — Recommended if you need exactly Python 3.12.7: use pyenv

   brew install pyenv
   pyenv install 3.12.7
   pyenv local 3.12.7
   python --version

4) Create and activate a virtualenv (zsh)

   python3 -m venv .venv
   source .venv/bin/activate

5) Install dependencies

   pip install -r requirements.txt

6) Verify environment

   python --version
   pip list

## Analysis versions

- **v1** — Initial analysis. 5 human graders scored Grade 1 audio on a 0-2 word-level scale; compared against Azure Pronunciation Assessment AI scores.
- **v2** — Expanded to 8 human graders with dual-grading assignments; inter-rater reliability, AI calibration, and WCPM analysis.
- **v3 (latest)** — Compares human transcriptions against 11 AI transcription models (Scribe, Gemini, Voxtral, GPT-4o, etc.) using WER and words-correct metrics. Includes pricing and WCPM analysis.

## Repository structure

- `requirements.txt` — Python dependencies.

- `scripts/`
  - `v1/` — data cleaning, merging, inter-rater reliability, and report generation (numbered `0.`–`5.`, plus `helper_functions.py`)
  - `v2/` — v2 analysis, word-level comparison, and report scripts (`v2_*.py`)
  - `v3/` — AI transcription WER analysis, duration extraction, and report generation

- `data/`
  - `v1/graded/` — raw v1 grader CSVs; `v1/clean/` — cleaned & derived CSVs
  - `v2/graded/` — v2 grader assignment CSVs; `v2/clean/` — v2 derived CSVs
  - `v3/ai_transcribed/` — AI model transcriptions; `v3/clean/` — v3 derived CSVs

- `plots/` — generated figures, organised into `v1/`, `v2/`, `v3/` subfolders
- `reports/` — Word documents, organised into `v1/`, `v2/`, `v3/` subfolders

> Note: some v3 scripts read data from `data/v1/` and `data/v2/` (e.g. human grader CSVs, AI response metadata). These files live where they were originally created; later versions reference them by path.

## Notes

- If you need precisely Python 3.12.7, use the pyenv option above (pyenv lets you install and pin patch versions).
