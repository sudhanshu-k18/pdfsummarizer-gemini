# 📄 PDF AI Summarizer (LangChain + Gemini)

An end-to-end PDF summarizer built with **Streamlit**, **LangChain** and **Google Gemini**.
Upload a PDF, pick a summary style, and get a summary, a document-structure analysis,
and optional key quotes. Export the result as text or the per-chunk statistics as CSV.

## Features

- Text extraction, cleaning, token counting and sentence-aware chunking
- Map-reduce summarization: summarize each chunk, then combine into one final summary
- Four summary styles: `brief`, `detailed`, `bullet_points`, `executive`
- Optional document analysis and key-quote extraction
- Per-chunk failure isolation with retry support
- Downloads: full report as `.txt`, chunk statistics as `.csv`
- Real API-key validation (presence check in the UI, live probe in Phase 0 tooling)

## Requirements

- Python 3.10+
- A Google Gemini API key from <https://aistudio.google.com/apikey>
- GitHub CLI (only needed to publish the repo)

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY

streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Phase 0: validate the environment

One script checks everything before you rely on the app:

```powershell
.venv\Scripts\python.exe scripts\check_environment.py
```

It verifies:

1. Python version
2. `.env` exists and `GEMINI_API_KEY` is set
3. All required packages are installed
4. The Gemini key is accepted by the API
5. The configured model exists and answers a live prompt
6. GitHub CLI is installed and authenticated

Exit code `0` means every check passed.

## Tests

```powershell
# unit tests (no API key or network needed)
.venv\Scripts\python.exe -m unittest discover -s tests -v

# end-to-end pipeline on a real PDF
.venv\Scripts\python.exe scripts\smoke_test.py "C:\path\to\file.pdf" --style brief

# extraction and chunking only
.venv\Scripts\python.exe scripts\smoke_test.py "C:\path\to\file.pdf" --no-llm
```

## Project structure

```
app.py                       Streamlit UI (settings, upload, results, downloads)
pdf_processor.py             PDF text extraction, cleaning, token counting, chunking
summarizer.py                Gemini calls: chunk summaries, combine, analysis, quotes
prompts.py                   All ChatPromptTemplates and summary styles
utils.py                     Display, export, statistics and API-key helpers
scripts/check_environment.py Phase 0 validation (keys, packages, GitHub login)
scripts/smoke_test.py        End-to-end CLI test on a single PDF
tests/test_pdf_processor.py  Unit tests for the ingestion layer
.env.example                 Environment variable template
```

## How it works

```
PDF -> extract text -> clean -> chunk (by tokens)
                                     |
                 +-------------------+-------------------+
                 v                   v                   v
            summary 1           summary 2   ...     summary N      (map)
                 +-------------------+-------------------+
                                     v
                           combine -> final summary             (reduce)
```

Chunks are summarized concurrently (`max_concurrency=4`). If a chunk fails, the
remaining chunks still produce a combined summary; the failure is reported in the
statistics table instead of breaking the whole run.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | required | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Default model; can also be changed in the sidebar |

Supported styles live in `prompts.py` under `STYLES`. To add one, add an entry to
`STYLES` and add its name to the "Summary Type" options in `app.py`.

## Troubleshooting

- **"No extractable text"**: the PDF is probably scanned images. Run OCR first.
- **429 / rate limit**: wait a minute and retry, use a smaller document, or switch model.
- **Model not available / 404**: set `GEMINI_MODEL` in `.env` to a model your key can access.
- **API key not found**: make sure `.env` sits next to `app.py` and contains `GEMINI_API_KEY`.
- **Key rejected**: run `scripts/check_environment.py`; regenerate the key if needed.
- **`tiktoken` install failure**: use Python 3.12 (`py -3.12 -m venv .venv`) and reinstall.

## Publishing to GitHub

```powershell
gh auth login
gh repo create <repo-name> --public --source=. --remote=origin --push
```

`.env` and `.venv/` are ignored by git, so secrets never leave your machine.
