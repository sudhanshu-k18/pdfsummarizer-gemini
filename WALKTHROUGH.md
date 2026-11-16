# Walkthrough

How this project was built, phase by phase, and how the pieces fit together.

## Phase 0 - Validate access before writing code

Everything external is proven working first:

```powershell
.venv\Scripts\python.exe scripts\check_environment.py
```

The script checks the Python version, `.env`, installed packages, the Gemini key
(`listModels` + a live `generateContent` call), and GitHub CLI authentication.

Why: an invalid key or an unauthenticated `gh` turns into confusing failures later.

## Phase 1 - Scaffold

Create the five application modules plus scripts, tests, `.env.example`, `.gitignore`
and `README.md`. Initialize git. Confirm `.env` and `.venv/` are ignored before the
first commit.

## Phase 2 - Ingestion (`pdf_processor.py`)

`PDFProcessor` turns a PDF into clean, token-bounded chunks:

| Method | Purpose |
| --- | --- |
| `extract_text_from_pdf(file)` | Reads every page, skips pages that fail |
| `extract_text_from_path(path)` | Same, for CLI/tests |
| `clean_text(text)` | Removes control chars, normalizes whitespace |
| `count_tokens(text)` | `tiktoken` `cl100k_base` count (Gemini approximation) |
| `chunk_text(text, max_tokens)` | Sentence-aware packing into chunks |
| `get_pdf_metadata(file)` | Page count, title, author, subject |

Chunking is sentence-aware: split on `.!?`, then pack sentences until the token
budget is reached. A single sentence longer than the limit is hard-split by tokens.

Verified with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\smoke_test.py "C:\path\to.pdf" --no-llm
```

## Phase 3 - Summarization (`prompts.py`, `summarizer.py`)

Prompts are plain `ChatPromptTemplate`s with a shared system persona. Four styles
(`brief`, `detailed`, `bullet_points`, `executive`) are injected through `{style}`.

`PDFSummarizer` builds four LangChain chains:

```
CHUNK_SUMMARY | llm | parser    -> per-chunk summary
COMBINE       | llm | parser    -> merged final summary
ANALYSIS      | llm | parser    -> structure analysis
QUOTES        | llm | parser    -> key quotes
```

`summarize_chunks()` is the map-reduce step:

1. Build one input per chunk.
2. Run them concurrently with `batch(max_concurrency=4, return_exceptions=True)`.
3. Record per-chunk success/failure, lengths and numbers.
4. Merge the successful summaries; if merging fails, fall back to concatenation.

A failed chunk never aborts the run.

## Phase 4 - User interface (`app.py`, `utils.py`)

Streamlit app with:

- Sidebar: model, summary style, chunk size, analysis/quotes toggles
- Live key validation button that performs a real API call (cached for 10 minutes)
- PDF upload, metadata cards, staged progress messages
- Results rendered from `st.session_state` so downloads do not wipe them
- TXT report and CSV statistics downloads

`utils.py` holds presentation/export helpers plus `probe_api_key()`, a dependency-free
REST check used by both the UI and the Phase 0 script.

## Phase 5 - Quality checks

- Unit tests for cleaning, token counting, chunk bounds, empty/scanned PDFs
- End-to-end CLI run on a real PDF (`scripts/smoke_test.py`)
- Streamlit boot test
- Manual matrix: 1-page, multi-page, scanned, all four styles

## Phase 6 - Publish

```powershell
gh repo create pdfsummarizer-gemini --public --source=. --remote=origin --push
```

Then verify a fresh clone installs and runs with only `.env.example` present.

## Data flow

```
PDF -> extract -> clean -> count -> chunk
                                    |
                    map: chunk summaries (concurrent)
                                    |
                    reduce: combine summaries
                                    |
                    analysis + quotes (optional)
                                    |
                    Streamlit display + TXT/CSV export
```
