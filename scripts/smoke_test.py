"""Run the extraction -> chunking -> summarization pipeline on a single PDF.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_test.py path\\to\\file.pdf
    .venv\\Scripts\\python.exe scripts\\smoke_test.py path\\to\\file.pdf --no-llm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from pdf_processor import PDFProcessor  # noqa: E402

load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the PDF summarizer pipeline")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument(
        "--style",
        default="brief",
        choices=["brief", "detailed", "bullet_points", "executive"],
    )
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=2, help="Max chunks to summarize (0 = all)")
    parser.add_argument("--no-llm", action="store_true", help="Only extract and chunk")
    args = parser.parse_args()

    start = time.time()
    processor = PDFProcessor()
    text = processor.clean_text(processor.extract_text_from_path(args.pdf))

    print(f"File:        {args.pdf}")
    print(f"Extracted:   {len(text)} chars, {processor.count_tokens(text)} tokens")
    if not text:
        print("No extractable text. The PDF may be scanned images (OCR required).")
        return 1

    chunks = processor.chunk_text(text, args.max_tokens)
    print(f"Chunks:      {len(chunks)} (max {args.max_tokens} tokens each)")

    if args.no_llm:
        print(f"Done in {time.time() - start:.1f}s (extract-only)")
        return 0

    from summarizer import PDFSummarizer  # noqa: E402

    selected = chunks if args.limit <= 0 else chunks[: args.limit]
    print(f"Summarizing: {len(selected)} chunk(s), style={args.style}")

    summarizer = PDFSummarizer(args.model)
    data = summarizer.summarize_chunks(selected, args.style)

    print("-" * 60)
    print(data["combined_summary"])
    print("-" * 60)
    ok = sum(1 for s in data["individual_summaries"] if s["ok"])
    print(f"Chunks OK:   {ok}/{data['total_chunks']}")
    print(f"Done in {time.time() - start:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
