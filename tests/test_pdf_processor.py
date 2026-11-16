"""Unit tests for the PDF ingestion layer (no API key required)."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfWriter  # noqa: E402

from pdf_processor import PDFProcessor  # noqa: E402


class TestPDFProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = PDFProcessor()

    def test_clean_text_collapses_whitespace(self):
        raw = "Hello\t\tworld  \n\n\n\nNext\x00 line "
        self.assertEqual(self.processor.clean_text(raw), "Hello world\n\nNext line")

    def test_count_tokens(self):
        self.assertGreater(self.processor.count_tokens("Hello world"), 0)
        self.assertEqual(self.processor.count_tokens(""), 0)

    def test_chunk_text_splits_long_input(self):
        text = "This is a sentence. " * 200
        chunks = self.processor.chunk_text(text, max_tokens=50)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(self.processor.count_tokens(chunk), 55)

    def test_chunk_text_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.processor.chunk_text("   ")

    def test_blank_pdf_metadata(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        metadata = self.processor.get_pdf_metadata(buffer)
        self.assertEqual(metadata["num_pages"], 1)
        self.assertEqual(metadata["title"], "Unknown")

    def test_blank_pdf_has_no_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        text = self.processor.extract_text_from_pdf(buffer)
        self.assertEqual(text.strip(), "")
        with self.assertRaises(ValueError):
            self.processor.chunk_text(text)


if __name__ == "__main__":
    unittest.main()
