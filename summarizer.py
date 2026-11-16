"""LangChain map-reduce summarization powered by Google Gemini."""

import os
import re

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

load_dotenv()

DEFAULT_MODEL = "gemini-3.1-flash-lite"


class PDFSummarizer:
    def __init__(self, model: str | None = None, max_concurrency: int = 4):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_concurrency = max_concurrency
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.3,
            google_api_key=api_key,
            max_retries=2,
        )
        # Gemini 3 content is a list of blocks; .text joins them, str() drops the TextAccessor type
        parser = RunnableLambda(lambda m: str(m.text))
        self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
        self.combine_chain = COMBINE | self.llm | parser
        self.analysis_chain = ANALYSIS | self.llm | parser
        self.quotes_chain = QUOTES | self.llm | parser

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        style = STYLES.get(summary_type, STYLES["detailed"])
        return self.chunk_chain.invoke({"style": style, "text": text})

    def summarize_chunks(self, chunks: list[str], summary_type: str = "detailed") -> dict:
        style = STYLES.get(summary_type, STYLES["detailed"])
        inputs = [{"style": style, "text": c} for c in chunks]
        results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        summaries = []
        for i, (chunk, res) in enumerate(zip(chunks, results), start=1):
            ok = not isinstance(res, Exception)
            summary = res if ok else f"Error processing chunk: {res}"
            summaries.append(
                {
                    "chunk_number": i,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        valid = [cs["summary"] for cs in chunk_summaries if cs.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(f"Section {i}: {s}" for i, s in enumerate(valid, start=1))
        style = STYLES.get(summary_type, STYLES["detailed"])
        try:
            return self.combine_chain.invoke({"style": style, "summaries": sections})
        except Exception:  # noqa: BLE001
            return f"Combined Summary:\n\n{sections}"

    def analyze_document_structure(self, text: str) -> dict:
        try:
            analysis = self.analysis_chain.invoke({"text": text[:3000]})
            return {"analysis": analysis, "status": "success"}
        except Exception as e:  # noqa: BLE001
            return {"analysis": f"Error analyzing document: {e}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        response = self.quotes_chain.invoke({"text": text})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s*", "", ln).strip(' "\u201c\u201d')
            for ln in response.splitlines()
        )
        return [ln for ln in lines if ln][:10]
