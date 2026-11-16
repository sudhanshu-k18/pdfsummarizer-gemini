"""Display, export, statistics and API-key helpers for the Streamlit app."""

import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st


def format_summary_display(summary_data: dict):
    """Render the summary statistics, final summary and per-chunk summaries."""
    st.subheader("📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data["total_chunks"])
    col2.metric(
        "Successful Summaries",
        sum(s["ok"] for s in summary_data["individual_summaries"]),
    )
    col3.metric("Summary Type", summary_data["summary_type"].replace("_", " ").title())

    st.subheader("📝 Final Summary")
    st.write(summary_data["combined_summary"])

    with st.expander("View Individual Chunk Summaries"):
        for s in summary_data["individual_summaries"]:
            st.write(f"**Chunk {s['chunk_number']}**" + ("" if s["ok"] else " ⚠ failed"))
            st.write(s["summary"])
            st.write(
                f"*Original length: {s['original_length']} chars, "
                f"Summary length: {s['summary_length']} chars*"
            )
            st.divider()


def export_summary_to_text(summary_data: dict, filename: str = "") -> str:
    """Build the plain-text report; filename (if given) appears in the header."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data["combined_summary"],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for s in summary_data["individual_summaries"]:
        lines += [
            f"Chunk {s['chunk_number']}{'' if s['ok'] else ' (FAILED)'}:",
            s["summary"],
            "",
        ]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame(
        [
            {
                "Chunk": s["chunk_number"],
                "Original Length": s["original_length"],
                "Summary Length": s["summary_length"],
                "Compression Ratio": (
                    round(s["summary_length"] / s["original_length"], 2)
                    if s["original_length"]
                    else 0
                ),
                "Status": "Success" if s["ok"] else "Error",
            }
            for s in summary_data["individual_summaries"]
        ]
    )


def estimate_processing_time(num_chunks: int) -> str:
    """Rough estimate: ~3s per chunk with 4 running concurrently, plus 5s to combine."""
    seconds = int(num_chunks * 3 / 4 + 5)
    return f"~{seconds} seconds" if seconds < 60 else f"~{seconds // 60} min {seconds % 60}s"


def validate_api_key() -> bool:
    """Check GEMINI_API_KEY is present and looks plausible; show a quiet caption when it is."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "❌ GEMINI_API_KEY not found! Copy `.env.example` to `.env`, "
            "add your key, and restart."
        )
        st.info("Get a free key at https://aistudio.google.com/apikey")
        return False
    if len(api_key) < 20:
        st.error(
            "❌ GEMINI_API_KEY seems incomplete. "
            "Get a new one at https://aistudio.google.com/apikey"
        )
        return False
    st.sidebar.caption("🔑 Gemini API key loaded")
    return True


def probe_api_key(api_key: str, model: str, timeout: int = 30) -> tuple[bool, str]:
    """Send one tiny prompt to Gemini to prove the key and model actually work."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = json.dumps(
        {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts).strip()
        return True, text or "connected"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body
        return False, f"HTTP {exc.code}: {message}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
