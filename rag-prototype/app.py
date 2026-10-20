"""Streamlit UI for the RAG prototype."""

from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from ingestion import IngestionError, ingest_file
from rag_pipeline import CHAT_MODEL, EMBED_MODEL, RAGError, RAGPipeline

load_dotenv()  # local development: reads .env

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")


def get_api_key() -> str:
    """Streamlit secrets (deployment) first, then environment variable (local)."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:  # no secrets file locally
        key = ""
    return key or os.getenv("OPENAI_API_KEY", "")


def init_state(api_key: str) -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(api_key)
        st.session_state.indexed = {}   # file name -> content hash
        st.session_state.messages = []  # [{"role", "content", "sources"}]


def process_files(uploaded) -> None:
    pipeline: RAGPipeline = st.session_state.pipeline
    new_files = []
    for f in uploaded:
        data = f.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.indexed.get(f.name) == digest:
            continue  # already indexed, unchanged
        new_files.append((f.name, data, digest))

    if not new_files:
        st.info("All selected files are already processed.")
        return

    with st.spinner("Reading, chunking and embedding documents…"):
        for name, data, digest in new_files:
            try:
                chunks = ingest_file(name, data)
                if name in st.session_state.indexed:
                    pipeline.store.remove_source(name)  # file re-uploaded with new content
                n = pipeline.index_chunks(chunks)
                st.session_state.indexed[name] = digest
                st.success(f"✅ {name}: {n} chunks indexed")
            except (IngestionError, RAGError) as exc:
                st.error(f"❌ {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {name}: unexpected error — {exc}")


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            page = f", page {s.page}" if s.page else ""
            st.markdown(f"**[{i}] {s.name}{page}** · relevance {s.score:.2f}")
            st.caption(s.snippet)


def main() -> None:
    st.title("📄 Document Q&A")
    st.caption("Upload PDF or TXT files, then ask questions. Answers come only from your documents.")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No OpenAI API key found. Locally, copy `.env.example` to `.env` and set "
            "`OPENAI_API_KEY`. On Streamlit Community Cloud, add it under App settings → Secrets."
        )
        st.stop()

    init_state(api_key)
    pipeline: RAGPipeline = st.session_state.pipeline

    with st.sidebar:
        st.header("1. Documents")
        uploaded = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        if st.button("Process documents", type="primary", disabled=not uploaded):
            process_files(uploaded)

        if st.session_state.indexed:
            st.markdown("**Indexed files**")
            for name in st.session_state.indexed:
                st.write(f"• {name}")
            st.caption(f"{len(pipeline.store)} chunks in index")

        st.header("2. Controls")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Clear documents & chat"):
            pipeline.store.clear()
            st.session_state.indexed = {}
            st.session_state.messages = []
            st.rerun()
        st.caption(f"Embeddings: `{EMBED_MODEL}` · Answers: `{CHAT_MODEL}`")

    if not st.session_state.indexed:
        st.info("👈 Upload and process at least one document to get started.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources"))

    question = st.chat_input("Ask a question about your documents…", disabled=not st.session_state.indexed)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating answer…"):
                    result = pipeline.answer(question)
                st.markdown(result.text)
                render_sources(result.sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.text, "sources": result.sources}
                )
            except RAGError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")


main()
