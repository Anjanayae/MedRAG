"""
app.py — Streamlit chat frontend for MedRAG. Talks to the FastAPI backend
over HTTP (src/api.py) rather than importing the pipeline directly.

Why a separate process instead of importing hybrid_retriever/generation
directly into this file: keeping API and UI as genuinely separate services
(not just separate files sharing one process) is the more realistic,
production-shaped architecture — the API can be swapped, scaled, or used by
another client (curl, a mobile app, etc.) independent of this UI. It also
means Streamlit's own process doesn't need to hold the embedding model,
Chroma index, and BM25 index in memory alongside the UI.

Run with:  streamlit run app.py
(requires `python src/api.py` running separately on localhost:8000)
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MedRAG", page_icon="🩺", layout="centered")

st.title("🩺 MedRAG — Medical Q&A Assistant")
st.caption(
    "Hybrid retrieval (dense + BM25) + cross-encoder reranking over MedQuAD "
    "(16k+ NIH/CDC medical Q&A pairs). Not a substitute for professional "
    "medical advice."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Sources to retrieve (top_k)", min_value=1, max_value=10, value=5)
    use_reranker = st.checkbox("Use reranker", value=True)
    st.divider()
    st.caption(
        "Backend: FastAPI + Chroma + BM25 + cross-encoder reranker + Groq "
        "(llama-3.3-70b-versatile)."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("retriever_loaded", False)
    except requests.exceptions.RequestException:
        return False


if not check_api_health():
    st.error(
        "Can't reach the MedRAG API. Make sure it's running:\n\n"
        "```\npython src/api.py\n```"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])}) · "
                              f"confidence {message['confidence']:.2f} · "
                              f"{message['retrieval_time_ms']:.0f}ms retrieval + "
                              f"{message['generation_time_ms']:.0f}ms generation"):
                for src in message["sources"]:
                    st.markdown(
                        f"**[{src['index']}] {src['focus']}** "
                        f"({src['source']}, score {src['score']:.3f})  \n"
                        f"*Original question: {src['question']}*  \n"
                        f"[Source link]({src['url']})" if src.get("url") else ""
                    )

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "top_k": top_k,
                        "use_reranker": use_reranker,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request to API failed: {e}")
                st.stop()

        st.markdown(data["answer"])
        if data["refused"]:
            st.info("⚠️ Low confidence — refused to answer rather than guess.")

        if data["sources"]:
            with st.expander(f"Sources ({len(data['sources'])}) · "
                              f"confidence {data['confidence']:.2f} · "
                              f"{data['retrieval_time_ms']:.0f}ms retrieval + "
                              f"{data['generation_time_ms']:.0f}ms generation"):
                for src in data["sources"]:
                    url_line = f"[Source link]({src['url']})" if src.get("url") else ""
                    st.markdown(
                        f"**[{src['index']}] {src['focus']}** "
                        f"({src['source']}, score {src['score']:.3f})  \n"
                        f"*Original question: {src['question']}*  \n"
                        f"{url_line}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "sources": data["sources"],
            "confidence": data["confidence"],
            "retrieval_time_ms": data["retrieval_time_ms"],
            "generation_time_ms": data["generation_time_ms"],
        }
    )