# MedRAG — Hybrid RAG Medical Q&A Assistant

A retrieval-augmented generation system over the [MedQuAD](https://github.com/abachaa/MedQuAD)
dataset (16k+ medical QA pairs from 9 NIH sources), built to demonstrate hybrid
retrieval, reranking, grounded generation, and RAG evaluation — not just a
"stuff docs into a vector DB" demo.

## Why this exists

Most RAG portfolio projects stop at "embed docs, retrieve top-k, ask an LLM."
This one is built to answer the questions an interviewer actually asks:
- *Why this chunk size?* → ablation study comparing 3 chunking strategies
- *How do you know retrieval works?* → recall@k measured, not assumed
- *How do you know the answer is grounded?* → RAGAS faithfulness scoring
- *What happens when retrieval fails?* → confidence-based refusal
- *Is this production-shaped?* → FastAPI + Streamlit as separate services,
  Docker, CI, tests — not a single notebook

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data

```bash
mkdir -p data/raw && cd data/raw
git clone --depth 1 https://github.com/abachaa/MedQuAD.git
cd ../..
```

MedQuAD ships 12 subsets; 3 (A.D.A.M., MedlinePlus Drugs, MedlinePlus Herbs)
have their answers stripped for copyright reasons — `ingest.py` excludes
those automatically rather than silently keeping empty rows.

onal HF Spaces deploy, resume bullets
