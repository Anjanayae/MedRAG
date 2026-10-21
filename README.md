# MedRAG

MedRAG is a medical question-answering application built on retrieval-augmented
generation (RAG). It uses the [MedQuAD](https://github.com/abachaa/MedQuAD)
dataset of medical questions and answers from NIH and CDC sources. The system
retrieves relevant source passages, asks a configurable language model to write
an answer using only those passages, and displays the answer with source
citations.

This project is intended for medical information retrieval and RAG evaluation.
It is not a diagnostic tool and does not replace advice from a healthcare
professional.

## What the project does

The data and query pipelines are separated into reproducible stages:

1. Parse the raw MedQuAD XML files into clean JSONL and CSV QA-pair files.
2. Split answers into fixed-size, sentence, or semantic chunks.
3. Embed the chunks into a persistent Chroma vector store and build a BM25
	sparse index.
4. Retrieve with dense search and BM25, combine results with reciprocal-rank
	fusion, and optionally rerank them with a cross-encoder.
5. Generate a grounded answer with inline citations. If retrieval confidence is
	too low, the system refuses to guess.
6. Optionally verify citations and detect contradictions between retrieved
	sources.

The FastAPI backend exposes the pipeline through HTTP. The Streamlit frontend
provides a chat interface with controls for the number of sources, reranking,
citation verification, and contradiction detection. The API also supports
background re-ingestion and hot-swapping the active index.

## Setup

Create an environment and install the dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the MedQuAD data into `data/raw/MedQuAD`:

```powershell
git clone --depth 1 https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD
```

Configure an LLM provider in a `.env` file at the project root. Groq is the
default:

```text
GROQ_API_KEY=your_key_here
```

The generation provider can be changed with `LLM_PROVIDER=groq`, `openai`,
`anthropic`, or `deepseek`, using the corresponding API key. The optional
`openai` and `anthropic` packages are commented in `requirements.txt` and must
be installed when those providers are used.

## Build the index

Run these commands from the repository root. The default application strategy
is `sentence`.

```powershell
python src/ingest.py
python src/chunking.py
python src/index.py --strategy sentence
```

The chunking script writes all three strategy outputs. Replace `sentence` in
the indexing command with `fixed_size` or `semantic` to choose another index.

## Run the application

Start the API in one terminal:

```powershell
python src/api.py
```

Start the Streamlit chat frontend in a second terminal:

```powershell
streamlit run app.py
```

The UI connects to `http://localhost:8000` by default. Set
`MEDRAG_API_URL` when the API is hosted elsewhere. The API provides interactive
documentation at `http://localhost:8000/docs` and includes these main routes:

- `POST /query` retrieves sources and generates an answer.
- `GET /documents` reports what is currently indexed.
- `POST /ingest` starts a background re-ingestion and re-indexing job.
- `GET /ingest/status` reports the job state.
- `GET /health` checks API and retriever readiness.

## Evaluation and tests

Run the automated tests with:

```powershell
pytest
```

The repository also contains evaluation and ablation scripts for comparing
chunking and retrieval strategies, including holdout evaluation of dense versus
hybrid retrieval.


<!-- === Held-out benchmark — strategy=sentence, top_k=5 ===
dev doc_ids: 2964 | heldout doc_ids: 523 | heldout eval items: 36
Dense  — Recall@5: 0.944  MRR: 0.801
Hybrid — Recall@5: 0.972  MRR: 0.912 -->