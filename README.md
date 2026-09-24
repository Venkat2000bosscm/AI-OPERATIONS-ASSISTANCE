# AI Operations Assistant

AI Operations Assistant is a self-reasoning operational assistant designed for cloud support, runbook lookup, and incident investigation. It combines private knowledge retrieval, evidence grading, query rewriting, and optional web search to provide grounded responses for technical support scenarios.

The system is built around a Self-RAG workflow, where the assistant first checks internal documents and runbooks, evaluates whether the retrieved content is useful, and only falls back to external search when needed.

## Overview

This application helps teams:

- troubleshoot production and cloud incidents
- answer operational questions using internal documentation
- reuse context across follow-up questions in the same session
- upload and index new private knowledge from the UI
- improve answer reliability through retrieval and validation steps

## Architecture

```text
User request
    ↓
LangGraph workflow
    ↓
Contextualize follow-up question
    ↓
Decide whether retrieval is required
    ↓
Search private Pinecone knowledge base
    ↓
Grade retrieved evidence
    ┌───────────────┬────────────────────┐
    │ Relevant      │ Weak / Missing     │
    ▼               ▼                    
Generate answer  Rewrite query         
    │               │                    
    └───────┬───────┴─────────┐          
            ▼                 ▼
      Check support       Web search fallback
            │                 │
            └──────┬──────────┘
                   ▼
             Final answer
```

The workflow emphasizes internal-first retrieval and evidence-based answer generation before using external sources.

## Tech Stack

- LangGraph — orchestration of the retrieval and reasoning workflow
- Groq — LLM hosting layer using a supported Groq model such as `openai/gpt-oss-20b`
- Hugging Face Sentence-Transformers — open-source embeddings
- Pinecone — vector database for private operational knowledge
- Tavily — web-search fallback for missing or weak internal coverage
- SQLite — persistent session memory and local audit storage
- FastAPI — API backend and web app
- HTML, CSS, and JavaScript — user interface for chat and document upload

## Project Structure

```text
AI-OPERATIONS/
├── app.py
├── data_ingestion.py
├── Dockerfile
├── requirements.txt
├── .env
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── ingestion.py
│   ├── models.py
│   ├── self_rag.py
│   └── vectorstore.py
├── data/
│   └── audit.db
├── documents/
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── uploads/
└── .venv/
```

## Key Features

- private runbook-first retrieval
- self-grading of retrieved evidence
- automatic query rewriting for weak searches
- optional external search fallback
- persistent session memory using SQLite
- upload support for new operational documents
- incident-focused answer generation for support and operations use cases

## Prerequisites

Before starting the project, make sure you have:

- Python 3.10 or later
- access to a Groq API key
- a Pinecone API key and configured index
- a Tavily API key for web fallback
- a valid embedding model available for local or hosted vectorization

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with values like:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=cloudops-sentinel-self-rag
PINECONE_NAMESPACE=incident-runbooks
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

TAVILY_API_KEY=your_tavily_key

TOP_K=5
MAX_SUPPORT_RETRIES=2
MAX_RETRIEVAL_REWRITES=2
MAX_WEB_REWRITES=2
DATABASE_PATH=data/audit.db
HOST=0.0.0.0
PORT=8000
```

## Build the Knowledge Base

Add your operational documents to the `documents/` folder and run:

```bash
python data_ingestion.py
```

This process:

- reads configuration from the environment
- validates the Pinecone index
- loads files from the document folder
- chunkizes them into searchable units
- embeds them using the configured embedding model
- upserts them into the target Pinecone namespace

## Run the Application

Start the app locally with:

```bash
python app.py
```

or:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Using the App

The interface includes a document upload area for operational files such as:

- PDF
- TXT
- MD
- DOCX

After upload, the backend ingests the files into the configured Pinecone collection so the assistant can answer questions using updated internal knowledge.

Example questions:

- "Our checkout API is returning 502 errors after deployment. What should we check first?"
- "What should I check next if the first step does not resolve the issue?"
- "CPU usage on the payments service is above 90%. What is the recommended action?"

## Persistent Memory

The workflow uses a SQLite-backed LangGraph checkpoint, allowing each user session to maintain a stable `thread_id` across multiple follow-up prompts.

This is useful for incident workflows where the operator keeps refining the investigation over time.

## Groq Rate-Limit Note

Groq can return a 429 rate-limit error when the account reaches its token-per-minute quota. A typical message looks like:

```text
Rate limit reached for model openai/gpt-oss-20b ... tokens per minute (TPM)
```

Recommended mitigations:

- upgrade to a higher Groq plan or usage tier
- reduce the number of model calls per workflow
- lower excessive retry volume
- cache repeated answers when possible
- keep prompts and retrieval flow efficient

The project includes retry and backoff handling for provider rate-limit responses, and the delay values can be tuned as needed.

## Troubleshooting

### Unsupported Groq model

If Groq reports a model-not-found or unsupported-model error, verify that the configured model is currently available in your account.

Current default:

```env
GROQ_MODEL=openai/gpt-oss-20b
```

### Missing environment variables

If the app fails to start, confirm that the following variables are present in `.env`:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `PINECONE_API_KEY`
- `TAVILY_API_KEY`

### Pinecone mismatch

Ensure the embedding model, vector dimension, and index configuration are consistent across ingestion and retrieval. A mismatch may cause poor retrieval or failed lookup behavior.

## Development Notes

- retrieval and ingestion use the same embedding configuration
- internal knowledge is preferred before web search fallback
- the architecture is designed to be extended with additional domain-specific runbooks and workflows

## License

This project is intended for educational and internal operational use. Add a license file if you plan to distribute it externally or use it in production beyond your own environment.
