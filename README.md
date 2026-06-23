<<<<<<< HEAD
# UPI Intelligence Analyst

AI-powered UPI analytics system using a Medallion architecture pipeline + multi-agent RAG.

## Project Structure

```
upi_intelligence/
├── data/                     ← raw Excel files + processed Parquet files
│   ├── chargeback_2023.xlsx
│   ├── chargeback_2024.xlsx
│   ├── chargeback_2025.xlsx
│   ├── bhim_2023_24.xlsx
│   ├── bhim_2024_25.xlsx
│   ├── bhim_2026_27.xlsx
│   └── processed/            ← auto-created by the pipeline
├── pipeline/
│   ├── bronze_layer.py       ← raw ingestion
│   ├── silver_layer.py       ← cleaning & feature engineering
│   └── gold_layer.py         ← business aggregations + knowledge base
├── rag/
│   ├── embeddings.py         ← generate & persist sentence embeddings
│   ├── retriever.py          ← semantic search
│   └── rag.py                ← Gemini RAG pipeline
├── agents/
│   ├── agents.py             ← Analytics, Merchant, Customer, Fraud, Reporting agents
│   └── router.py             ← keyword + LLM-based intelligent router
├── backend/
│   └── main.py               ← FastAPI REST API
├── frontend/
│   └── app.py                ← Streamlit chat UI
├── scripts/
│   └── run_pipeline.py       ← one-shot setup script
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

```bash
cp .env.example .env
# Then edit .env and set your GEMINI_API_KEY
```

Get a free key at: https://aistudio.google.com/app/apikey

### 3. Run the data pipeline (once)

This reads the Excel files, processes them through Bronze → Silver → Gold layers,
and generates sentence embeddings for the knowledge base.

```bash
python scripts/run_pipeline.py
```

### 4. Start the app

**Option A — Streamlit chat UI (recommended)**
```bash
streamlit run frontend/app.py
```

**Option B — FastAPI REST backend**
```bash
uvicorn backend.main:app --reload
# Open: http://127.0.0.1:8000/docs
```

## How it works

1. **Pipeline** reads raw Excel files → cleans & aggregates → builds a narrative knowledge base
2. **Embeddings** converts every knowledge-base document into a vector using `all-MiniLM-L6-v2`
3. **Retriever** finds the most relevant documents for any user query using cosine similarity
4. **RAG** feeds those documents as context to Gemini to generate a grounded answer
5. **Router** classifies each query (keyword-first, LLM fallback) and picks the right agent
6. **UI** provides a Streamlit chat or FastAPI endpoint

## Sample questions

- "What is CB Ratio?"
- "Which bank has the highest chargeback risk?"
- "How was BHIM growth in 2024?"
- "Find banks with suspicious transaction patterns"
- "Generate an executive summary of bank risk"
=======
# UPI-Intelligence-analyst
>>>>>>> a5afd95c537cb05d38df376710d622403b43b1bc
