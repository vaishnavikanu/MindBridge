# 🧠 MindBridge: Role-Aware Graph-Enhanced Retrieval-Augmented Generation for Mental Health Assistance
> **EMNLP Reproducibility Repository**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

Overview
This repository contains a modular Retrieval-Augmented Generation (RAG) system for mental-health assistance featuring role-aware retrieval, graph-enhanced retrieval, privacy-isolated vector stores, dual embedding models, Cross-Encoder reranking, and local LLM generation using Ollama.
Key Features
Role-aware retrieval (Patient / Clinician)
Privacy-isolated FAISS indexes
Dual embeddings (SBERT + MedCPT)
Graph-enhanced retrieval
Cross-Encoder reranking
Parent-child chunking
Local LLM inference (Ollama / Llama 3)
End-to-end evaluation
Ablation framework
Repository Structure
```text
configs/
data/
evaluation/
scripts/
src/
requirements.txt
README.md
```
# 🏗️ System Architecture

The proposed system follows a modular **5-layer Retrieval-Augmented Generation (RAG)** architecture designed for privacy-preserving mental health assistance. Separate retrieval pipelines are maintained for **patients** and **clinicians**, while both share a curated mental-health knowledge base.

```mermaid
flowchart LR

A[User Query]

A --> B{Role}

B -->|Patient| C[SBERT Embedding]
B -->|Clinician| D[MedCPT Embedding]

C --> E[Patient Retriever]
D --> F[Clinician Retriever]

E --> G[Dense Retrieval]
E --> H[BM25 Retrieval]

G --> I[RRF Fusion]
H --> I

F --> J[Dense Clinical Retrieval]
J --> K[Knowledge Graph Expansion]

I --> L[Reranker]
K --> L

L --> M[Prompt Builder]

M --> N[Ollama - Llama 3]

N --> O[Grounded Response]
```

---

# 📚 Layered Architecture

| Layer | Module | Description |
|--------|--------|-------------|
| **Layer 1** | Document Ingestion | Parses curated PDFs and user-specific documents into a unified internal representation. |
| **Layer 2** | Chunking & Embedding | Parent-child chunking followed by SBERT (patient) or MedCPT (clinician) embeddings. |
| **Layer 3** | Vector Storage | Privacy-isolated FAISS indexes for curated knowledge, patient memory and clinician memory. |
| **Layer 4** | Retrieval | Role-aware retrieval using Dense Search, BM25, Reciprocal Rank Fusion (RRF), Knowledge Graph Expansion and Cross-Encoder reranking. |
| **Layer 5** | Generation | Context-aware prompt construction followed by local LLM generation using Ollama (Llama 3). |

---

# 🔀 Retrieval Pipelines

## 👤 Patient Pipeline

```text
Patient Question
        │
        ▼
 SBERT Embedding
        │
        ▼
 Dense Retrieval ─────┐
                      │
 BM25 Retrieval ──────┤
                      ▼
        Reciprocal Rank Fusion
                      │
                      ▼
      Lightweight / CrossEncoder
             Reranking
                      │
                      ▼
      Patient Prompt Builder
                      │
                      ▼
        Ollama (Llama 3)
                      │
                      ▼
     Evidence-Grounded Answer
```

---

## 👨‍⚕️ Clinician Pipeline

```text
Clinical Query
      │
      ▼
 MedCPT Embedding
      │
      ▼
 Clinical Dense Retrieval
      │
      ▼
 Knowledge Graph Expansion
      │
      ▼
 CrossEncoder Reranking
      │
      ▼
 Clinical Prompt Builder
      │
      ▼
   Ollama (Llama 3)
      │
      ▼
 Clinical Response
```

---

## Multilingual Support

MindBridge supports multilingual user queries through an integrated preprocessing pipeline.

1. Detect the input language using Lingua.
2. Translate non-English queries to English using facebook/nllb-200-distilled-600M.
3. Perform GraphRAG retrieval and grounded generation.
4. Return responses in the configured output language.

This enables retrieval over a unified English knowledge base while allowing users to interact in multiple languages.
---

# 🔒 Privacy Isolation

The system enforces strict data isolation by maintaining independent FAISS indexes.

```text
Curated Knowledge Base
│
├── curated_kb_sbert.index
│
└── curated_kb_medcpt.index


Patient Data
│
├── user_001_private.index
├── user_002_private.index
└── ...


Clinician Data
│
├── clinician_001_private.index
├── clinician_002_private.index
└── ...
```

No patient embeddings are stored inside clinician indexes, and clinician-specific information is never retrieved for patient queries.

---

# 🧠 Core Components

| Component | Implementation |
|-----------|----------------|
| Chunking | Parent-Child Chunking |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Graph Retrieval | Knowledge Graph Expansion |
| Patient Embedding | SBERT |
| Clinical Embedding | MedCPT |
| Reranker | BAAI/bge-reranker-base |
| Generator | Ollama (Llama 3) |
| Framework | PyTorch + Hugging Face |

The Hugging Face models download automatically on first use. Install the generator separately:
```bash
ollama pull llama3
```

The following Hugging Face models are used by the system and will be cached automatically during the first execution:

- sentence-transformers/all-MiniLM-L6-v2
- ncbi/MedCPT-Article-Encoder
- BAAI/bge-reranker-base
- facebook/nllb-200-distilled-600M

  
Installation
```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets
CounselChat (patient evaluation)
Curated clinician benchmark (corpus-aligned evaluation)
Place datasets under `evaluation/data/`.
Build Knowledge Base
```bash
python scripts/ingest.py
```
Run Queries
Patient:
```bash
python scripts/query.py --role patient --user-id patient_001 --query "I feel anxious."
```
Clinician:
```bash
python scripts/query.py --role clinician --user-id clinician_001 --query "DSM-5 criteria for MDD"
```
Evaluation
```bash
python -m evaluation.run_evaluation
```
Ablation
```bash
python -m evaluation.run_ablation
```
Experiments:
Dense
Graph
Role-Aware
Full
Outputs are written to `evaluation/results/`.
Configuration
All runtime settings are in `configs/config.yaml`.
Hardware
Experiments were performed on CUDA-enabled NVIDIA GPUs (Tesla V100 / A100 class). CPU execution is supported but slower.
Troubleshooting
Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Verify Ollama:
```bash
ollama ps
```
---

# 🌐 Web Application Setup

The project also includes a React.js frontend integrated with the FastAPI backend and PostgreSQL database.


Create a `.env` file in the project root and configure your PostgreSQL connection:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/mindbridge
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

Backend:

```
http://127.0.0.1:8000
```

Swagger API:

```
http://127.0.0.1:8000/docs
```

---

## Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```
http://localhost:5173
```

---

## Notes

- PostgreSQL must be running before starting the backend.
- Configure the database connection in the `.env` file before running the application.
- Start the backend before launching the frontend.

## Web Application Features

- Secure authentication for Patients and Clinicians
- Role-based dashboards
- AI-powered multilingual chatbot
- Mood tracking and journaling
- Chat history management
- Patient history access for clinicians
- File attachment support
- Responsive React.js interface
- PostgreSQL-backed data management
- Dark mode support

License
MIT