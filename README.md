# 🎯 Resume Tailor Agent

An autonomous AI agent that ingests candidate resume PDFs, analyzes semantic gaps against Job Descriptions (JDs), suggests strictly honest bullet rewrites, and maintains candidate goals across sessions. Built with a hand-written agent loop (no LangChain, CrewAI, or AutoGen).

---

## 🛠️ Tech Stack

- **Language & Runtime:** Python 3.11+
- **LLM Runtime:** Ollama (REST API via `httpx`, default model: `llama3.2:latest` or `qwen2.5:7b`)
- **Embeddings / RAG:** `all-MiniLM-L6-v2` / `nomic-embed-text` with CUDA GPU tensor acceleration
- **Backend Framework:** FastAPI, Pydantic v2
- **Database & ORM:** SQLAlchemy 2.0 with SQLite (Postgres-ready via `DATABASE_URL`)
- **PDF Extraction:** `pdfplumber`
- **Frontend UI:** Streamlit
- **Testing & Evals:** `pytest`, custom evaluation harness
- **Containerization:** Docker & Docker Compose

---

## 📁 Project Structure

```text
resume-tailor-agent/
│
├── app/                                 # Core backend and agent logic
│   ├── __init__.py
│   ├── config.py                        # Pydantic BaseSettings, env loading & Ollama health checks
│   ├── schemas.py                       # Pydantic domain models (Profile, Requirement, Gap, Rewrite)
│   ├── db.py                            # SQLAlchemy models (User, Resume, Session, StepTrace, JDCache)
│   ├── llm.py                           # Ollama REST client, GPU offloading (num_gpu: 99), retry reflection
│   ├── verifier.py                      # Anti-hallucination checker (regex metrics + 150-tool catalog)
│   ├── memory.py                        # User preference store & 3-5 line session summarizer
│   ├── agent.py                         # Hand-written tool-calling loop, budget guards & fallback pipeline
│   ├── demo.py                          # Terminal CLI runner for quick end-to-end testing
│   ├── main.py                          # FastAPI REST endpoints (/upload, /chat, /sessions/trace, /users)
│   └── tools/                           # Native tools callable by the agent
│       ├── __init__.py                  # Tool registry & native Ollama JSON function schemas
│       ├── parse_resume.py              # pdfplumber PDF text extraction + structured Profile generation
│       ├── analyze_jd.py                # JD requirement extraction backed by SQLite SHA-256 cache
│       ├── match_skills.py              # Batched GPU matrix cosine similarity RAG + gap assessor
│       ├── suggest_rewrite.py           # Grounded bullet rewriter with verifier pass
│       └── memory_tools.py              # save_preference and recall_preferences agent wrappers
│
├── ui/
│   └── streamlit_app.py                 # Interactive web interface (PDF upload, chat, gap matrix, trace)
│
├── evals/                               # Evaluation benchmark harness
│   ├── cases/                           # Realistic test cases (resume + JD + ground-truth must-haves)
│   │   ├── case_01_backend_intern.json
│   │   ├── case_02_ai_agent_engineer.json
│   │   ├── case_03_data_infrastructure.json
│   │   ├── case_04_fullstack_ai.json
│   │   └── case_05_junior_mlops.json
│   ├── run_evals.py                     # Evaluates schema validity, recall, fabrication & latency
│   └── results.md                       # Output benchmark markdown table
│
├── tests/                               # Comprehensive unit and integration test suite
│   ├── conftest.py                      # In-memory SQLite fixtures & mock utilities
│   ├── test_schemas.py                  # Pydantic v2 field and type validation tests
│   ├── test_verifier.py                 # Metric regex & tech catalog claim detection tests
│   ├── test_memory.py                   # Cross-session continuity & summary persistence tests
│   ├── test_jd_cache.py                 # SHA-256 cache hit/miss verification tests
│   ├── test_parse_resume.py             # PDF text extraction & 1-time schema retry tests
│   ├── test_agent_loop.py               # Tool-calling path, fallback path, and token budget tests
│   └── test_api.py                      # FastAPI route integration tests (/health, /upload, /users)
│
├── .env.example                         # Template configuration file
├── .gitignore                           # Excludes .env, *.db, uploads/, __pycache__, and test caches
├── Dockerfile                           # Production container definition
├── docker-compose.yml                   # Multi-service setup (FastAPI on 8000 + Streamlit on 8501)
├── generate_sample_pdf.py               # Script to generate sample_resume.pdf for testing
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Pinned project dependencies
└── sample_jd.txt                        # Sample job description for CLI demo
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3.11+**
- **Ollama** installed and running ([Download Ollama](https://ollama.com/download))
- *(Optional)* NVIDIA GPU with CUDA for hardware acceleration

Pull the required local models:
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/FABvivek/resume-tailor-agent.git
cd resume-tailor-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate     # On Windows
# source venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file:
```bash
copy .env.example .env      # On Windows
# cp .env.example .env       # On Linux/macOS
```

Default `.env` settings:
```ini
# Ollama Host (default local port 11434)
# When running inside Docker, use http://host.docker.internal:11434
OLLAMA_HOST=http://localhost:11434

# Active local models
OLLAMA_MODEL=llama3.2:latest
EMBED_MODEL=nomic-embed-text

# Database & Storage
DATABASE_URL=sqlite:///./resume_tailor.db
UPLOAD_DIR=./uploads

# Agent Budget Controls
MAX_STEPS=8
MAX_SESSION_TOKENS=16000

# Verifier Settings
ENABLE_LLM_VERIFIER=false
```

---

## 🚀 How to Run

### 1. Run Automated Tests (Offline-Ready)
All 22 unit and integration tests run without requiring Ollama or network calls:
```bash
pytest -v
```

### 2. Run the Terminal Demo
Test the end-to-end agent loop directly from your command line:
```bash
# 1. Generate sample resume PDF (if not already present)
python generate_sample_pdf.py

# 2. Run CLI demo
python -m app.demo sample_resume.pdf sample_jd.txt
```

### 3. Run the Full Stack (API + UI)

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run ui/streamlit_app.py
```
Open your browser at: [http://localhost:8501](http://localhost:8501)

### 4. Run with Docker Compose
To spin up both the FastAPI backend and Streamlit UI in containers:
```bash
docker compose up --build
```
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI API: [http://localhost:8000](http://localhost:8000)

*(Note: Containers reach your host Ollama instance via `host.docker.internal:11434`)*

### 5. Run the Evaluation Benchmark
Run quantitative evaluations across the 5 benchmark cases:
```bash
python evals/run_evals.py
```
Results and metrics will be saved directly to `evals/results.md`.
