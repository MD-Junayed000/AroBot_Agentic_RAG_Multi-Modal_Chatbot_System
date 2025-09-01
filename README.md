# AroBot – Agentic RAG Multi-Modal Medical Chatbot

An advanced AI medical assistant featuring prescription OCR, PDF RAG, medicine database integration, web search capabilities, hybrid conversation memory, and monitoring. Built with **_FastAPI, Ollama, and Pinecone_** for scalable, local deployment.

## Features

### Core Capabilities

- **Prescription OCR**: Analyze prescription images using PaddleOCR and computer vision. (PaddleOCR; lazy init; GPU-ready)
- **Multi-Modal LLM**: Separate models for text and vision processing using Ollama.Text LLM via Ollama (`llama3.2:3b`), vision via `llava:7b`
- **RAG (Retrieval-Augmented Generation)**: Knowledge base search using Pinecone vector database
- **Medical Knowledge Base**: PDF documents and medicine database integration
- **Web Search**: DuckDuckGo (ddgs) for recent medical info/news with normalised results.
- **Conversation Memory**: MCP server for maintaining chat history and context.(regex perfect recall + LLM context)
- **LangSmith Integration**: Monitoring and tracing of LLM interactions

### Key Features

1. **Prescription Analysis**: Upload prescription images for automated analysis
2. **Medicine Search**: Find medications by condition or symptoms
3. **Medical Q&A**: Ask questions about diseases, treatments, and medications
4. **_Upload PDFs_**: to the knowledge base (namespaces + vector store)
5. **_Search anatomy figures_**: via CLIP image retrieval.

## System Architecture

```
AroBot_Agentic_RAG_Multi-Modal_Chatbot_System/
├── agents/
│   ├── __init__.py
│   ├── medical_agent.py           # Orchestration
│   ├── rag_agent.py               # RAG ops
│   └── ocr_agent.py               # OCR/image utilities
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app (entry or import)
│   └── core_routes.py                  # Endpoints (chat, pdf, vector, etc.)
│   └── route/
│       └── __init__.py
│       └──  image.py
├── config/
│   ├── __init__.py
│   └── env_config.py              # Env and paths
├── core/
│   ├── __init__.py
│   ├── embeddings.py              # Sentence Transformers embedder
│   ├── llm_handler.py             # Text/vision LLM, OCR-first flows
│   ├── image_index.py             # CLIP index (anatomy figures)
│   └── vector_store.py            # Pinecone wrapper (auto-create indexes)
│   ├── multimodal_processor.py
├── data/
│   └── pdfs/
│       ├── WHO Guide to Good Prescribing.pdf
│       ├── National Guideline on Hypertension Bangladesh.pdf
│       ├── National Drug Policy, Bangladesh (2016).pdf
│       ├── Over-the-Counter (OTC) Medicines List (official list).pdf
|       ├── WHO Model List of Essential Medicines for Children (9th, 2023).pdf
|       ├── National Protocol for Management of.pdf
|       ├── Drugs & Cosmetics Law.pdf
|       ├── A_Study_on_the_National_Drug_Policies_of_Banglades.pdf
│       └── anatomy/
│           ├── Human Anatomy.pdf
│           ├── ross-and-wilson-anatomy-and-physiology-in-health-a.pdf
│           └── color-atlas-of-anatomy.pdf
├── mcp_server/
│   ├── __init__.py
|   ├── conversation_memory.py
│   └── mcp_handler.py             # Persistent memory & session store
├── scripts/
│   ├── __init__.py
│   ├── ingest_pdfs_bd.py          # Ingest text PDFs → Pinecone (namespaces)
│   └── ingest_anatomy_images.py   # Index anatomy figures → CLIP index
│   ├── refresh_pharma_cache.py
├
├── pharma/
│   ├── __init__.py
│   ├── resolver.py
│   ├── provider/
│       ├── __init__.py
├        ├── medex.py
├
├── templates/
│   └── chat_enhanced.html         # Web chat UI
│── prescription_data/   # Sample prescription images (129 images)
├── utils/
│   ├── __init__.py
│   ├── logging_filter.py
│   ├── weather_utils.py
│   ├── pdf_image_extractor.py
│   ├── medicine_intent.py
│   ├── data_ingestion.py
│   ├── ocr_pipeline.py            # PaddleOCR wrapper + parsing
│   ├── setup_knowledge_base.py    # Orchestrates both ingest scripts
│   └── web_search.py              # ddgs + BD medicine resolvers
├── Web Scrape/
│   ├── generic.csv
│   └── medicine.csv
|   └──dosage_form.csv
|   └──drug_class.csv
|   └──indication.csv
|   └──manufacturer.csv
├── app.py                         # Simple launcher for the API
└── requirements.txt

```

## Setup

### 1. Prerequisites

- Python 3.10.0
- Ollama running at http://localhost:11434
- Pinecone account (recommended)
- langsmith account (optional for monitoring)

### 2. Install

```bash
git clone https://github.com/MD-Junayed000/AroBot_Agentic_RAG_Multi-Modal_Chatbot_System.git
cd AroBot_Agentic_RAG_Multi-Modal_Chatbot_System
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Pull models (Ollama)

```bash
ollama pull llama3.2:3b
ollama pull llava:7b
```

### 4. Configure .env

```ini
# API Keys - Replace with your actual keys
PINECONE_API_KEY=your_pinecone_api_key
LANGSMITH_API_KEY=your_langsmith_api_key

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_PDF_INDEX=arobot-medical-pdfs
PINECONE_MEDICINE_INDEX=arobot-medicine-data
PINECONE_BD_PHARMACY_INDEX= arobot-bd-pharmacy
PINECONE_IMAGE_INDEX= arobot-anatomy-images

# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=arobot-multimodal-chatbot-system

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=llava:7b
# Faster formatter model (optional if you have it pulled)
OLLAMA_FAST_TEXT_MODEL=llama3.2:1b

# Regional defaults (used by medicine)
DEFAULT_REGION =BD


# OCR Configuration
OCR_LANGUAGE=en
OCR_CONFIDENCE_THRESHOLD=0.5

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true

```

### 5. Local Run and Testing

Setup knowledge base

```bash
python -m utils.setup_knowledge_base
```

This will:

> > ingest text PDFs into the BD Pinecone index (with namespaces)

> > index anatomy figures into the CLIP image index

Start the system

```bash
python app.py
```

##### Access Points

- **🌐 Main Interface**: http://localhost:8000
- **💬 Chat UI**: http://localhost:8000/chat
- **📖 API Docs**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/health

## APIendpoints

### Core Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
  "message": "What is paracetamol used for?",
  "session_id": "user_session_123",
  "use_web_search": true
}
```

### Prescription Analysis (Image)

```http
POST /prescription/upload
Content-Type: multipart/form-data

file: [image.jpg|png]
image_type: "prescription" | "general"
query: "What meds are prescribed?"
session_id: "user_session_123"
```

> > > Compatibility alias: POST /image/analyze (same params; image_type="general").

### Prescription (text query)

```http
POST /prescription/analyze
Content-Type: application/json

{
  "query": "Patient takes metformin and glimepiride—any duplications?",
  "session_id": "user_session_123"
}
```

### PDF Processing

```http
POST /api/v1/pdf/analyze
Content-Type: multipart/form-data

file: [medical_document.pdf]
query: "Summarize the key points"  # Optional
session_id: "user_session_123"
```

### Vector Database Management

```http
POST /vector/create-index
# Creates a brand-new Pinecone index from a single PDF
# (auto-formats index name, uses dimension=384)
```

```http
GET /api/v1/vector/indexes
```

### Weather Information

```http
GET /api/v1/weather?location=New%20York
```

## How it works

- **1. Request Routing**: `agents/medical_agent.py` chooses OCR, RAG, web, image, or direct memory paths and augments with conversation context.
- **2. RAG Processing**: `agents/rag_agent.py` queries Pinecone (BD namespaces first) and builds compact context.
- **3. LLM Integration**: `core/llm_handler.py` formats prompts (defaults to careful clinical style), calls Ollama text/vision models.
- **4. Multi-Modal Processing**: `- OCR pipeline using PaddleOCR with GPU acceleration
- Vision analysis with LLaVA model
- PDF text extraction and processing
- **4. Memory Management**`mcp_server/mcp_handler.py` maintains conversation context across sessions
- PerfectMemoryProcessor provides instant recall for personal information
- Hybrid approach combines rule-based extraction with LLM context

## Troubleshooting

- Ollama errors: ensure `OLLAMA_BASE_URL` is reachable; `ollama list` pull the models above.
- Pinecone name error: avoid spaces/underscores; server auto-formats names.
- Slow OCR: prefer GPU and clear scans.
- Memory recall: ensure same `session_id`; UI stores sessions.
- For PDF description without vector write, use `/api/v1/pdf/analyze`.

## Notes

- LangSmith enables automatically if `LANGSMITH_API_KEY` is set.
- Config is centralized in `config/env_config.py` and loaded from `.env`.
- UI lives at `/chat`, no quick actions, copy supported.

## License & Disclaimer

For education/research. Not medical advice. Always consult a clinician for diagnosis or treatment.
