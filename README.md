# AroBot – Agentic RAG Multi-Modal Medical Chatbot

An AI assistant for medical Q&A with prescription OCR, PDF RAG, medicine CSV lookup, web search, hybrid conversation memory, and LangSmith tracing. Runs locally via FastAPI + Ollama. Knowledge is stored in Pinecone.

## Features

### Core Capabilities

- **Prescription OCR**: Analyze prescription images using PaddleOCR and computer vision. (PaddleOCR; lazy init; GPU-ready)
- **Multi-Modal LLM**: Separate models for text and vision processing using Ollama.Text LLM via Ollama (`neural-chat:7b`), vision via `llava:7b`
- **RAG (Retrieval-Augmented Generation)**: Knowledge base search using Pinecone vector database
- **Medical Knowledge Base**: PDF documents and medicine database integration
- **Web Search**: DuckDuckGo integration for up-to-date medical information
- **Conversation Memory**: MCP server for maintaining chat history and context.(regex perfect recall + LLM context)
- **LangSmith Integration**: Monitoring and tracing of LLM interactions

### Key Features

1. **Prescription Analysis**: Upload prescription images for automated analysis
2. **Medicine Search**: Find medications by condition or symptoms
3. **Medical Q&A**: Ask questions about diseases, treatments, and medications
4. **Drug Interactions**: Check for potential medication interactions
5. **Symptom Analysis**: Get information about symptoms and possible conditions

## System Architecture

```
AroBot_Agentic_RAG_Multi-Modal_Chatbot_System/
├── 📁 agents/               # AI agents for different tasks
│   ├── medical_agent.py     # Main orchestrating agent
│   ├── rag_agent.py         # RAG operations
│   └── ocr_agent.py         # OCR and image processing
├── 📁 api/                  # FastAPI web interface
│   ├── main.py              # FastAPI application
│   └── routes.py            # API endpoints
├── 📁 config/               # Configuration
│   └── env_config.py        # Environment settings
├── 📁 core/                 # Core components
│   ├── embeddings.py        # Text embeddings
│   ├── llm_handler.py       # LLM management
│   ├── multimodal_processor.py # Multi-modal processing
│   └── vector_store.py      # Pinecone integration
├── 📁 data/                 # PDF knowledge base
│   ├── Human Anatomy.pdf
│   └── Medical_book.pdf
├── 📁 mcp_server/          # Memory and context
│   ├── conversation_memory.py
│   └── mcp_handler.py
├── 📁 prescribtion data/   # Sample prescription images (129 images)
├── 📁 static/              # Static web files
├── 📁 templates/           # HTML templates
│   └── chat_enhanced.html  # Web chat interface
├── 📁 utils/               # Utilities
│   ├── data_ingestion.py   # Data loading
│   ├── ocr_pipeline.py     # OCR processing
│   ├── setup_knowledge_base.py # KB setup
│   └── web_search.py       # Web search integration
├── 📁 Web Scrape/          # Medicine database
│   ├── generic.csv         # Generic medicine data
│   └── medicine.csv        # Brand medicine data
├── app.py                  # Main application entry
├── setup_system.py        # Automated setup script
└── requirements.txt       # Python dependencies
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


# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=arobot-multimodal-chatbot-system

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=llava:7b

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

Start the system

```bash
python app.py
# Chat UI:  http://localhost:8000/chat
# Docs:     http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

## Key endpoints

- POST `/api/v1/chat`

  - JSON: `{ "message": str, "session_id": str|null, "use_web_search": bool|null }`
  - Hybrid memory first (name/department/hospital), else RAG + LLM

- POST `/api/v1/prescription/upload`

  - multipart/form-data: `file`, `image_type` (prescription|general), optional `query`, `session_id`
  - Default image description when no prompt

- POST `/api/v1/pdf/analyze`

  - Describe or QA a PDF without writing to vectors

- POST `/api/v1/pdf/upload`

  - Ingest PDF into default PDF store

- POST `/api/v1/vector/create-index`

  - Create new Pinecone index from uploaded PDF; name auto-formatted

- GET `/api/v1/vector/indexes`
  - List Pinecone indexes with stats

## How it works (short)

- `medical_agent.py` routes queries to RAG, OCR, image analysis, or web search; adds conversation context.
- `rag_agent.py` retrieves from Pinecone and calls `llm_handler`.
- `llm_handler.py` builds prompts (with conversation history) and calls Ollama.
- `ocr_pipeline.py` runs PaddleOCR with tuned defaults.
- `api/routes.py` includes PerfectMemoryProcessor for instant recall of name/department/hospital.

## Troubleshooting

- Ollama errors: ensure Ollama is running; `ollama list`; pull the models above.
- Pinecone name error: avoid spaces/underscores; server auto-formats names.
- Slow OCR: prefer GPU and clear scans.
- Memory recall: ensure same `session_id`; UI stores sessions.
- For PDF description without vector write, use `/api/v1/pdf/analyze`.

## Notes

- LangSmith enables automatically if `LANGSMITH_API_KEY` is set.
- Config is centralized in `config/env_config.py` and loaded from `.env`.
- UI lives at `/chat`, no quick actions, copy supported.

## License & Disclaimer

For education/research. Not medical advice.
