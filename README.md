# AroBot – LLM-as-Agent Medical Assistant

An intelligent AI medical assistant featuring **LLM-as-Agent architecture** where the AI automatically selects tools based on input. Features prescription OCR, PDF analysis, medicine database, web search, and conversation memory. Built with **FastAPI, Ollama, and Pinecone** for scalable local deployment.

## 🎯 Core Features

### LLM-as-Agent Architecture

- **Intelligent Tool Selection**: LLM automatically chooses the right tools for any input
- **Unified Interface**: Single endpoint handles text, images, PDFs, and any combination
- **Adaptive Processing**: Context-aware tool selection based on conversation history
- **Extensible Design**: Easy to add new tools without changing routing logic

### Multi-Modal Capabilities

- **Prescription OCR**: Analyze prescription images using PaddleOCR and computer vision
- **PDF Analysis**: Extract and analyze medical documents, papers, and guidelines
- **Image Understanding**: Process medical diagrams, anatomy charts, and general images
- **Text Analysis**: Medical Q&A, symptom analysis, drug information
- **Web Search**: Real-time search for current medical information
- **Memory Access**: Recall previous conversations and context

### Knowledge Integration

- **RAG (Retrieval-Augmented Generation)**: Medical knowledge base using Pinecone
- **Medicine Database**: Bangladesh pharmacy and drug information
- **Medical PDFs**: WHO guidelines, national protocols, anatomy textbooks
- **Anatomy Images**: CLIP-indexed medical diagrams and figures

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Ollama running at http://localhost:11434
- Pinecone account (recommended)

### Installation

```bash
git clone https://github.com/MD-Junayed000/AroBot_Agentic_RAG_Multi-Modal_Chatbot_System.git
cd AroBot_Agentic_RAG_Multi-Modal_Chatbot_System
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Setup Models

```bash
ollama pull llama3.2:3b
ollama pull llava:7b
```

### Configure Environment

Create `.env` file:

```ini
# Required
PINECONE_API_KEY=your_pinecone_api_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=llava:7b

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key
ENABLE_ADMIN_ROUTES=true  # For admin access
DEBUG=true
```

### Run the System

```bash
# Setup knowledge base (first time)
python -m utils.setup_knowledge_base

# Start the server
python app.py
```

## 🌐 Access Points

- **🤖 Agent Interface**: http://localhost:8000/api/v1/agent (**Primary**)
- **💬 Chat UI**: http://localhost:8000/chat
- **📖 API Documentation**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/health

## 📡 API Usage

### Unified Agent Endpoint (Recommended)

The agent automatically determines which tools to use based on your input:

```bash
# Text query
curl -X POST "/api/v1/agent" \
  -F "message=What is diabetes and how is it treated?"

# Image analysis
curl -X POST "/api/v1/agent" \
  -F "file=@prescription.jpg" \
  -F "message=What medicines are prescribed?"

# PDF analysis
curl -X POST "/api/v1/agent" \
  -F "file=@medical_paper.pdf" \
  -F "message=Summarize the key findings"

# Combined input
curl -X POST "/api/v1/agent" \
  -F "file=@image.jpg" \
  -F "message=Explain this anatomy diagram" \
  -F "session_id=user123"
```

### Tool Discovery

```bash
# List all available tools
GET /api/v1/agent/tools

# Filter tools by category
GET /api/v1/agent/tools?category=medical

# See which tools would be selected (without execution)
POST /api/v1/agent/explain
```

### Response Format

```json
{
  "response": "Generated response text",
  "session_id": "session-uuid",
  "tools_used": ["analyze_text", "get_medicine_info"],
  "sources": {"tools": [...], "llm_agent": true},
  "status": "success",
  "agent_type": "llm_agent"
}
```

## 🧰 Available Tools

### Medical Tools (Priority: 4-5)

- **analyze_text**: Medical Q&A, symptoms, general conversation
- **analyze_image**: Prescription OCR, medical diagrams, anatomy
- **get_medicine_info**: Drug information, dosages, prices

### Document Tools (Priority: 3)

- **analyze_pdf**: PDF extraction, summarization, Q&A

### Utility Tools (Priority: 1-2)

- **get_weather**: Weather information by location
- **web_search**: Real-time web search for current information
- **access_memory**: Conversation history and context

## 🏗️ Architecture

### LLM-as-Agent Flow

```
[User Input] → [LLM Agent] → [Tool Selection] → [Tool Execution] → [Response Synthesis]
```

### System Components

```
AroBot_Agentic_RAG_Multi-Modal_Chatbot_System/
├── core/
│   ├── agent_core.py              # NEW: LLM Agent with tool registry
│   ├── llm_handler.py             # Text/vision LLM integration
│   ├── vector_store.py            # Pinecone wrapper
│   ├── embeddings.py              # Sentence transformers
│   └── image_index.py             # CLIP image indexing
├── api/
│   ├── main.py                    # FastAPI application
│   ├── core_routes.py             # Essential routes (cleaned)
│   └── routes/
│       ├── agent.py               # NEW: Unified agent endpoints
│       └── admin.py               # NEW: Admin routes (hidden)
├── agents/
│   ├── medical_agent.py           # Medical orchestration
│   └── rag_agent.py               # RAG operations
├── scripts/
│   ├── ocr_agent.py               # MOVED: OCR utilities (offline)
│   ├── ingest_pdfs_bd.py          # PDF knowledge base setup
│   └── ingest_anatomy_images.py   # Image index setup
├── mcp_server/
│   └── mcp_handler.py             # Conversation memory
├── data/pdfs/                     # Medical knowledge base
├── prescription_data/             # Sample prescription images
└── templates/chat_enhanced.html   # Web UI
```

## 🔧 Development

### Adding Custom Tools

```python
from core.agent_core import tool, LLMAgent

class CustomAgent(LLMAgent):
    @tool(
        name="calculate_bmi",
        description="Calculate BMI from height and weight",
        category="medical_calculations",
        priority=3
    )
    async def calculate_bmi(self, height_cm: float, weight_kg: float):
        bmi = weight_kg / ((height_cm / 100) ** 2)
        return {"response": f"BMI: {bmi:.1f}", "bmi": bmi}
```

### Admin Routes

Admin endpoints are hidden from the main API schema and require environment flag:

```bash
# Enable admin routes
export ENABLE_ADMIN_ROUTES=true

# Access admin endpoints
GET /api/v1/admin/system/health
GET /api/v1/admin/vector/indexes
POST /api/v1/admin/pdf/upload
```

## 📊 Monitoring

### Built-in Monitoring

- LangSmith integration for LLM tracing
- Tool usage analytics
- Performance metrics
- Error tracking

### Debug Endpoints

```bash
# System health
GET /api/v1/admin/system/health

# Tool registry info
GET /api/v1/admin/debug/agent-tools

# Explain tool selection
POST /api/v1/agent/explain
```

## 🔄 Migration from Legacy

### Old vs New API

**Before (Multiple Endpoints)**:

```bash
POST /api/v1/chat              # Text only
POST /api/v1/image/analyze     # Images only
POST /api/v1/pdf/analyze       # PDFs only
```

**After (Single Endpoint)**:

```bash
POST /api/v1/agent             # Everything!
```

### Legacy Support

All legacy endpoints are preserved for backward compatibility but marked as deprecated. The system will guide users to migrate to the unified agent endpoint.

## 🎯 Benefits

### For Users

- **Simplified Usage**: One endpoint for all input types
- **Intelligent Processing**: LLM chooses the best tools automatically
- **Better Context**: Conversation-aware responses
- **Faster Responses**: Optimized tool selection and execution

### For Developers

- **Easy Extension**: Add new tools without changing routes
- **Clean Architecture**: Separation of concerns
- **Better Testing**: Unified interface simplifies testing
- **Maintainable**: Less routing logic to maintain

## 🚀 Performance

### Optimizations Applied

- Eliminated double RAG calls (168s → ~5s response time)
- Reduced context explosion (21,600 → ~4,000 chars max)
- Fast paths for simple follow-up questions
- Smart tool selection caching
- Parallel tool execution where possible

### Model Configuration

- Text LLM: `llama3.2:3b` with `num_ctx=4096`, `num_predict=250`
- Vision LLM: `llava:7b` for image analysis
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`

## 📚 Documentation

- **[Agent Architecture](AGENT_ARCHITECTURE.md)**: Detailed architecture overview
- **[Tool Registry](TOOL_REGISTRY.md)**: Tool development guide
- **[Custom Tools Example](examples/custom_tools.py)**: Working examples

## ⚡ Troubleshooting

- **Ollama Issues**: Ensure models are pulled and service is running
- **Pinecone Errors**: Check API key and index configurations
- **Slow Responses**: Monitor tool selection and disable unused namespaces
- **Memory Issues**: Clear conversation memory via admin endpoints

## 📄 License & Disclaimer

For educational and research purposes. **Not for medical advice.** Always consult healthcare professionals for diagnosis and treatment.

---

**🎉 Experience the future of AI assistants with intelligent tool selection!**
