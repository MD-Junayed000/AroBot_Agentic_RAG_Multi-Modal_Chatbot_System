# 🏥 AroBot - Agentic RAG Multi-Modal Medical Chatbot System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.26-green.svg)](https://langchain.com/)

AroBot is an advanced multi-modal medical chatbot system that combines Retrieval-Augmented Generation (RAG), agentic AI capabilities, and comprehensive medical knowledge processing. Built with modern AI technologies, it provides intelligent medical assistance through text, image, and document analysis.

## 🌟 Key Features

### 🤖 Agentic AI Architecture
- **LLM-as-Agent**: Intelligent tool selection and orchestration
- **Dynamic Tool Registry**: Extensible tool system with automatic discovery
- **Context-Aware Processing**: Maintains conversation state and context
- **Multi-Agent Coordination**: Specialized agents for different medical tasks

### 🔍 Advanced RAG System
- **Multi-Vector Store Support**: Pinecone integration with multiple indexes
- **Cross-Encoder Reranking**: Enhanced relevance scoring
- **Semantic Caching**: Optimized query performance
- **Context Chunking**: Intelligent document segmentation

### 🖼️ Multi-Modal Capabilities
- **Vision Processing**: CLIP-based image understanding
- **OCR Pipeline**: PaddleOCR and Tesseract integration
- **Medical Image Analysis**: Prescription, lab results, X-rays, anatomy diagrams
- **PDF Processing**: Automated text and image extraction

### 💊 Medical Specialization
- **Prescription Analysis**: Automated prescription parsing and validation
- **Medicine Database**: Comprehensive Bangladesh pharmacy data
- **Drug Interaction Checking**: Safety validation
- **Clinical Decision Support**: Evidence-based recommendations

### 🌐 Modern Web Interface
- **Enhanced Chat UI**: Modern, responsive design
- **File Upload Support**: Drag-and-drop for images and PDFs
- **Session Management**: Persistent conversation history
- **Real-time Processing**: WebSocket-like experience

### 🔧 Enterprise Features
- **Rate Limiting**: Configurable request throttling
- **Error Handling**: Comprehensive error management
- **Logging & Monitoring**: LangSmith integration
- **Admin Dashboard**: System monitoring and analytics
- **API Versioning**: V1/V2 endpoint support

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AroBot Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (FastAPI + Jinja2)                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   Chat UI       │ │   Admin Panel   │ │   API Docs      ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI Routes)                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   V1 Routes     │ │   V2 Routes     │ │  Admin Routes   ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Core Agent System                                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │  LLM Agent      │ │ Medical Agent   │ │ Tool Registry   ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Processing Modules                                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ Multimodal      │ │   RAG System    │ │   LLM Handler   ││
│  │ Processor       │ │                 │ │                 ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │  Vector Store   │ │   SQLite DB     │ │   File Storage  ││
│  │  (Pinecone)     │ │                 │ │                 ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows

### Required Services
- **Ollama**: Local LLM inference server
- **Pinecone**: Vector database (free tier available)
- **Tesseract OCR**: Text recognition (optional but recommended)

### API Keys
- **Pinecone API Key**: For vector database access
- **LangSmith API Key**: For monitoring (optional)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AroBot_Agentic_RAG_Multi-Modal_Chatbot_System.git
cd AroBot_Agentic_RAG_Multi-Modal_Chatbot_System
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv arobot-env

# Activate environment
# On Linux/macOS:
source arobot-env/bin/activate
# On Windows:
arobot-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng
sudo apt install poppler-utils  # For PDF processing
```

#### macOS:
```bash
brew install tesseract
brew install poppler
```

#### Windows:
- Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Download and install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)

### 4. Install and Configure Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:3b      # Text generation
ollama pull llava:7b         # Vision processing

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

### 5. Environment Configuration
```bash
# Run setup script to create .env file
python setup_env.py
```

Edit the generated `.env` file with your configuration:
```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_REGION=us-east-1
PINECONE_ENABLE=1
PINECONE_MEDICINE_INDEX=arobot-medicine-data
PINECONE_BD_PHARMACY_INDEX=arobot-bd-pharmacy
PINECONE_IMAGE_INDEX=arobot-clip

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=llava:7b
OLLAMA_FAST_TEXT_MODEL=llama3.2:3b

# App Configuration
DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000

# OCR Configuration
OCR_LANGUAGE=en
OCR_CONFIDENCE_THRESHOLD=0.5

# Optional: LangSmith for tracing
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=AroBot

# Optional: Web search
DEFAULT_REGION=Bangladesh
```

### 6. Initialize Knowledge Base
```bash
# Ingest medicine data (if available)
python scripts/ingest_pdfs_bd.py

# Ingest anatomy images (if available)
python scripts/ingest_anatomy_images.py

# Refresh pharmacy cache
python scripts/refresh_pharma_cache.py
```

### 7. Start the Application
```bash
python app.py
```

The application will be available at:
- **Main Chat Interface**: http://localhost:8000/chat
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Admin Panel**: http://localhost:8000/admin

## 📚 Detailed Setup Guide

### Pinecone Setup
1. Sign up at [Pinecone.io](https://pinecone.io)
2. Create a new project
3. Get your API key from the dashboard
4. Update `PINECONE_API_KEY` in `.env`

### Data Preparation
The system supports various data sources:

#### PDF Documents
Place PDF files in these directories:
- `data/pdfs/` - General medical documents
- `data/pdfs/anatomy/` - Anatomy and medical diagrams

#### Medicine Database
The system includes Bangladesh pharmacy data in `Web Scrape/` directory:
- `medicine.csv` - Medicine information
- `generic.csv` - Generic drug data
- `manufacturer.csv` - Manufacturer details
- `indication.csv` - Medical indications
- `dosage_form.csv` - Dosage forms
- `drug_class.csv` - Drug classifications

#### Prescription Images
Upload prescription images through the web interface or API for automatic processing.

### Advanced Configuration

#### OCR Languages
Configure OCR for multiple languages:
```env
OCR_LANGUAGE=en+ben  # English + Bengali
```

#### Performance Tuning
```env
PINECONE_QUERY_TIMEOUT_S=3.5
EMBEDDING_DIMENSION=384
PINECONE_BATCH=100
```

#### Memory Management
```env
EMBEDDINGS_DEVICE=cuda  # Use GPU if available
```

## 🔧 Usage

### Web Interface
1. Navigate to http://localhost:8000/chat
2. Start a conversation by typing a medical query
3. Upload images (prescriptions, lab results, X-rays)
4. Upload PDF documents for analysis

### API Usage

#### Text Query
```bash
curl -X POST "http://localhost:8000/api/v2/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the side effects of paracetamol?",
    "session_id": "user123"
  }'
```

#### Image Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/agent" \
  -F "file=@prescription.jpg" \
  -F "message=Analyze this prescription" \
  -F "image_type=prescription"
```

#### Multi-modal Query
```bash
curl -X POST "http://localhost:8000/api/v2/multimodal/analyze" \
  -F "file=@medical_report.pdf" \
  -F "query=Summarize the key findings"
```

### Available Endpoints

#### V1 API (Stable)
- `POST /api/v1/agent` - Unified agent endpoint
- `GET /api/v1/health` - Health check
- `POST /api/v1/session/create` - Create session

#### V2 API (Enhanced)
- `POST /api/v2/chat` - Enhanced chat
- `POST /api/v2/multimodal/analyze` - Multi-modal analysis
- `GET /api/v2/tools/stats` - Tool statistics

#### Admin API
- `GET /admin/health` - System health
- `GET /admin/vector/indexes` - Vector store status
- `POST /admin/cache/clear` - Clear caches

## 🛠️ Development

### Project Structure
```
AroBot_Agentic_RAG_Multi-Modal_Chatbot_System/
├── agents/                 # AI agents
│   ├── medical_agent.py   # Medical specialist agent
│   └── __init__.py
├── api/                   # FastAPI application
│   ├── main.py           # Main application
│   ├── routes/           # API routes
│   ├── middleware/       # Custom middleware
│   └── schemas/          # Pydantic schemas
├── core/                 # Core functionality
│   ├── agent_core.py     # Main agent system
│   ├── multimodal_processor.py
│   ├── vector_store.py   # Pinecone integration
│   ├── llm/             # LLM modules
│   ├── rag/             # RAG system
│   └── prompts/         # System prompts
├── utils/               # Utility functions
│   ├── ocr_pipeline.py  # OCR processing
│   ├── web_search.py    # Web search tool
│   └── data_ingestion.py
├── scripts/            # Data processing scripts
├── templates/          # HTML templates
├── static/            # Static files
├── data/              # Data storage
├── memory/            # Session memory
└── config/            # Configuration
```

### Adding New Tools
```python
# In your agent class
@tool(
    name="new_medical_tool",
    description="Description of what this tool does",
    category="medical",
    priority=1
)
def new_medical_tool(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Implement your tool logic here
    """
    return {"result": "tool output"}
```

### Custom Middleware
```python
# In api/middleware/
class CustomMiddleware:
    async def __call__(self, request: Request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response
```

### Running Tests
```bash
# Run basic health checks
python -c "from api.main import app; print('✅ Import successful')"

# Test OCR pipeline
python -c "from utils.ocr_pipeline import OCRPipeline; ocr = OCRPipeline(); print('✅ OCR ready')"

# Test vector store
python -c "from core.vector_store import PineconeStore; print('✅ Vector store ready')"
```

## 🔍 Monitoring and Debugging

### LangSmith Integration
Configure LangSmith for detailed tracing:
```env
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AroBot
```

### Logging Configuration
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Monitoring
- System health: `GET /admin/health`
- Vector store status: `GET /admin/vector/indexes`
- Memory usage: Available in admin panel

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

#### 2. Pinecone Connection Issues
- Verify API key in `.env`
- Check index names match configuration
- Ensure proper region setting

#### 3. OCR Not Working
```bash
# Test Tesseract installation
tesseract --version

# Install language packs
sudo apt install tesseract-ocr-eng tesseract-ocr-ben
```

#### 4. Memory Issues
- Reduce `EMBEDDING_DIMENSION`
- Lower `PINECONE_BATCH` size
- Use CPU instead of GPU for embeddings

#### 5. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

### Performance Optimization

#### 1. Vector Store
- Use appropriate embedding dimensions
- Implement query caching
- Optimize chunk sizes

#### 2. LLM Processing
- Use faster models for simple queries
- Implement response caching
- Adjust temperature settings

#### 3. Image Processing
- Resize large images before processing
- Use appropriate image formats
- Enable GPU acceleration if available

## 📄 API Documentation

### Authentication
Currently, the system uses session-based authentication. API keys can be implemented for production use.

### Rate Limiting
- Default: 60 requests per minute
- Burst limit: 10 requests
- Configurable in middleware settings

### Error Handling
All endpoints return standardized error responses:
```json
{
  "error": "Error description",
  "status": "error",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Response Formats
Successful responses follow this structure:
```json
{
  "response": "Generated response text",
  "sources": {
    "mode": "rag|web|direct",
    "chunks": ["context1", "context2"],
    "confidence": 0.95
  },
  "status": "success",
  "session_id": "session123"
}
```

## 🔒 Security Considerations

### Data Privacy
- Session data stored locally
- No persistent user data collection
- Medical information processed locally

### API Security
- Rate limiting implemented
- Input validation on all endpoints
- File upload restrictions

### Production Deployment
- Use HTTPS in production
- Implement proper authentication
- Configure firewall rules
- Regular security updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Keep functions focused and small

### Testing
- Add unit tests for new features
- Test API endpoints thoroughly
- Validate with different data types

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **FastAPI**: For the web framework
- **Pinecone**: For vector database services
- **OpenAI CLIP**: For image understanding
- **PaddleOCR**: For text recognition
- **Ollama**: For local LLM inference

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review API documentation at `/docs`

## 🗺️ Roadmap

### Upcoming Features
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Mobile app
- [ ] Advanced analytics dashboard
- [ ] Integration with EHR systems
- [ ] Telemedicine features

### Performance Improvements
- [ ] Distributed processing
- [ ] Advanced caching strategies
- [ ] Model quantization
- [ ] Edge deployment options

---

**AroBot** - Empowering healthcare with intelligent AI assistance. 🏥✨ 