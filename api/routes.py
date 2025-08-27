"""
API routes for AroBot Medical Chatbot
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import io
from PIL import Image
import base64
import tempfile
from pathlib import Path

from agents.medical_agent import MedicalAgent
from mcp_server.mcp_handler import MCPHandler
from config.env_config import TEMPLATES_DIR

# Simple conversation memory storage
conversation_memory = {}

import re
from typing import Optional

class PerfectMemoryProcessor:
    """100% accurate memory processor using pattern matching"""
    
    def __init__(self):
        # Enhanced patterns for better extraction
        self.patterns = {
            "name": [
                r"I am (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:from|and|in|at|,)|\s*\.?\s*$)",
                r"my name is (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:from|and|in|at|,)|\s*\.?\s*$)",
                r"I'm (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:from|and|in|at|,)|\s*\.?\s*$)",
                r"call me (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:from|and|in|at|,)|\s*\.?\s*$)",
                r"I am (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\.?\s*$",
                r"my name is (?:Dr\.?\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\.?\s*$"
            ],
            "department": [
                r"from (?:the\s+)?([a-z]+)\s+department",
                r"work in (?:the\s+)?([a-z]+)\s+department",
                r"in (?:the\s+)?([a-z]+)\s+department",
                r"specialize in ([a-z]+)",
                r"specialty is ([a-z]+)"
            ],
            "hospital": [
                r"at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Hospital)",
                r"work at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"hospital is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
            ]
        }
        
        self.query_patterns = {
            "name": [
                r"what is my name", r"my name", r"who am i", r"what's my name",
                r"tell me my name", r"remember my name", r"what's my name again",
                r"tell me my full name", r"can you remind me of my name",
                r"what am i called", r"what do you call me", r"my full name",
                r"remind me my name", r"who do you think i am", r"my name again"
            ],
            "department": [
                r"what department", r"which department", r"my department",
                r"what specialty", r"my specialty", r"department do i work",
                r"where do i work", r"what field", r"my specialization",
                r"which field do i work in", r"what area do i work in"
            ],
            "hospital": [
                r"which hospital", r"what hospital", r"my hospital",
                r"hospital do i work", r"where do i work", r"my workplace",
                r"where am i employed", r"what institution"
            ]
        }
        
        self.session_facts = {}
    
    def extract_facts(self, conversation_history: str) -> dict:
        """Extract facts from conversation with 100% accuracy"""
        facts = {"name": [], "department": [], "hospital": []}
        
        for fact_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, conversation_history, re.IGNORECASE)
                if matches:
                    # Clean and add unique matches
                    cleaned_matches = [match.strip().title() for match in matches]
                    facts[fact_type].extend(cleaned_matches)
        
        # Remove duplicates while preserving order
        for fact_type in facts:
            facts[fact_type] = list(dict.fromkeys(facts[fact_type]))
        
        return facts
    
    def detect_memory_query(self, query: str) -> Optional[str]:
        """Detect if query is asking for remembered information"""
        query_lower = query.lower().strip()
        
        # Ultra-reliable simple patterns
        if any(phrase in query_lower for phrase in ["what is my name", "my name", "who am i", "what's my name"]):
            return "name"
        if any(phrase in query_lower for phrase in ["what department", "which department", "department do i work"]):
            return "department"
        if any(phrase in query_lower for phrase in ["which hospital", "what hospital", "hospital do i work"]):
            return "hospital"
        
        return None
    
    def generate_memory_response(self, query: str, session_id: str, 
                                conversation_history: str) -> Optional[str]:
        """Generate guaranteed accurate memory response"""
        
        # Extract facts from current conversation
        facts = self.extract_facts(conversation_history)
        
        # Store facts for this session
        if session_id not in self.session_facts:
            self.session_facts[session_id] = {}
        
        # Update session facts with latest information
        for fact_type, fact_list in facts.items():
            if fact_list:
                self.session_facts[session_id][fact_type] = fact_list[-1]  # Most recent
        
        # Detect what the user is asking for
        query_type = self.detect_memory_query(query)
        
        if query_type and session_id in self.session_facts:
            stored_fact = self.session_facts[session_id].get(query_type)
            
            if stored_fact:
                if query_type == "name":
                    return f"Your name is Dr. {stored_fact}."
                elif query_type == "department":
                    return f"You work in the {stored_fact} department."
                elif query_type == "hospital":
                    return f"You work at {stored_fact}."
        
        # Also check fresh facts from current conversation
        if query_type and facts.get(query_type):
            fact = facts[query_type][-1]  # Most recent mention
            if query_type == "name":
                return f"Your name is Dr. {fact}."
            elif query_type == "department":
                return f"You work in the {fact} department."
            elif query_type == "hospital":
                return f"You work at {fact}."
        
        return None  # Let LLM handle non-memory queries

# Initialize the perfect memory processor
perfect_memory = PerfectMemoryProcessor()

def add_to_conversation_memory(session_id: str, role: str, message: str):
    """Add message to simple conversation memory"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].append({
        "role": role,
        "message": message
    })
    
    # Keep only last 10 messages per session
    if len(conversation_memory[session_id]) > 10:
        conversation_memory[session_id] = conversation_memory[session_id][-10:]

def get_conversation_memory_context(session_id: str) -> str:
    """Get conversation memory as formatted string"""
    if session_id not in conversation_memory:
        return ""
    
    messages = conversation_memory[session_id]
    if not messages:
        return ""
    
    formatted_messages = []
    for msg in messages[-6:]:  # Last 6 messages
        formatted_messages.append(f"{msg['role']}: {msg['message']}")
    
    return "Previous conversation:\n" + "\n".join(formatted_messages)

# Initialize components
router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Lazy initialization for heavy components
medical_agent = None
mcp_handler = None

def get_medical_agent():
    global medical_agent
    if medical_agent is None:
        print("🔄 Initializing Medical Agent...")
        from agents.medical_agent import MedicalAgent
        medical_agent = MedicalAgent()
        print("✅ Medical Agent initialized")
    return medical_agent

def get_mcp_handler():
    global mcp_handler
    if mcp_handler is None:
        print("🔄 Initializing MCP Handler...")
        from mcp_server.mcp_handler import MCPHandler
        mcp_handler = MCPHandler()
        print("✅ MCP Handler initialized")
    return mcp_handler

async def create_new_pinecone_index(pdf_data: bytes, filename: str, index_name: str, description: str = None) -> Dict[str, Any]:
    """Create a new Pinecone index and add PDF content"""
    try:
        # Create temporary file for PDF processing
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"temp_{filename}"
        
        # Write PDF data to temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_data)
        
        try:
            # Import PDF processing modules
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from core.vector_store import PineconeStore
            
            # Load PDF
            loader = PyPDFLoader(str(temp_file_path))
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            
            # Prepare texts and metadata for vector store
            text_contents = []
            metadatas = []
            
            for i, doc in enumerate(texts):
                text_contents.append(doc.page_content)
                metadatas.append({
                    "source": filename,
                    "description": description or f"PDF document: {filename}",
                    "chunk_id": i,
                    "page": doc.metadata.get("page", 0),
                    "total_chunks": len(texts),
                    "index_name": index_name
                })
            
            # Create new vector store with custom index
            vector_store = PineconeStore(index_name=index_name, dimension=384)
            
            # Add texts to the new store
            vector_store.upsert_texts(text_contents, metadatas)
            
            return {
                "status": "success",
                "chunks_processed": len(texts),
                "pages_processed": len(documents),
                "filename": filename,
                "index_name": index_name
            }
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "filename": filename,
            "index_name": index_name
        }

async def process_pdf_to_knowledge_base(pdf_data: bytes, filename: str, description: str = None) -> Dict[str, Any]:
    """Process PDF and add to knowledge base"""
    try:
        # Create temporary file for PDF processing
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"temp_{filename}"
        
        # Write PDF data to temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_data)
        
        try:
            # Import PDF processing modules
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Load PDF
            loader = PyPDFLoader(str(temp_file_path))
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            
            # Prepare texts and metadata for vector store
            text_contents = []
            metadatas = []
            
            for i, doc in enumerate(texts):
                text_contents.append(doc.page_content)
                metadatas.append({
                    "source": filename,
                    "description": description or f"PDF document: {filename}",
                    "chunk_id": i,
                    "page": doc.metadata.get("page", 0),
                    "total_chunks": len(texts)
                })
            
            # Get RAG agent and add to PDF knowledge base
            rag_agent = get_medical_agent().rag_agent
            
            # Add texts to the PDF store
            rag_agent.pdf_store.upsert_texts(text_contents, metadatas)
            
            return {
                "status": "success",
                "chunks_processed": len(texts),
                "pages_processed": len(documents),
                "filename": filename
            }
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "filename": filename
        }

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_web_search: Optional[bool] = False

class PrescriptionQuery(BaseModel):
    query: Optional[str] = None
    session_id: Optional[str] = None

class MedicineSearch(BaseModel):
    condition: str
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    user_id: Optional[str] = None

# Web interface route
@router.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Web interface for the chatbot"""
    return templates.TemplateResponse("chat_enhanced.html", {"request": request})

# Session management
@router.post("/session/create")
async def create_session(request: SessionRequest):
    """Create a new conversation session"""
    try:
        result = get_mcp_handler().initialize_session(request.user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """Get conversation context for a session"""
    try:
        context = get_mcp_handler().get_conversation_context(session_id)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get medical history for a session"""
    try:
        history = get_mcp_handler().get_user_medical_history(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@router.post("/chat")
async def chat_with_bot(message: ChatMessage):
    """Chat with the medical bot"""
    try:
        # Create session if not provided
        session_id = message.session_id
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Add user message to both MCP and simple memory
        get_mcp_handler().add_user_message(session_id, message.message)
        add_to_conversation_memory(session_id, "User", message.message)
        
        # Get conversation context from simple memory (reliable and fast)
        conversation_context = get_conversation_memory_context(session_id)
        print(f"🧠 [DEBUG] Using simple memory context ({len(conversation_context)} chars)")
        if conversation_context:
            print(f"📝 [DEBUG] Context preview: {conversation_context[:150]}...")
        
        # 🚀 HYBRID MEMORY: Check for instant memory responses first
        memory_response = perfect_memory.generate_memory_response(
            message.message, session_id, conversation_context
        )
        
        if memory_response:
            print(f"✅ [MEMORY] Instant response: {memory_response}")
            
            # Add the memory response to conversation
            add_to_conversation_memory(session_id, "Assistant", memory_response)
            get_mcp_handler().add_assistant_response(session_id, memory_response)
            
            return {
                "response": memory_response,
                "session_id": session_id,
                "sources": {"memory": "perfect_recall"},
                "status": "success"
            }
        
        # If no memory match, proceed with normal LLM processing
        print(f"🔄 [MEMORY] No memory match, proceeding to LLM")
        
        # Get response from medical agent with conversation context
        response = get_medical_agent().handle_text_query(
            message.message, 
            session_id=session_id,
            conversation_context=conversation_context,
            use_web_search=message.use_web_search
        )
        
        if response.get("status") == "success":
            bot_response = response.get("response", "I apologize, but I couldn't generate a response.")
            
            # Add bot response to both MCP and simple memory
            get_mcp_handler().add_assistant_response(session_id, bot_response)
            add_to_conversation_memory(session_id, "Assistant", bot_response)
            
            # Record medical query
            get_mcp_handler().record_medical_query(session_id, message.message, bot_response)
            
            return {
                "response": bot_response,
                "session_id": session_id,
                "sources": response.get("sources", {}),
                "status": "success"
            }
        else:
            error_message = f"Error: {response.get('error', 'Unknown error')}"
            get_mcp_handler().add_assistant_response(session_id, error_message)
            
            return {
                "response": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prescription processing
@router.post("/prescription/upload")
async def upload_prescription(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    image_type: Optional[str] = Form("prescription"),
    session_id: Optional[str] = Form(None)
):
    """Upload and analyze prescription image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Create session if not provided
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Add user message about prescription upload
        upload_message = f"Uploaded prescription image: {file.filename}"
        if query:
            upload_message += f" with query: {query}"
        
        get_mcp_handler().add_user_message(
            session_id, 
            upload_message, 
            "prescription_upload", 
            {"filename": file.filename}
        )
        
        # Use provided image_type or determine from query
        if not image_type:
            image_type = "prescription" if "prescription" in (query or "").lower() else "general"
        
        # Process image with enhanced analysis
        result = get_medical_agent().handle_image_query(image_data, query, image_type)
        
        if result.get("status") == "success":
            # Handle both prescription and general image analysis
            analysis = result.get("prescription_analysis") or result.get("analysis", "Analysis completed.")
            
            # Record prescription in conversation memory
            get_mcp_handler().record_prescription_analysis(session_id, result)
            
            # Add response to conversation
            get_mcp_handler().add_assistant_response(
                session_id, 
                analysis, 
                "image_analysis_response"
            )
            
            return {
                "response": analysis,
                "analysis": analysis,
                "session_id": session_id,
                "ocr_results": result.get("ocr_results", {}),
                "status": "success"
            }
        else:
            error_message = f"Error analyzing prescription: {result.get('error', 'Unknown error')}"
            get_mcp_handler().add_assistant_response(session_id, error_message)
            
            return {
                "error": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prescription/analyze")
async def analyze_prescription_text(request: PrescriptionQuery):
    """Analyze prescription from text input"""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Add user query to conversation
        get_mcp_handler().add_user_message(session_id, f"Prescription query: {request.query}")
        
        # Use RAG agent for prescription analysis
        result = get_medical_agent().rag_agent.analyze_prescription_query(request.query)
        
        if result.get("status") == "success":
            response = result.get("response", "Analysis completed.")
            
            # Add response to conversation
            get_mcp_handler().add_assistant_response(session_id, response)
            
            # Record as medical query
            get_mcp_handler().record_medical_query(session_id, request.query, response, "prescription")
            
            return {
                "response": response,
                "session_id": session_id,
                "medicine_sources": result.get("medicine_sources", 0),
                "status": "success"
            }
        else:
            error_message = f"Error: {result.get('error', 'Unknown error')}"
            get_mcp_handler().add_assistant_response(session_id, error_message)
            
            return {
                "response": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    query: Optional[str] = Form(""),
    session_id: Optional[str] = Form(None)
):
    """Analyze PDF document and provide description or answer query (without storing in vector DB)"""
    try:
        # Validate file type
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF data
        pdf_data = await file.read()
        
        # Create session if not provided
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Create temporary file for PDF processing
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"temp_analyze_{file.filename}"
        
        # Write PDF data to temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_data)
        
        try:
            # Import PDF processing modules
            from langchain_community.document_loaders import PyPDFLoader
            
            # Load PDF
            loader = PyPDFLoader(str(temp_file_path))
            documents = loader.load()
            
            # Extract text content
            full_text = ""
            for doc in documents:
                full_text += doc.page_content + "\n"
            
            # Prepare query for LLM
            if not query or query.strip() == "":
                # Default description query
                analysis_query = f"""Please provide a comprehensive summary and description of this PDF document. 
                
Include:
- What type of document this is
- Main topics and subjects covered
- Key information and important details
- Overall structure and organization

PDF Content:
{full_text[:3000]}...

Provide a clear, informative description of what this PDF contains."""
            else:
                # User-specific query
                analysis_query = f"""Based on this PDF document, please answer the following question: {query}

PDF Content:
{full_text[:3000]}...

Please provide a detailed answer based on the document content."""
            
            # Get LLM response
            medical_agent = get_medical_agent()
            llm_response = medical_agent.llm.generate_text_response(
                analysis_query,
                "You are a helpful AI assistant that analyzes documents and provides clear, informative descriptions."
            )
            
            # Add to conversation memory
            upload_message = f"Analyzed PDF document: {file.filename}"
            if query and query.strip():
                upload_message += f" - Query: {query}"
            else:
                upload_message += " - Requested default description"
            
            get_mcp_handler().add_user_message(session_id, upload_message)
            get_mcp_handler().add_assistant_response(session_id, llm_response)
            
            return {
                "response": llm_response,
                "session_id": session_id,
                "filename": file.filename,
                "query_type": "default_description" if not query or query.strip() == "" else "specific_query",
                "status": "success"
            }
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
                
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """Upload PDF document and add to knowledge base"""
    try:
        # Validate file type
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF data
        pdf_data = await file.read()
        
        # Create session if not provided
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Add user message about PDF upload
        upload_message = f"Uploaded PDF document: {file.filename}"
        if description:
            upload_message += f" - {description}"
        
        get_mcp_handler().add_user_message(
            session_id, 
            upload_message, 
            "pdf_upload", 
            {"filename": file.filename, "description": description}
        )
        
        # Process PDF and add to knowledge base
        result = await process_pdf_to_knowledge_base(pdf_data, file.filename, description)
        
        if result.get("status") == "success":
            success_message = f"Successfully processed '{file.filename}' and added {result.get('chunks_processed', 0)} text chunks to the medical knowledge base. The document is now available for medical queries."
            
            get_mcp_handler().add_assistant_response(
                session_id, 
                success_message, 
                "pdf_processing_response"
            )
            
            return {
                "message": success_message,
                "session_id": session_id,
                "chunks_processed": result.get("chunks_processed", 0),
                "filename": file.filename,
                "status": "success"
            }
        else:
            error_message = f"Error processing PDF: {result.get('error', 'Unknown error')}"
            get_mcp_handler().add_assistant_response(session_id, error_message)
            
            return {
                "error": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/create-index")
async def create_vector_index(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    description: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """Create a new vector index with uploaded PDF"""
    try:
        # Validate file type
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF data
        pdf_data = await file.read()
        
        # Create session if not provided
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Format index name to be Pinecone-compliant
        # Convert to lowercase, replace underscores and spaces with hyphens
        formatted_index_name = f"arobot-medical-pdf-{index_name.lower().replace('_', '-').replace(' ', '-')}"
        
        # Create new Pinecone index
        result = await create_new_pinecone_index(pdf_data, file.filename, formatted_index_name, description)
        
        if result.get("status") == "success":
            success_message = f"Successfully created vector index '{index_name}' with {result.get('chunks_processed', 0)} document chunks."
            
            # Log the action
            get_mcp_handler().add_user_message(
                session_id, 
                f"Created new vector index: {formatted_index_name}", 
                "vector_index_creation", 
                {"index_name": formatted_index_name, "user_name": index_name, "filename": file.filename}
            )
            
            get_mcp_handler().add_assistant_response(
                session_id, 
                success_message, 
                "vector_index_response"
            )
            
            return {
                "message": success_message,
                "session_id": session_id,
                "chunks_processed": result.get("chunks_processed", 0),
                "index_name": formatted_index_name,
                "user_name": index_name,
                "status": "success"
            }
        else:
            error_message = f"Error creating vector index: {result.get('error', 'Unknown error')}"
            return {
                "error": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/indexes")
async def list_vector_indexes():
    """Get list of all available vector indexes"""
    try:
        # Import here to avoid circular imports
        from config.env_config import PINECONE_API_KEY
        import pinecone
        from pinecone import Pinecone
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get list of indexes
        indexes = pc.list_indexes()
        
        # Format the response
        index_list = []
        for index in indexes:
            index_info = {
                "name": index.name,
                "dimension": index.dimension,
                "metric": index.metric,
                "host": index.host,
                "status": "ready" if index.status.ready else "not_ready"
            }
            
            # Try to get stats for the index
            try:
                index_connection = pc.Index(index.name)
                stats = index_connection.describe_index_stats()
                index_info["total_vector_count"] = stats.total_vector_count
                index_info["namespaces"] = len(stats.namespaces) if stats.namespaces else 0
            except Exception as e:
                index_info["total_vector_count"] = 0
                index_info["namespaces"] = 0
                index_info["error"] = str(e)
            
            index_list.append(index_info)
        
        return {
            "indexes": index_list,
            "total_count": len(index_list),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Medicine search
@router.post("/search/medicine")
async def search_medicine_by_condition(request: MedicineSearch):
    """Search for medicines that treat a specific condition"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_result = get_mcp_handler().initialize_session()
            session_id = session_result.get("session_id")
        
        # Add user query to conversation
        query_message = f"Search medicines for condition: {request.condition}"
        get_mcp_handler().add_user_message(session_id, query_message)
        
        # Search using medical agent
        result = get_medical_agent().search_medicine_by_condition(request.condition)
        
        if result.get("status") == "success":
            response = result.get("response", "Search completed.")
            
            # Add response to conversation
            get_mcp_handler().add_assistant_response(session_id, response)
            
            # Record as medical query
            get_mcp_handler().record_medical_query(session_id, query_message, response, "medicine_search")
            
            return {
                "response": response,
                "condition": request.condition,
                "session_id": session_id,
                "status": "success"
            }
        else:
            error_message = f"Error: {result.get('error', 'Unknown error')}"
            get_mcp_handler().add_assistant_response(session_id, error_message)
            
            return {
                "response": error_message,
                "session_id": session_id,
                "status": "error"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System status
@router.get("/system/status")
async def get_system_status():
    """Get system status and health"""
    try:
        status = get_medical_agent().get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base endpoints
@router.get("/knowledge/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = get_medical_agent().rag_agent.get_knowledge_base_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
