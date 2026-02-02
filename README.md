# RAG Backend API - Palm Mind Technology Task

Production-grade REST API backend for Document Ingestion and Conversational RAG with interview booking capabilities.

## Task Requirements Completed ✅

### 1. Document Ingestion API
- ✅ Upload PDF and TXT files
  
- ✅ Text extraction using PyPDF2 and pdfplumber
  
- ✅ Two chunking strategies: Fixed-size and Semantic
  
- ✅ Generate embeddings using local sentence-transformers model\
  
- ✅ Store in Qdrant vector database (also supports Pinecone, Weaviate, Milvus)
  
- ✅ Save metadata in SQLite/PostgreSQL database

### 2. Conversational RAG API
- ✅ Custom RAG implementation (no RetrievalQAChain)
  
- ✅ Redis-based chat memory]
  
- ✅ Multi-turn query handling
  
- ✅ Interview booking detection using LLM patterns
  
- ✅ Store booking information in database

### 3. Constraints Met
- ✅ No FAISS/Chroma used
  
- ✅ No UI (REST API only)
  
- ✅ No RetrievalQAChain
  
- ✅ Clean, modular code
  
- ✅ Industry-standard typing and annotations

## Tech Stack

- **Framework**: FastAPI
  
- **Embeddings**: Sentence Transformers (local model)
  
- **Vector DB**: Qdrant (supports Pinecone, Weaviate, Milvus)
  
- **Chat Memory**: Redis
  
- **Database**: SQLite/PostgreSQL (async)
  
- **Document Processing**: PyPDF2, pdfplumber

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for Qdrant and Redis)

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/rag-backend-api.git
cd rag-backend-api

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d

# Configure environment
cp .env.example .env
# Edit .env as needed

# Run server
uvicorn app.main:app --reload
```

### Access

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Qdrant Dashboard: http://localhost:6333/dashboard

## API Endpoints

### Document Ingestion
- `POST /api/v1/documents/upload` - Upload and process documents
- `GET /api/v1/documents/{id}` - Get document information

### Conversational RAG
- `POST /api/v1/chat/query` - Send conversational query
- `GET /api/v1/chat/history/{session_id}` - Get conversation history
- `DELETE /api/v1/chat/history/{session_id}` - Clear conversation

### Interview Booking
- `POST /api/v1/chat/booking` - Create booking
- `GET /api/v1/chat/booking/{id}` - Get booking info
- `PATCH /api/v1/chat/booking/{id}` - Update booking

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Testing

Detailed testing instructions available in interactive API docs at `/docs`.

## Author

**Ujwalta**  
Submitted for: Palm Mind Technology Technical Assessment  
Date: February 2026
