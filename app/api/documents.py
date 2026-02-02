"""
Document Ingestion API endpoint.
Handles file upload, text extraction, chunking, embedding, and storage.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Literal
import time

from app.models.schemas import DocumentUploadResponse, ErrorResponse
from app.utils.document_processor import DocumentProcessor
from app.utils.chunking import ChunkingStrategyFactory
from app.services.embeddings import get_embedding_service
from app.services.vector_db import VectorDBFactory, get_collection_name
from app.services.database import get_database_service
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description="Upload a PDF or TXT file, extract text, chunk it, generate embeddings, and store in vector database"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or TXT file to upload"),
    chunking_strategy: Literal["fixed_size", "semantic"] = Form(
        default="fixed_size",
        description="Chunking strategy to use"
    ),
    chunk_size: int = Form(
        default=500,
        description="Chunk size (for fixed_size strategy)",
        ge=100,
        le=2000
    ),
    chunk_overlap: int = Form(
        default=50,
        description="Chunk overlap (for fixed_size strategy)",
        ge=0,
        le=500
    )
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    - Validates file type and size
    - Extracts text content
    - Applies selected chunking strategy
    - Generates embeddings
    - Stores in vector database and SQL database
    """
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file
        is_valid, error_msg = DocumentProcessor.validate_file(
            filename=file.filename,
            file_size=file_size,
            max_size_mb=settings.MAX_FILE_SIZE_MB,
            allowed_extensions=settings.ALLOWED_EXTENSIONS
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Extract file type
        file_type = file.filename.split('.')[-1].lower()
        file_ext = f".{file_type}"
        
        # Extract text from document
        try:
            text = DocumentProcessor.extract_text(file_content, file_ext)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from document: {str(e)}"
            )
        
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contains no extractable text"
            )
        
        # Create chunking strategy
        if chunking_strategy == "fixed_size":
            strategy = ChunkingStrategyFactory.create_strategy(
                "fixed_size",
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
        else:
            strategy = ChunkingStrategyFactory.create_strategy(
                "semantic",
                max_chunk_size=1000,
                min_chunk_size=100
            )
        
        # Chunk the text
        chunks = strategy.chunk_text(
            text,
            metadata={"filename": file.filename, "file_type": file_ext}
        )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create chunks from document"
            )
        
        # Generate embeddings
        embedding_service = get_embedding_service()
        
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        # Store in vector database
        vector_db = VectorDBFactory.get_vector_db()
        collection_name = get_collection_name()
        
        # Ensure collection exists
        embedding_dim = embedding_service.get_embedding_dimension()
        await vector_db.create_collection(collection_name, embedding_dim)
        
        # Prepare metadata for vector storage
        vector_metadata = [
            {
                **chunk["metadata"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Upsert vectors
        vector_ids = await vector_db.upsert_vectors(
            collection_name=collection_name,
            vectors=embeddings,
            texts=chunk_texts,
            metadata=vector_metadata
        )
        
        # Store in SQL database
        db_service = get_database_service()
        
        document = await db_service.create_document(
            filename=file.filename,
            file_type=file_ext,
            file_size=file_size,
            chunking_strategy=chunking_strategy,
            chunk_count=len(chunks),
            vector_ids=vector_ids,
            metadata={
                "original_text_length": len(text),
                "chunk_params": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                } if chunking_strategy == "fixed_size" else {}
            }
        )
        
        # Store individual chunks
        for i, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
            await db_service.create_document_chunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk["content"],
                vector_id=vector_id,
                metadata=chunk["metadata"]
            )
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            file_size=document.file_size,
            chunking_strategy=document.chunking_strategy,
            chunk_count=document.chunk_count,
            message=f"Document processed successfully with {len(chunks)} chunks",
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {str(e)}"
        )


@router.get(
    "/{document_id}",
    summary="Get document information",
    description="Retrieve information about a previously uploaded document"
)
async def get_document_info(document_id: int):
    """Get information about a document."""
    db_service = get_database_service()
    
    document = await db_service.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "file_size": document.file_size,
        "chunking_strategy": document.chunking_strategy,
        "chunk_count": document.chunk_count,
        "upload_date": document.upload_date,
        "metadata": document.doc_metadata
    }