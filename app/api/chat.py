"""
Conversational RAG API endpoint.
Handles multi-turn conversations with RAG and interview booking.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    BookingRequest,
    BookingResponse,
    BookingUpdate,
    BookingInfo
)
from app.services.rag_service import RAGService
from app.services.redis_service import get_redis_service
from ..services.database import get_database_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="Send a conversational query",
    description="Process a user query with multi-turn conversation support and RAG"
)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """
    Process a conversational query.
    
    - Maintains conversation history using Redis
    - Retrieves relevant context from vector database if RAG is enabled
    - Detects interview booking intent
    - Generates contextual response
    """
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Process query
        response, sources, booking_info = await rag_service.process_query(
            session_id=request.session_id,
            query=request.query,
            use_rag=request.use_rag,
            top_k=request.top_k or 5
        )
        
        # Update chat session in database
        db_service = get_database_service()
        await db_service.create_or_update_chat_session(
            session_id=request.session_id,
            metadata={"last_query": request.query}
        )
        
        # Format sources for response
        formatted_sources = [
            {
                "content": source.content[:200] + "..." if len(source.content) > 200 else source.content,
                "score": round(source.score, 3),
                "metadata": source.metadata
            }
            for source in sources
        ]
        
        return ChatResponse(
            session_id=request.session_id,
            response=response,
            sources=formatted_sources,
            booking_detected=booking_info is not None,
            booking_info=booking_info,
            metadata={
                "use_rag": request.use_rag,
                "sources_count": len(sources)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get(
    "/history/{session_id}",
    response_model=ConversationHistory,
    summary="Get conversation history",
    description="Retrieve the full conversation history for a session"
)
async def get_conversation_history(
    session_id: str,
    max_messages: int = None
) -> ConversationHistory:
    """Get conversation history for a session."""
    try:
        redis_service = get_redis_service()
        
        # Check if session exists
        if not await redis_service.session_exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        # Get history
        messages = await redis_service.get_conversation_history(
            session_id,
            max_messages=max_messages
        )
        
        return ConversationHistory(
            session_id=session_id,
            messages=messages,
            message_count=len(messages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation history: {str(e)}"
        )


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
    description="Clear all messages for a session"
)
async def clear_conversation_history(session_id: str) -> Dict[str, str]:
    """Clear conversation history for a session."""
    try:
        redis_service = get_redis_service()
        
        await redis_service.clear_conversation(session_id)
        
        return {
            "message": f"Conversation history cleared for session {session_id}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing conversation history: {str(e)}"
        )


@router.post(
    "/booking",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create interview booking",
    description="Create a new interview booking with user information"
)
async def create_booking(request: BookingRequest) -> BookingResponse:
    """
    Create a new interview booking.
    
    - Validates booking information
    - Stores in database
    - Associates with conversation session if provided
    """
    try:
        db_service = get_database_service()
        
        # Create booking
        booking = await db_service.create_booking(
            name=request.name,
            email=request.email,
            date=request.date,
            time=request.time,
            conversation_id=request.session_id,
            additional_info=request.additional_info
        )
        
        return BookingResponse(
            booking_id=booking.id,
            name=booking.name,
            email=booking.email,
            date=booking.date,
            time=booking.time,
            status=booking.status,
            created_at=booking.created_at,
            message="Interview booking created successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating booking: {str(e)}"
        )


@router.get(
    "/booking/{booking_id}",
    response_model=BookingInfo,
    summary="Get booking information",
    description="Retrieve information about a specific booking"
)
async def get_booking(booking_id: int) -> BookingInfo:
    """Get booking information by ID."""
    try:
        db_service = get_database_service()
        
        booking = await db_service.get_booking(booking_id)
        
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Booking with ID {booking_id} not found"
            )
        
        return BookingInfo(
            id=booking.id,
            name=booking.name,
            email=booking.email,
            date=booking.date,
            time=booking.time,
            status=booking.status,
            conversation_id=booking.conversation_id,
            created_at=booking.created_at,
            updated_at=booking.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving booking: {str(e)}"
        )


@router.patch(
    "/booking/{booking_id}",
    response_model=BookingInfo,
    summary="Update booking",
    description="Update booking information or status"
)
async def update_booking(
    booking_id: int,
    update: BookingUpdate
) -> BookingInfo:
    """Update an existing booking."""
    try:
        db_service = get_database_service()
        
        # Get updates as dict, excluding None values
        updates = {k: v for k, v in update.dict().items() if v is not None}
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        # Update booking
        booking = await db_service.update_booking(booking_id, **updates)
        
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Booking with ID {booking_id} not found"
            )
        
        return BookingInfo(
            id=booking.id,
            name=booking.name,
            email=booking.email,
            date=booking.date,
            time=booking.time,
            status=booking.status,
            conversation_id=booking.conversation_id,
            created_at=booking.created_at,
            updated_at=booking.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating booking: {str(e)}"
        )