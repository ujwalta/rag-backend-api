"""
Custom RAG service for conversational queries with interview booking detection.
Implements multi-turn conversation handling without using RetrievalQAChain.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import json

from ..services.embeddings import get_embedding_service
from ..services.vector_db import VectorDBFactory, get_collection_name, SearchResult
from ..services.redis_service import get_redis_service
from ..models.schemas import ChatMessage
from ..core.config import settings


class RAGService:
    """Custom RAG service for conversational queries."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBFactory.get_vector_db()
        self.redis_service = get_redis_service()
        self.collection_name = get_collection_name()
    
    async def process_query(
        self,
        session_id: str,
        query: str,
        use_rag: bool = True,
        top_k: int = 5
    ) -> Tuple[str, List[SearchResult], Optional[Dict[str, Any]]]:
        """
        Process a conversational query with RAG.
        
        Args:
            session_id: Unique session identifier
            query: User query
            use_rag: Whether to use RAG for context
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (response, sources, booking_info)
        """
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=query,
            timestamp=datetime.utcnow()
        )
        await self.redis_service.add_message(session_id, user_message)
        
        # Check for booking intent
        booking_info = self._detect_booking_intent(query)
        
        # Retrieve relevant context if RAG is enabled
        sources = []
        context = ""
        
        if use_rag:
            sources = await self._retrieve_context(query, top_k)
            context = self._format_context(sources)
        
        # Get conversation history
        history = await self.redis_service.get_conversation_history(
            session_id,
            max_messages=10  # Last 10 messages for context
        )
        
        # Generate response
        response = await self._generate_response(
            query=query,
            context=context,
            history=history[:-1],  # Exclude the current message
            booking_info=booking_info
        )
        
        # Add assistant response to history
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.utcnow()
        )
        await self.redis_service.add_message(session_id, assistant_message)
        
        return response, sources, booking_info
    
    async def _retrieve_context(
        self,
        query: str,
        top_k: int
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents from vector database.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search vector database
        results = await self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result.score >= settings.SIMILARITY_THRESHOLD
        ]
        
        return filtered_results
    
    def _format_context(self, sources: List[SearchResult]) -> str:
        """
        Format retrieved sources into context string.
        
        Args:
            sources: List of search results
            
        Returns:
            Formatted context string
        """
        if not sources:
            return ""
        
        context_parts = ["Relevant information from documents:\n"]
        
        for i, source in enumerate(sources, 1):
            context_parts.append(f"\n[Source {i}]")
            context_parts.append(source.content)
            if source.metadata.get("filename"):
                context_parts.append(f"(from {source.metadata['filename']})")
        
        return "\n".join(context_parts)
    
    def _detect_booking_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the query contains interview booking information.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with booking details if detected, None otherwise
        """
        query_lower = query.lower()
        
        # Check for booking keywords
        booking_keywords = [
            "book", "schedule", "appointment", "interview",
            "meeting", "reserve", "slot", "time"
        ]
        
        has_booking_intent = any(keyword in query_lower for keyword in booking_keywords)
        
        if not has_booking_intent:
            return None
        
        # Extract information using regex patterns
        booking_info = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        if emails:
            booking_info["email"] = emails[0]
        
        # Extract name (simplified - looks for capitalized words before email or "name is")
        name_patterns = [
            r"(?:name is|i'm|i am|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:here|at)"
        ]
        
        for pattern in name_patterns:
            names = re.findall(pattern, query)
            if names:
                booking_info["name"] = names[0]
                break
        
        # Extract date (simplified patterns)
        date_patterns = [
            r"(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?",
            r"(?:on\s+)?(tomorrow|today|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))"
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            if dates:
                booking_info["date"] = dates[0] if isinstance(dates[0], str) else dates[0][0]
                break
        
        # Extract time
        time_patterns = [
            r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))",
            r"(?:at\s+)?(\d{1,2}\s*(?:AM|PM|am|pm))",
            r"(?:at\s+)?(\d{1,2}:\d{2})"
        ]
        
        for pattern in time_patterns:
            times = re.findall(pattern, query, re.IGNORECASE)
            if times:
                booking_info["time"] = times[0]
                break
        
        # Only return if we found at least 2 pieces of information
        if len(booking_info) >= 2:
            booking_info["detected"] = True
            booking_info["confidence"] = "medium" if len(booking_info) >= 3 else "low"
            return booking_info
        
        return None
    
    async def _generate_response(
        self,
        query: str,
        context: str,
        history: List[ChatMessage],
        booking_info: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate response using context and conversation history.
        
        Args:
            query: User query
            context: Retrieved context from documents
            history: Conversation history
            booking_info: Detected booking information
            
        Returns:
            Generated response
        """
        # Build prompt
        prompt_parts = []
        
        # System instructions
        prompt_parts.append(
            "You are a helpful AI assistant. "
            "Answer questions based on the provided context and conversation history. "
            "Be concise and informative."
        )
        
        # Add context if available
        if context:
            prompt_parts.append(f"\n\n{context}")
        
        # Add conversation history
        if history:
            prompt_parts.append("\n\nConversation history:")
            for msg in history[-5:]:  # Last 5 messages
                prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
        
        # Add current query
        prompt_parts.append(f"\n\nUser: {query}")
        
        # Handle booking detection
        if booking_info:
            prompt_parts.append(
                "\n\nNote: The user appears to be trying to book an interview. "
                "If you have enough information (name, email, date, time), "
                "confirm the booking details. Otherwise, ask for missing information."
            )
            
            if booking_info.get("detected"):
                info_str = ", ".join(
                    f"{k}: {v}" 
                    for k, v in booking_info.items() 
                    if k not in ["detected", "confidence"]
                )
                prompt_parts.append(f"Detected information: {info_str}")
        
        prompt_parts.append("\n\nAssistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # For this implementation, we'll use a simple rule-based response
        # In production, you would call an LLM API here
        response = self._generate_simple_response(query, context, booking_info)
        
        return response
    
    def _generate_simple_response(
        self,
        query: str,
        context: str,
        booking_info: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a simple rule-based response.
        Replace this with actual LLM API call in production.
        
        Args:
            query: User query
            context: Retrieved context
            booking_info: Detected booking information
            
        Returns:
            Generated response
        """
        # Handle booking intent
        if booking_info and booking_info.get("detected"):
            required_fields = ["name", "email", "date", "time"]
            missing_fields = [
                field for field in required_fields 
                if field not in booking_info
            ]
            
            if not missing_fields:
                return (
                    f"Great! I'd be happy to help you book an interview. "
                    f"Let me confirm the details:\n"
                    f"Name: {booking_info.get('name')}\n"
                    f"Email: {booking_info.get('email')}\n"
                    f"Date: {booking_info.get('date')}\n"
                    f"Time: {booking_info.get('time')}\n\n"
                    f"Please confirm if these details are correct."
                )
            else:
                provided = [k for k in required_fields if k in booking_info]
                response = "I'd be happy to help you schedule an interview. "
                
                if provided:
                    response += "I have the following information:\n"
                    for field in provided:
                        response += f"- {field.capitalize()}: {booking_info[field]}\n"
                
                response += f"\nTo complete the booking, I still need: {', '.join(missing_fields)}."
                return response
        
        # Handle regular query with context
        if context:
            return (
                f"Based on the available documents, here's what I found:\n\n"
                f"{context[:500]}...\n\n"
                f"Is there anything specific you'd like to know more about?"
            )
        
        # Fallback response
        return (
            "I'd be happy to help! However, I don't have specific information "
            "about that in my current knowledge base. Could you provide more details "
            "or ask about something else?"
        )