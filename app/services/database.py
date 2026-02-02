"""
Database service for managing SQL database connections and operations.
"""
from typing import AsyncGenerator, Optional, List
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import select, delete
from contextlib import asynccontextmanager

from app.core.config import settings
from app.models.database import Base, Document, DocumentChunk, InterviewBooking, ChatSession


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        """Initialize database service."""
        self.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            future=True
        )
        
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.
        
        Yields:
            AsyncSession instance
        """
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        chunking_strategy: str,
        chunk_count: int,
        vector_ids: List[str],
        metadata: dict = None
    ) -> Document:
        """Create a new document record."""
        async with self.get_session() as session:
            document = Document(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                chunking_strategy=chunking_strategy,
                chunk_count=chunk_count,
                vector_ids=vector_ids,
                metadata=metadata or {}
            )
            
            session.add(document)
            await session.flush()
            await session.refresh(document)
            
            return document
    
    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get a document by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()
    
    async def create_document_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        vector_id: str,
        metadata: dict = None
    ) -> DocumentChunk:
        """Create a document chunk record."""
        async with self.get_session() as session:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                vector_id=vector_id,
                metadata=metadata or {}
            )
            
            session.add(chunk)
            await session.flush()
            await session.refresh(chunk)
            
            return chunk
    
    async def create_booking(
        self,
        name: str,
        email: str,
        date: str,
        time: str,
        conversation_id: Optional[str] = None,
        additional_info: dict = None
    ) -> InterviewBooking:
        """Create a new interview booking."""
        async with self.get_session() as session:
            booking = InterviewBooking(
                name=name,
                email=email,
                date=date,
                time=time,
                conversation_id=conversation_id,
                additional_info=additional_info or {},
                status="pending"
            )
            
            session.add(booking)
            await session.flush()
            await session.refresh(booking)
            
            return booking
    
    async def get_booking(self, booking_id: int) -> Optional[InterviewBooking]:
        """Get a booking by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                select(InterviewBooking).where(InterviewBooking.id == booking_id)
            )
            return result.scalar_one_or_none()
    
    async def update_booking(
        self,
        booking_id: int,
        **updates
    ) -> Optional[InterviewBooking]:
        """Update a booking."""
        async with self.get_session() as session:
            result = await session.execute(
                select(InterviewBooking).where(InterviewBooking.id == booking_id)
            )
            booking = result.scalar_one_or_none()
            
            if booking:
                for key, value in updates.items():
                    if hasattr(booking, key) and value is not None:
                        setattr(booking, key, value)
                
                await session.flush()
                await session.refresh(booking)
            
            return booking
    
    async def get_bookings_by_email(self, email: str) -> List[InterviewBooking]:
        """Get all bookings for an email."""
        async with self.get_session() as session:
            result = await session.execute(
                select(InterviewBooking)
                .where(InterviewBooking.email == email)
                .order_by(InterviewBooking.created_at.desc())
            )
            return list(result.scalars().all())
    
    async def create_or_update_chat_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: dict = None
    ) -> ChatSession:
        """Create or update a chat session."""
        async with self.get_session() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = result.scalar_one_or_none()
            
            if chat_session:
                chat_session.message_count += 1
                if metadata:
                    chat_session.metadata = metadata
            else:
                chat_session = ChatSession(
                    session_id=session_id,
                    user_id=user_id,
                    message_count=1,
                    metadata=metadata or {}
                )
                session.add(chat_session)
            
            await session.flush()
            await session.refresh(chat_session)
            
            return chat_session


# Global database service instance
_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """
    Get or create the global database service instance.
    
    Returns:
        DatabaseService instance
    """
    global _db_service
    
    if _db_service is None:
        _db_service = DatabaseService()
    
    return _db_service