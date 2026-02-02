"""
Redis service for managing chat conversation history and session state.
"""
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import redis.asyncio as aioredis

from ..core.config import settings
from ..models.schemas import ChatMessage


class RedisService:
    """Service for managing chat memory using Redis."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client: Optional[aioredis.Redis] = None
        self.ttl = settings.REDIS_CHAT_TTL
    
    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
    
    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"chat:session:{session_id}"
    
    def _get_metadata_key(self, session_id: str) -> str:
        """Generate Redis key for session metadata."""
        return f"chat:metadata:{session_id}"
    
    async def add_message(
        self,
        session_id: str,
        message: ChatMessage
    ) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Unique session identifier
            message: Chat message to add
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        
        # Serialize message
        message_data = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat() if message.timestamp else datetime.utcnow().isoformat()
        }
        
        # Add to list
        await self.redis_client.rpush(key, json.dumps(message_data))
        
        # Set expiration
        await self.redis_client.expire(key, self.ttl)
        
        # Update metadata
        await self._update_metadata(session_id)
    
    async def get_conversation_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            max_messages: Maximum number of messages to retrieve (most recent)
            
        Returns:
            List of chat messages
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        
        # Get messages
        if max_messages:
            messages_data = await self.redis_client.lrange(key, -max_messages, -1)
        else:
            messages_data = await self.redis_client.lrange(key, 0, -1)
        
        # Deserialize messages
        messages = []
        for msg_json in messages_data:
            msg_dict = json.loads(msg_json)
            messages.append(
                ChatMessage(
                    role=msg_dict["role"],
                    content=msg_dict["content"],
                    timestamp=datetime.fromisoformat(msg_dict["timestamp"])
                )
            )
        
        return messages
    
    async def clear_conversation(self, session_id: str) -> None:
        """
        Clear all messages for a session.
        
        Args:
            session_id: Unique session identifier
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        metadata_key = self._get_metadata_key(session_id)
        
        await self.redis_client.delete(key)
        await self.redis_client.delete(metadata_key)
    
    async def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        return await self.redis_client.exists(key) > 0
    
    async def get_message_count(self, session_id: str) -> int:
        """
        Get the number of messages in a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Number of messages
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        return await self.redis_client.llen(key)
    
    async def _update_metadata(self, session_id: str) -> None:
        """Update session metadata."""
        metadata_key = self._get_metadata_key(session_id)
        
        metadata = {
            "last_activity": datetime.utcnow().isoformat(),
            "message_count": await self.get_message_count(session_id)
        }
        
        await self.redis_client.set(
            metadata_key,
            json.dumps(metadata),
            ex=self.ttl
        )
    
    async def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Metadata dictionary or None
        """
        await self.connect()
        
        metadata_key = self._get_metadata_key(session_id)
        metadata_json = await self.redis_client.get(metadata_key)
        
        if metadata_json:
            return json.loads(metadata_json)
        
        return None
    
    async def extend_session_ttl(self, session_id: str) -> None:
        """
        Extend the TTL of a session.
        
        Args:
            session_id: Unique session identifier
        """
        await self.connect()
        
        key = self._get_session_key(session_id)
        metadata_key = self._get_metadata_key(session_id)
        
        await self.redis_client.expire(key, self.ttl)
        await self.redis_client.expire(metadata_key, self.ttl)


# Global Redis service instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """
    Get or create the global Redis service instance.
    
    Returns:
        RedisService instance
    """
    global _redis_service
    
    if _redis_service is None:
        _redis_service = RedisService()
    
    return _redis_service