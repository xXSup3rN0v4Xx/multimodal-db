"""
Conversation Management Router
Store and retrieve conversation history.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..dependencies import get_db

router = APIRouter(prefix="/conversations", tags=["conversations"])

class MessageCreate(BaseModel):
    role: str
    content: str

class Message(BaseModel):
    id: str
    agent_id: str
    role: str
    content: str
    timestamp: datetime

@router.post("/{agent_id}/messages")
async def add_message(agent_id: str, message: MessageCreate, db=Depends(get_db)):
    """Add a message to agent's conversation history."""
    msg_id = db.add_message(agent_id, message.role, message.content)
    return {"message_id": msg_id, "status": "stored"}

@router.get("/{agent_id}/messages")
async def get_messages(agent_id: str, limit: int = 50, db=Depends(get_db)):
    """Get conversation history for an agent."""
    messages = db.get_messages(agent_id, limit=limit)
    return {"agent_id": agent_id, "messages": messages, "count": len(messages)}

@router.delete("/{agent_id}/messages")
async def clear_messages(agent_id: str, db=Depends(get_db)):
    """Clear all messages for an agent."""
    try:
        db.clear_messages(agent_id)
        return {"status": "cleared", "agent_id": agent_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
