"""
Agent Management Router
CRUD operations for agent configurations.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..dependencies import get_db, get_vector_db
from core import create_corecoder_agent, create_multimodal_agent

router = APIRouter(prefix="/agents", tags=["agents"])

class AgentCreateRequest(BaseModel):
    name: str
    agent_type: str = "corecoder"
    description: Optional[str] = None
    tags: Optional[List[str]] = None

@router.get("/")
async def list_agents(include_full: bool = False, db=Depends(get_db)):
    """
    List all agents. Queries database on every request for real-time updates.
    
    Args:
        include_full: If True, includes complete configuration with prompts and models
    """
    return db.list_agents(include_full_config=include_full)

@router.post("/")
async def create_agent(request: AgentCreateRequest, db=Depends(get_db)):
    """Create new agent."""
    if request.agent_type == "corecoder":
        agent = create_corecoder_agent(request.name)
    elif request.agent_type == "multimodal":
        agent = create_multimodal_agent(request.name)
    else:
        raise HTTPException(status_code=400, detail="Invalid agent_type")
    
    agent_id = db.store_agent(agent)
    return {"agent_id": agent_id, "status": "created"}

@router.get("/{agent_id}")
async def get_agent(agent_id: str, db=Depends(get_db)):
    """Get specific agent."""
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()