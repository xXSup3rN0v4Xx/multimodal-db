"""
Agent Management Router
CRUD operations for agent configurations.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..dependencies import get_db, get_vector_db
from core import create_corecoder_agent, create_example_agent

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
    """Create new agent (corecoder or example)."""
    if request.agent_type == "corecoder":
        agent = create_corecoder_agent()
    elif request.agent_type == "example" or request.agent_type == "multimodal":
        agent = create_example_agent()
    else:
        raise HTTPException(status_code=400, detail="Invalid agent_type. Use 'corecoder' or 'example'")
    
    # Override name with request
    agent.name = request.name
    if request.description:
        agent.description = request.description
    
    agent_id = db.store_agent(agent)
    
    # Return full agent config, not just ID
    stored_agent = db.get_agent(agent_id)
    if not stored_agent:
        raise HTTPException(status_code=500, detail="Failed to create agent")
    
    return stored_agent.to_dict()

@router.get("/{agent_id}")
async def get_agent(agent_id: str, db=Depends(get_db)):
    """Get specific agent."""
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, db=Depends(get_db)):
    """Delete an agent and all its conversations."""
    # Check if agent exists
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Delete agent's conversations
    db.clear_messages(agent_id)
    
    # Delete the agent itself
    db.delete_agent(agent_id)
    
    return {"success": True, "message": f"Agent {agent_id} deleted successfully"}

@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: AgentCreateRequest, db=Depends(get_db)):
    """Update an existing agent."""
    # Check if agent exists
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update agent properties
    agent.name = request.name
    if request.description:
        agent.description = request.description
    
    # Store updated agent
    db.store_agent(agent)
    
    # Return updated agent
    updated_agent = db.get_agent(agent_id)
    return updated_agent.to_dict()