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
    agent.agent_name = request.name
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

@router.post("/deduplicate")
async def deduplicate_agents(db=Depends(get_db)):
    """Remove duplicate agent entries from the database."""
    db.deduplicate_agents()
    return {"success": True, "message": "Agent duplicates removed"}

@router.patch("/{agent_id}")
async def patch_agent(agent_id: str, updates: Dict[str, Any], db=Depends(get_db)):
    """
    Partially update an agent configuration.
    Supports updating any agent property including models, prompts, tools, etc.
    """
    # Check if agent exists
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Apply updates to agent
    if "models" in updates:
        # Deep merge models configuration
        models_update = updates["models"]
        for model_type, model_configs in models_update.items():
            if model_type not in agent.models:
                agent.models[model_type] = {}
            for model_name, model_data in model_configs.items():
                if model_name not in agent.models[model_type]:
                    agent.models[model_type][model_name] = {
                        "enabled": False,
                        "instances": [],
                        "system_prompt_supported": True
                    }
                # Update instances if provided
                if "instances" in model_data:
                    agent.models[model_type][model_name]["instances"] = model_data["instances"]
                    agent.models[model_type][model_name]["enabled"] = True
                # Update enabled status if explicitly provided
                if "enabled" in model_data:
                    agent.models[model_type][model_name]["enabled"] = model_data["enabled"]
    
    if "name" in updates:
        agent.agent_name = updates["name"]
    
    if "description" in updates:
        agent.description = updates["description"]
    
    if "tags" in updates:
        agent.tags = updates["tags"]
    
    if "prompts" in updates:
        # Deep merge prompts
        for prompt_type, prompt_data in updates["prompts"].items():
            if prompt_type in agent.prompts:
                if isinstance(prompt_data, dict):
                    agent.prompts[prompt_type].update(prompt_data)
                else:
                    agent.prompts[prompt_type] = prompt_data
    
    if "tools" in updates:
        agent.tools.update(updates["tools"])
    
    if "databases" in updates:
        agent.databases.update(updates["databases"])
    
    if "rag_config" in updates:
        agent.rag_config.update(updates["rag_config"])
    
    # Update timestamp
    agent._update_timestamp()
    
    # Store updated agent
    db.store_agent(agent)
    
    # Return updated agent
    updated_agent = db.get_agent(agent_id)
    return updated_agent.to_dict()

@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: AgentCreateRequest, db=Depends(get_db)):
    """Update an existing agent (full replacement of basic properties)."""
    # Check if agent exists
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update agent properties
    agent.agent_name = request.name
    if request.description:
        agent.description = request.description
    
    # Store updated agent
    db.store_agent(agent)
    
    # Return updated agent
    updated_agent = db.get_agent(agent_id)
    return updated_agent.to_dict()