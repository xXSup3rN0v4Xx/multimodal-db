"""
Multimodal-DB CLI Tool

Comprehensive command-line interface for multimodal database operations.
Provides commands for agent management, database operations, queries, and exports.

Usage:
    python -m multimodal-db.cli.cli [command] [options]
    
Or install and use:
    multimodal-db [command] [options]

Author: Multimodal-DB Team
Version: 1.0.0
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import (
    PolarsDB,
    QdrantVectorDB,
    MultimodalDB,
    GraphitiDB,
    AgentConfig,
    ModelType,
    PromptType,
    DatabaseCategory,
    MediaType,
    ParquetExporter,
    PandasNLQueryEngine,
    PolarsNLQueryEngine,
    QdrantHybridSearchEngine
)


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    click.echo(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    click.echo(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    click.echo(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    click.echo(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    click.echo(f"{Colors.FAIL}✗ {text}{Colors.ENDC}", err=True)


def print_info(text: str):
    """Print info message."""
    click.echo(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    click.echo(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


@click.group()
@click.version_option(version='1.0.0', prog_name='multimodal-db')
def cli():
    """
    Multimodal-DB - Comprehensive Database Management System
    
    A powerful database system for AI agents, conversations, and multimodal content.
    """
    pass


# ============================================================================
# AGENT COMMANDS
# ============================================================================

@cli.group()
def agent():
    """Manage AI agents and configurations."""
    pass


@agent.command()
@click.option('--name', required=True, help='Agent name')
@click.option('--description', default='', help='Agent description')
@click.option('--db-path', default='multimodal_db', help='Database path')
def create(name: str, description: str, db_path: str):
    """Create a new agent configuration."""
    try:
        print_header(f"Creating Agent: {name}")
        
        # Initialize database
        db = PolarsDB(db_path)
        
        # Create agent config
        agent = AgentConfig(agent_name=name)
        if description:
            agent.set_description(description)
        
        # Add to database
        agent_id = db.add_agent(agent)
        
        print_success(f"Agent created with ID: {agent_id}")
        print_info(f"Name: {name}")
        print_info(f"Description: {description or 'None'}")
        print_info(f"Database: {db_path}")
        
    except Exception as e:
        print_error(f"Failed to create agent: {e}")
        sys.exit(1)


@agent.command()
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def list(db_path: str, format: str):
    """List all agents in the database."""
    try:
        print_header("Agents List")
        
        db = PolarsDB(db_path)
        agents = db.list_agents()
        
        if not agents:
            print_warning("No agents found in database.")
            return
        
        if format == 'json':
            click.echo(json.dumps(agents, indent=2, default=str))
        else:
            for agent in agents:
                click.echo(f"\n{Colors.BOLD}Agent ID:{Colors.ENDC} {agent['agent_id']}")
                click.echo(f"{Colors.BOLD}Name:{Colors.ENDC} {agent['name']}")
                click.echo(f"{Colors.BOLD}Created:{Colors.ENDC} {agent['created_at']}")
                click.echo("-" * 60)
        
        print_success(f"Found {len(agents)} agent(s)")
        
    except Exception as e:
        print_error(f"Failed to list agents: {e}")
        sys.exit(1)


@agent.command()
@click.argument('agent_id')
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.option('--format', type=click.Choice(['summary', 'full', 'json']), default='summary', help='Output format')
def show(agent_id: str, db_path: str, format: str):
    """Show agent configuration details."""
    try:
        print_header(f"Agent Details: {agent_id}")
        
        db = PolarsDB(db_path)
        agent = db.get_agent(agent_id)
        
        if not agent:
            print_error(f"Agent not found: {agent_id}")
            sys.exit(1)
        
        if format == 'json':
            click.echo(json.dumps(agent.to_dict(), indent=2, default=str))
        elif format == 'summary':
            summary = agent.get_summary()
            for key, value in summary.items():
                click.echo(f"{Colors.BOLD}{key}:{Colors.ENDC} {value}")
        else:  # full
            config = agent.to_dict()
            click.echo(json.dumps(config, indent=2, default=str))
        
        print_success("Agent details retrieved successfully")
        
    except Exception as e:
        print_error(f"Failed to show agent: {e}")
        sys.exit(1)


@agent.command()
@click.argument('agent_id')
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.confirmation_option(prompt='Are you sure you want to delete this agent?')
def delete(agent_id: str, db_path: str):
    """Delete an agent from the database."""
    try:
        print_header(f"Deleting Agent: {agent_id}")
        
        db = PolarsDB(db_path)
        
        # Verify agent exists
        agent = db.get_agent(agent_id)
        if not agent:
            print_error(f"Agent not found: {agent_id}")
            sys.exit(1)
        
        # Delete agent
        # Note: Implement delete method in PolarsDB if not exists
        print_warning("Delete functionality needs to be implemented in PolarsDB")
        
    except Exception as e:
        print_error(f"Failed to delete agent: {e}")
        sys.exit(1)


@agent.command()
@click.argument('agent_id')
@click.option('--model-type', required=True, help='Model type (e.g., large_language_model)')
@click.option('--model-name', required=True, help='Model name (e.g., ollama)')
@click.option('--model-config', help='Model configuration as JSON string')
@click.option('--db-path', default='multimodal_db', help='Database path')
def enable_model(agent_id: str, model_type: str, model_name: str, model_config: Optional[str], db_path: str):
    """Enable a model for an agent."""
    try:
        print_header(f"Enabling Model for Agent: {agent_id}")
        
        db = PolarsDB(db_path)
        agent = db.get_agent(agent_id)
        
        if not agent:
            print_error(f"Agent not found: {agent_id}")
            sys.exit(1)
        
        # Parse config if provided
        config = json.loads(model_config) if model_config else None
        
        # Enable model
        agent.enable_model(model_type, model_name, config)
        
        # Update in database
        db.update_agent(agent_id, agent)
        
        print_success(f"Model enabled: {model_type}/{model_name}")
        
    except Exception as e:
        print_error(f"Failed to enable model: {e}")
        sys.exit(1)


@agent.command()
@click.argument('agent_id')
@click.option('--system-prompt', help='System prompt text')
@click.option('--helper-name', help='Helper prompt name')
@click.option('--helper-text', help='Helper prompt text')
@click.option('--db-path', default='multimodal_db', help='Database path')
def set_prompt(agent_id: str, system_prompt: Optional[str], helper_name: Optional[str], 
               helper_text: Optional[str], db_path: str):
    """Set prompts for an agent."""
    try:
        print_header(f"Setting Prompts for Agent: {agent_id}")
        
        db = PolarsDB(db_path)
        agent = db.get_agent(agent_id)
        
        if not agent:
            print_error(f"Agent not found: {agent_id}")
            sys.exit(1)
        
        # Set system prompt
        if system_prompt:
            # Find first enabled LLM
            enabled_models = agent.get_enabled_models()
            if ModelType.LLM.value in enabled_models and enabled_models[ModelType.LLM.value]:
                model_name = enabled_models[ModelType.LLM.value][0]
                agent.set_system_prompt(ModelType.LLM.value, model_name, system_prompt)
                print_success(f"System prompt set for {ModelType.LLM.value}/{model_name}")
        
        # Set helper prompt
        if helper_name and helper_text:
            agent.add_helper_prompt(helper_name, helper_text)
            print_success(f"Helper prompt '{helper_name}' added")
        
        # Update in database
        db.update_agent(agent_id, agent)
        
        print_success("Prompts updated successfully")
        
    except Exception as e:
        print_error(f"Failed to set prompts: {e}")
        sys.exit(1)


# ============================================================================
# CONVERSATION COMMANDS
# ============================================================================

@cli.group()
def conversation():
    """Manage conversations and chat history."""
    pass


@conversation.command()
@click.argument('agent_id')
@click.option('--limit', default=50, help='Number of messages to show')
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.option('--format', type=click.Choice(['chat', 'json']), default='chat', help='Output format')
def show(agent_id: str, limit: int, db_path: str, format: str):
    """Show conversation history for an agent."""
    try:
        print_header(f"Conversation History: {agent_id}")
        
        db = PolarsDB(db_path)
        messages = db.get_messages(agent_id, limit=limit)
        
        if not messages:
            print_warning("No conversation history found.")
            return
        
        if format == 'json':
            click.echo(json.dumps(messages, indent=2, default=str))
        else:
            for msg in messages:
                role = msg['role']
                content = msg['content']
                timestamp = msg['timestamp']
                
                if role == 'user':
                    click.echo(f"\n{Colors.OKBLUE}{Colors.BOLD}User:{Colors.ENDC} {content}")
                else:
                    click.echo(f"{Colors.OKGREEN}{Colors.BOLD}Assistant:{Colors.ENDC} {content}")
                click.echo(f"{Colors.WARNING}[{timestamp}]{Colors.ENDC}")
        
        print_success(f"Showed {len(messages)} message(s)")
        
    except Exception as e:
        print_error(f"Failed to show conversation: {e}")
        sys.exit(1)


@conversation.command()
@click.argument('agent_id')
@click.argument('message')
@click.option('--role', default='user', help='Message role (user/assistant)')
@click.option('--db-path', default='multimodal_db', help='Database path')
def add(agent_id: str, message: str, role: str, db_path: str):
    """Add a message to conversation history."""
    try:
        print_header(f"Adding Message to: {agent_id}")
        
        db = PolarsDB(db_path)
        
        # Verify agent exists
        agent = db.get_agent(agent_id)
        if not agent:
            print_error(f"Agent not found: {agent_id}")
            sys.exit(1)
        
        # Add message
        msg_id = db.add_message(agent_id, role, message)
        
        print_success(f"Message added with ID: {msg_id}")
        print_info(f"Role: {role}")
        print_info(f"Content: {message[:100]}...")
        
    except Exception as e:
        print_error(f"Failed to add message: {e}")
        sys.exit(1)


# ============================================================================
# DATABASE COMMANDS
# ============================================================================

@cli.group()
def database():
    """Database operations and management."""
    pass


@database.command()
@click.option('--db-path', default='multimodal_db', help='Database path')
def info(db_path: str):
    """Show database information and statistics."""
    try:
        print_header("Database Information")
        
        db = PolarsDB(db_path)
        
        # Get statistics
        agents = db.list_agents()
        
        click.echo(f"{Colors.BOLD}Database Path:{Colors.ENDC} {db_path}")
        click.echo(f"{Colors.BOLD}Agents:{Colors.ENDC} {len(agents)}")
        
        # Calculate total messages
        total_messages = 0
        for agent in agents:
            messages = db.get_messages(agent['agent_id'])
            total_messages += len(messages)
        
        click.echo(f"{Colors.BOLD}Total Messages:{Colors.ENDC} {total_messages}")
        
        print_success("Database information retrieved")
        
    except Exception as e:
        print_error(f"Failed to get database info: {e}")
        sys.exit(1)


@database.command()
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.option('--format', type=click.Choice(['parquet', 'json', 'csv']), default='parquet', help='Export format')
@click.option('--output-dir', default='exports', help='Output directory')
def export(db_path: str, format: str, output_dir: str):
    """Export entire database to files."""
    try:
        print_header(f"Exporting Database to {format.upper()}")
        
        db = PolarsDB(db_path)
        exporter = ParquetExporter(db)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export all data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'parquet':
            agents_file = output_path / f"agents_{timestamp}.parquet"
            conversations_file = output_path / f"conversations_{timestamp}.parquet"
            
            # Export agents
            db.agents.write_parquet(agents_file)
            print_success(f"Agents exported: {agents_file}")
            
            # Export conversations
            db.conversations.write_parquet(conversations_file)
            print_success(f"Conversations exported: {conversations_file}")
            
        elif format == 'json':
            agents_file = output_path / f"agents_{timestamp}.json"
            conversations_file = output_path / f"conversations_{timestamp}.json"
            
            # Export agents
            agents_data = db.agents.to_dicts()
            with open(agents_file, 'w') as f:
                json.dump(agents_data, f, indent=2, default=str)
            print_success(f"Agents exported: {agents_file}")
            
            # Export conversations
            conversations_data = db.conversations.to_dicts()
            with open(conversations_file, 'w') as f:
                json.dump(conversations_data, f, indent=2, default=str)
            print_success(f"Conversations exported: {conversations_file}")
        
        print_success(f"Database exported to: {output_dir}")
        
    except Exception as e:
        print_error(f"Failed to export database: {e}")
        sys.exit(1)


# ============================================================================
# QUERY COMMANDS
# ============================================================================

@cli.group()
def query():
    """Natural language database queries."""
    pass


@query.command()
@click.argument('query_text')
@click.option('--db-path', default='multimodal_db', help='Database path')
@click.option('--engine', type=click.Choice(['polars', 'pandas']), default='polars', help='Query engine')
@click.option('--model', default='qwen2.5-coder:3b', help='LLM model for query generation')
def run(query_text: str, db_path: str, engine: str, model: str):
    """Run a natural language query on the database."""
    try:
        print_header(f"Running Query ({engine.upper()} engine)")
        print_info(f"Query: {query_text}")
        
        db = PolarsDB(db_path)
        
        if engine == 'polars':
            query_engine = PolarsNLQueryEngine(llm_model=model)
            
            # Query agents table
            result = query_engine.query(db.agents, query_text)
            
            if result['success']:
                print_success("Query executed successfully!")
                click.echo(f"\n{Colors.BOLD}Response:{Colors.ENDC}\n{result['response']}")
                
                if result.get('polars_code'):
                    click.echo(f"\n{Colors.BOLD}Generated Code:{Colors.ENDC}\n{result['polars_code']}")
            else:
                print_error(f"Query failed: {result.get('error')}")
        
        else:  # pandas
            query_engine = PandasNLQueryEngine(model=model)
            df_pandas = db.agents.to_pandas()
            
            result = query_engine.query(df_pandas, query_text)
            
            if result['success']:
                print_success("Query executed successfully!")
                click.echo(f"\n{Colors.BOLD}Response:{Colors.ENDC}\n{result['response']}")
            else:
                print_error(f"Query failed: {result.get('error')}")
        
    except Exception as e:
        print_error(f"Failed to run query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# SEARCH COMMANDS
# ============================================================================

@cli.group()
def search():
    """Semantic search operations."""
    pass


@search.command()
@click.argument('query_text')
@click.option('--collection', default='knowledge_base', help='Qdrant collection name')
@click.option('--top-k', default=5, help='Number of results')
@click.option('--hybrid', is_flag=True, help='Use hybrid search (dense + sparse)')
def semantic(query_text: str, collection: str, top_k: int, hybrid: bool):
    """Perform semantic search on vector database."""
    try:
        print_header("Semantic Search")
        print_info(f"Query: {query_text}")
        print_info(f"Collection: {collection}")
        print_info(f"Mode: {'Hybrid' if hybrid else 'Dense'}")
        
        if hybrid:
            search_engine = QdrantHybridSearchEngine(collection_name=collection)
            results = search_engine.hybrid_search(query_text, top_k=top_k)
            
            print_success(f"Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                click.echo(f"\n{Colors.BOLD}Result {i}:{Colors.ENDC}")
                click.echo(f"Score: {result.get('score', 'N/A')}")
                click.echo(f"Content: {result.get('content', '')[:200]}...")
        else:
            print_warning("Dense-only search not yet implemented")
        
    except Exception as e:
        print_error(f"Failed to perform search: {e}")
        sys.exit(1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    cli()
