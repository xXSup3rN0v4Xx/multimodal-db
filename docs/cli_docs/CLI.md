# CLI Tools Documentation

Command-line utilities for database management and agent operations.

## Available Tools

1. **cleanup_agents.py** - Interactive database cleanup tool
2. **initialize_corecoder.py** - Create sample CoreCoder agent
3. **run_api.py** - Start the FastAPI server

---

## cleanup_agents.py

Interactive tool to clean up duplicate and test agents from the database.

### Location

```bash
scripts/cleanup_agents.py
```

### Usage

```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the script
python scripts/cleanup_agents.py
```

### Features

The script provides an interactive menu with 5 options:

#### Option 1: Remove All Test Agents

Removes all agents named `test_db_agent`.

**Process:**
1. Shows count of test agents found
2. Asks for confirmation (type `yes`)
3. Deletes all test agents
4. Shows count of deleted agents

**Example Output:**
```
⚠️  This will delete 5 test agents!
Type 'yes' to confirm: yes
✅ Deleted 5 test agents
```

#### Option 2: Remove Duplicate CoreCoder Agents

Removes duplicate agents with the same name, keeping only the newest one.

**Process:**
1. Finds agents named `CoreCoder`
2. Sorts by creation date
3. Keeps the newest, marks others for deletion
4. Asks for confirmation
5. Deletes duplicates

**Example Output:**
```
⚠️  This will delete 3 duplicate CoreCoder agent(s)
   Keeping: 550e8400-e29b-41d4-a716-... (newest)
Type 'yes' to confirm: yes
✅ Deleted 3 duplicate agents
```

#### Option 3: Remove Specific Agent

Remove a single agent by its number in the list.

**Process:**
1. Shows numbered list of all agents
2. You enter the number of agent to delete
3. Asks for confirmation
4. Deletes the agent

**Example:**
```
🔍 All Agents:

  [1] CoreCoder
      ID: 550e8400-e29b-41d4-a716...
      Created: 2025-10-13 10:00:00

  [2] test_db_agent
      ID: 660e8400-e29b-41d4-a716...
      Created: 2025-10-13 11:00:00

Select option (1-5): 3
Enter agent number to delete: 2
Type 'yes' to confirm: yes
✅ Deleted agent: 660e8400-e29b-41d4-a716...
```

#### Option 4: Remove Multiple Agents

Remove multiple agents at once by their numbers.

**Process:**
1. Shows numbered list of all agents
2. You enter comma-separated numbers (e.g., `1,3,5`)
3. Asks for confirmation
4. Deletes all selected agents

**Example:**
```
Enter agent numbers (comma-separated, e.g., 1,2,3): 2,4,6
Type 'yes' to confirm: yes
✅ Deleted 3 agents
```

#### Option 5: Exit

Exit without making any changes.

### Complete Example Session

```bash
$ python scripts/cleanup_agents.py

🧹 Multimodal-DB Agent Cleanup Tool

📊 Total agents in database: 8

📋 Agent Summary:
  ✅ CoreCoder: 1 instance(s)
  ⚠️  DUPLICATE test_db_agent: 5 instance(s)
  ✅ MultimodalAgent: 2 instance(s)

============================================================

🔍 All Agents:

  [1] CoreCoder
      ID: 550e8400-e29b-41d4-a716...
      Created: 2025-10-13 10:00:00

  [2] test_db_agent
      ID: 660e8400-e29b-41d4-a716...
      Created: 2025-10-13 09:00:00

  [3] test_db_agent
      ID: 770e8400-e29b-41d4-a716...
      Created: 2025-10-13 09:05:00

  [4] test_db_agent
      ID: 880e8400-e29b-41d4-a716...
      Created: 2025-10-13 09:10:00

  [5] test_db_agent
      ID: 990e8400-e29b-41d4-a716...
      Created: 2025-10-13 09:15:00

  [6] test_db_agent
      ID: aa0e8400-e29b-41d4-a716...
      Created: 2025-10-13 09:20:00

  [7] MultimodalAgent
      ID: bb0e8400-e29b-41d4-a716...
      Created: 2025-10-13 10:30:00

  [8] MultimodalAgent
      ID: cc0e8400-e29b-41d4-a716...
      Created: 2025-10-13 10:35:00

============================================================

🗑️  Cleanup Options:

  [1] Remove all 'test_db_agent' entries
  [2] Remove duplicate CoreCoder agents (keep newest)
  [3] Remove specific agent by number
  [4] Remove multiple agents by numbers (e.g., 1,2,3)
  [5] Exit without changes

Select option (1-5): 1

⚠️  This will delete 5 test agents!
Type 'yes' to confirm: yes
✅ Deleted 5 test agents

Final agent count: 3
```

### Script Details

**Database Path**: Uses default `MultimodalDB()` path (`data/multimodal_db`)

**Safety Features**:
- Always shows what will be deleted
- Requires explicit `yes` confirmation
- Shows before/after counts
- Non-destructive preview mode

**Exit Codes**:
- `0` - Success or user cancelled
- `1` - Error occurred

---

## initialize_corecoder.py

Creates a sample CoreCoder agent with default configuration.

### Location

```bash
scripts/initialize_corecoder.py
```

### Usage

```bash
python scripts/initialize_corecoder.py
```

### What It Does

1. Creates a CoreCoder agent with:
   - Name: "CoreCoder"
   - Type: Coding assistant
   - Model: qwen2.5-coder:3b
   - Helper prompts for software engineering
   - System prompt for development tasks

2. Stores agent in database
3. Prints agent ID

### Example Output

```bash
$ python scripts/initialize_corecoder.py

Created CoreCoder agent: 550e8400-e29b-41d4-a716-446655440000
Agent ready to use!
```

### Use Cases

- Quick setup for testing
- Create initial agent after installation
- Template for creating custom agents
- CI/CD initialization

### Customizing

To create your own initialization script, copy and modify:

```python
#!/usr/bin/env python3
"""
Create custom agent
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "multimodal-db"))

from core import create_corecoder_agent, MultimodalDB

def main():
    # Create agent with custom config
    agent = create_corecoder_agent(name="MyCustomAgent")
    agent.add_helper_prompt("custom", "My custom instructions")
    
    # Store in database
    db = MultimodalDB()
    agent_id = db.store_agent(agent)
    
    print(f"Created agent: {agent_id}")

if __name__ == "__main__":
    main()
```

---

## run_api.py

Starts the FastAPI development server.

### Location

```bash
multimodal-db/api/run_api.py
```

### Usage

```bash
python multimodal-db/api/run_api.py
```

### Configuration

Default settings:
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8000`
- **Reload**: `True` (auto-reload on code changes)
- **Log Level**: `info`

### Example Output

```bash
$ python multimodal-db/api/run_api.py

🗾 Starting Multimodal-DB Unified API...
📍 API Documentation: http://localhost:8000/docs
🚀 Ready for chatbot-python-core and chatbot-nextjs-webui integration!

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Command Line Options

To customize, modify `run_api.py` or use uvicorn directly:

```bash
# Custom port
uvicorn api.main:app --port 9000

# No auto-reload (production)
uvicorn api.main:app --no-reload

# Bind to localhost only
uvicorn api.main:app --host 127.0.0.1

# With workers (production)
uvicorn api.main:app --workers 4
```

### Environment Variables

Create `.env` file:

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
LOG_LEVEL=info
```

### Stopping the Server

Press `Ctrl+C` in the terminal.

---

## Creating Custom CLI Tools

### Template Script

```python
#!/usr/bin/env python3
"""
Custom CLI tool template
"""
import sys
import argparse
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent.parent / "multimodal-db"))

from core import MultimodalDB, QdrantVectorDB

def main():
    parser = argparse.ArgumentParser(description="My custom tool")
    parser.add_argument("--agent-id", help="Agent ID to process")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize database
    db = MultimodalDB()
    
    # Your logic here
    if args.agent_id:
        agent = db.get_agent(args.agent_id)
        if agent:
            print(f"Found agent: {agent.agent_name}")
        else:
            print("Agent not found")
            sys.exit(1)
    
    # More operations...
    
    if args.verbose:
        print("Operation completed successfully")

if __name__ == "__main__":
    main()
```

### Using argparse

```python
import argparse

parser = argparse.ArgumentParser(description="Tool description")

# Positional argument
parser.add_argument("filename", help="File to process")

# Optional argument
parser.add_argument("--output", "-o", help="Output file")

# Flag
parser.add_argument("--verbose", "-v", action="store_true")

# Choice
parser.add_argument("--format", choices=["json", "csv", "xml"])

# Multiple values
parser.add_argument("--tags", nargs="+", help="Multiple tags")

args = parser.parse_args()
```

### Example: Export Agents Tool

```python
#!/usr/bin/env python3
"""
Export all agents to JSON file
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "multimodal-db"))
from core import MultimodalDB

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export agents to JSON")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--full", action="store_true", help="Include full config")
    args = parser.parse_args()
    
    # Get agents
    db = MultimodalDB()
    agents = db.list_agents(include_full_config=args.full)
    
    # Write to file
    with open(args.output, 'w') as f:
        json.dump(agents, f, indent=2, default=str)
    
    print(f"Exported {len(agents)} agents to {args.output}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/export_agents.py agents.json
python scripts/export_agents.py agents_full.json --full
```

---

## Automation & Scripting

### Batch Processing

```bash
#!/bin/bash
# Process multiple operations

# Start API in background
python multimodal-db/api/run_api.py &
API_PID=$!

# Wait for API to start
sleep 5

# Run operations
python scripts/initialize_corecoder.py
python scripts/my_custom_script.py

# Cleanup
kill $API_PID
```

### Cron Jobs

```bash
# Edit crontab
crontab -e

# Daily cleanup at 2 AM
0 2 * * * cd /path/to/multimodal-db && /path/to/venv/bin/python scripts/cleanup_agents.py

# Weekly export at midnight Sunday
0 0 * * 0 cd /path/to/multimodal-db && /path/to/venv/bin/python scripts/export_agents.py backup.json
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily/Weekly
4. Action: Start a program
5. Program: `C:\path\to\venv\Scripts\python.exe`
6. Arguments: `scripts\cleanup_agents.py`
7. Start in: `C:\path\to\multimodal-db`

---

## Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Database Not Found

```bash
# Check if data directory exists
ls data/multimodal_db/

# Initialize if missing (API auto-creates)
python multimodal-db/api/run_api.py
# Press Ctrl+C after startup
```

### Script Won't Run

```bash
# Make executable (Linux/Mac)
chmod +x scripts/cleanup_agents.py

# Run with explicit python
python scripts/cleanup_agents.py
```

### Permission Errors

**Windows:**
```bash
# Run as administrator or check file permissions
icacls data\multimodal_db /grant Users:F
```

**Linux/Mac:**
```bash
# Check ownership
ls -la data/multimodal_db/

# Fix permissions
chmod -R u+rw data/multimodal_db/
```

---

## Best Practices

1. **Always activate virtual environment** before running scripts
2. **Backup database** before cleanup operations:
   ```bash
   cp -r data/multimodal_db data/multimodal_db_backup
   ```
3. **Test on sample data** before production use
4. **Use confirmation prompts** for destructive operations
5. **Log operations** for audit trail
6. **Handle errors gracefully** with try/except

---

## See Also

- **Python Library**: [LIBRARY.md](LIBRARY.md) - For scripting
- **REST API**: [API.md](API.md) - HTTP endpoints
- **Examples**: [EXAMPLES.md](EXAMPLES.md) - Usage examples
