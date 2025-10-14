# Examples Documentation

Complete guide to the example applications and user interfaces.

## Available Examples

1. **enhanced_gradio_ui.py** - Full-featured Gradio interface (recommended)
2. **simple_gradio_ui.py** - Minimal Gradio interface

---

## enhanced_gradio_ui.py

Professional full-featured web interface for multimodal-db with AI chat capabilities.

### Location

```bash
examples/enhanced_gradio_ui.py
```

### Starting the UI

```bash
# Ensure API is running first (Terminal 1)
python multimodal-db/api/run_api.py

# Start Enhanced UI (Terminal 2)
python examples/enhanced_gradio_ui.py

# Access at: http://localhost:7860
```

### Features Overview

#### 1. Status Dashboard

Shows real-time status of:
- **API Status**: Connection to FastAPI backend
- **Ollama Status**: AI model availability
- **Refresh Button**: Update status indicators

**Status Messages**:
- ✅ `API is operational` - Backend ready
- ❌ `API unavailable` - Check if API is running
- ✅ `Ollama available (qwen2.5-coder:3b)` - AI chat ready
- ⚠️ `Ollama not available` - Install Ollama or start service

#### 2. System Stats Tab

View comprehensive system metrics:
- Total agents count
- Agents grouped by name
- Total content count
- Content breakdown by media type
- Vector collection statistics
- Database file sizes

**Usage**:
1. Click **System Stats** tab
2. Click **📊 Refresh Stats** button
3. View JSON output with all metrics

**Example Output**:
```json
{
  "agents": {
    "total": 5,
    "by_name": {
      "CoreCoder": 2,
      "MultimodalAgent": 3
    }
  },
  "content": {
    "total": 150,
    "by_type": {
      "text": 100,
      "image": 30
    }
  }
}
```

#### 3. Agent Management Tab

Complete agent CRUD operations.

**Create New Agent**:
1. Enter agent name (e.g., `my_assistant`)
2. Select agent type:
   - **corecoder**: Coding assistant with qwen2.5-coder:3b
   - **multimodal**: General-purpose with multi-modal support
3. Add optional description
4. Add comma-separated tags (e.g., `coding, python, ai`)
5. Click **✨ Create Agent**

**View Agents (Simple List)**:
1. Click **📋 List Agents** button
2. View table with: Name, ID (short), Created, Updated, Tags

**View Detailed Agent Information**:
1. Enter agent ID (full UUID or partial match)
2. Click **🔍 View Details**
3. See expandable detailed view with:
   - Full agent ID
   - Name, description, tags
   - Creation/update timestamps
   - **Helper Prompts** - All custom prompts
   - **System Prompt** - Base instruction
   - **Models** - Enabled AI models

**Delete Agent**:
1. Enter agent ID (full or partial)
2. Click **🗑️ Delete Agent**
3. Confirm action
4. Agent and associated data removed

**Tips**:
- Use partial IDs for convenience (first 8+ characters usually unique)
- Detailed view shows all hidden configuration
- Delete is permanent - no undo!

#### 4. Content Management Tab

Upload and browse content associated with agents.

**Upload Content**:
1. Enter agent ID
2. Enter content text or file path
3. Select media type:
   - `text` - Plain text, documents
   - `image` - Image file paths
   - `audio` - Audio file paths
   - `video` - Video file paths
   - `document` - PDF, DOCX, etc.
   - `embedding` - Vector embeddings
4. Add optional JSON metadata: `{"key": "value"}`
5. Click **📤 Upload Content**

**Browse Content**:
1. Click **📚 List Content** button
2. View table with: Media Type, Content Preview, Agent ID, Timestamp, Metadata

**Example Usage**:
```
Agent ID: 550e8400-e29b-41d4-a716-446655440000
Content: "This is important research data about AI"
Media Type: text
Metadata: {"category": "research", "priority": "high"}
```

#### 5. AI Chat Tab

Real-time conversational AI using Ollama integration.

**Requirements**:
- Ollama must be installed and running
- Model `qwen2.5-coder:3b` must be pulled
- API must be running

**Using Chat**:
1. Click **💬 AI Chat** tab
2. Enter agent ID to provide context
3. Type your message
4. Press Enter or click **Send**
5. View AI response in chat history

**Features**:
- **Context-Aware**: Uses agent's system prompts and helper prompts
- **Conversation History**: Maintains chat context (last 10 messages)
- **Status Indicators**: Shows if response from Ollama or fallback
- **Error Handling**: Graceful degradation if Ollama unavailable

**Example Conversation**:
```
You: Explain Python decorators
AI: Decorators are a way to modify or enhance functions...

You: Show me an example
AI: Here's a simple decorator example:
    
    def my_decorator(func):
        def wrapper():
            print("Before function")
            func()
            print("After function")
        return wrapper
```

**Chat Tips**:
- Ollama responses can take 5-30 seconds depending on complexity
- Longer messages and longer conversations take more time
- Clear chat history by refreshing page
- Use appropriate agent for best results (e.g., CoreCoder for coding questions)

### Complete Workflow Example

**Scenario**: Create a research assistant and use it for AI chat

1. **Start Services**:
   ```bash
   # Terminal 1: API
   python multimodal-db/api/run_api.py
   
   # Terminal 2: Ollama
   ollama serve
   
   # Terminal 3: UI
   python examples/enhanced_gradio_ui.py
   ```

2. **Create Agent**:
   - Tab: **Agent Management**
   - Name: `ResearchBot`
   - Type: `multimodal`
   - Description: `AI research assistant`
   - Tags: `research, analysis, science`
   - Click **Create**
   - Copy the agent ID from result

3. **Upload Research Data**:
   - Tab: **Content Management**
   - Agent ID: `<paste agent ID>`
   - Content: `Research shows AI models improve with scale`
   - Media Type: `text`
   - Metadata: `{"source": "paper_2025"}`
   - Click **Upload**

4. **Chat with Agent**:
   - Tab: **AI Chat**
   - Agent ID: `<paste agent ID>`
   - Message: `What does research show about AI?`
   - Send and wait for response

5. **View Stats**:
   - Tab: **System Stats**
   - Click **Refresh Stats**
   - See your new agent and content in metrics

### UI Configuration

Default settings (in code):
```python
API_BASE = "http://localhost:8000"  # FastAPI backend
PORT = 7860                          # Gradio server port
```

To customize, edit `examples/enhanced_gradio_ui.py`:
```python
# Change API endpoint
API_BASE = "http://192.168.1.100:8000"

# Change Gradio port
demo.launch(server_port=8080)
```

### Keyboard Shortcuts

- **Enter** in chat input: Send message
- **Ctrl+Enter** in text fields: New line
- **Ctrl+R**: Refresh page (clears state)

---

## simple_gradio_ui.py

Minimal interface for basic operations and testing.

### Location

```bash
examples/simple_gradio_ui.py
```

### Starting the UI

```bash
# Ensure API is running first
python multimodal-db/api/run_api.py

# Start Simple UI (different port to avoid conflict)
python examples/simple_gradio_ui.py

# Access at: http://localhost:7861
```

### Features

Streamlined interface with 5 tabs:

#### 1. Health Check Tab
- View API status
- Check system health
- Refresh status button

#### 2. System Stats Tab
- Display system statistics
- JSON format output
- Refresh button

#### 3. Agents Tab
- Create new agent (name, type, description, tags)
- List all agents in table format
- Simple CRUD operations

#### 4. Content Tab
- Upload content (agent ID, text, media type, metadata)
- List all content in table format
- Basic content management

#### 5. Chat Tab
- Basic AI chat interface
- Agent ID input
- Message input/output
- Chat history display

### When to Use Simple UI

**Use Simple UI for**:
- Quick testing
- API endpoint validation
- Minimal resource usage
- Learning the basics
- Automated testing scripts

**Use Enhanced UI for**:
- Production use
- Full feature access
- Better user experience
- Agent detail inspection
- Professional interface

### Differences from Enhanced UI

| Feature | Simple | Enhanced |
|---------|--------|----------|
| Status Dashboard | ✓ | ✓✓ Better |
| Agent Creation | ✓ | ✓ |
| Agent Details | ✗ | ✓✓ |
| Agent Deletion | ✗ | ✓ |
| Content Upload | ✓ | ✓ |
| AI Chat | ✓ Basic | ✓✓ Advanced |
| Conversation History | ✓ Limited | ✓✓ Full |
| Helper Prompts Display | ✗ | ✓ |
| Partial ID Matching | ✗ | ✓ |
| UI Polish | Basic | ✓✓ Professional |

---

## Creating Custom UIs

### Gradio Basics

```python
import gradio as gr

def my_function(input_text):
    return f"You entered: {input_text}"

with gr.Blocks() as demo:
    gr.Markdown("# My Custom UI")
    
    with gr.Row():
        input_box = gr.Textbox(label="Input")
        output_box = gr.Textbox(label="Output")
    
    button = gr.Button("Process")
    button.click(my_function, inputs=input_box, outputs=output_box)

demo.launch()
```

### Integrating with Multimodal-DB API

```python
import gradio as gr
import requests

API_BASE = "http://localhost:8000"

def create_agent(name, agent_type):
    """Create agent via API"""
    try:
        response = requests.post(
            f"{API_BASE}/agents/",
            json={"name": name, "agent_type": agent_type}
        )
        if response.status_code == 200:
            return f"✅ Created: {response.json()['agent_id']}"
        else:
            return f"❌ Error: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_agents():
    """List agents via API"""
    try:
        response = requests.get(f"{API_BASE}/agents/")
        agents = response.json()
        return [[a['name'], a['agent_id'][:24], a.get('created_at', 'N/A')] 
                for a in agents]
    except:
        return [["Error", "Could not fetch agents", ""]]

with gr.Blocks(title="Custom UI") as demo:
    gr.Markdown("# Custom Multimodal-DB Interface")
    
    # Create agent section
    with gr.Row():
        name_input = gr.Textbox(label="Agent Name")
        type_input = gr.Radio(["corecoder", "multimodal"], label="Type")
        create_btn = gr.Button("Create")
        result_output = gr.Textbox(label="Result")
    
    create_btn.click(
        create_agent,
        inputs=[name_input, type_input],
        outputs=result_output
    )
    
    # List agents section
    list_btn = gr.Button("List Agents")
    agents_table = gr.Dataframe(
        headers=["Name", "ID", "Created"],
        label="Agents"
    )
    
    list_btn.click(list_agents, outputs=agents_table)

demo.launch(server_port=7862)
```

### Example: Specialized Research UI

```python
"""
Specialized UI for research document management
"""
import gradio as gr
import requests

API_BASE = "http://localhost:8000"

def upload_research_doc(agent_id, title, content, author, year):
    """Upload research document with metadata"""
    metadata = {
        "title": title,
        "author": author,
        "year": year,
        "type": "research"
    }
    
    response = requests.post(
        f"{API_BASE}/content/",
        json={
            "agent_id": agent_id,
            "content": content,
            "media_type": "text",
            "metadata": metadata
        }
    )
    
    if response.status_code == 200:
        return "✅ Document uploaded successfully"
    else:
        return f"❌ Error: {response.text}"

def search_research(query):
    """Search research documents"""
    response = requests.post(
        f"{API_BASE}/search/content",
        json={
            "query": query,
            "media_type": "text",
            "limit": 10
        }
    )
    
    results = response.json()
    return [[
        r.get('metadata', {}).get('title', 'Untitled'),
        r.get('metadata', {}).get('author', 'Unknown'),
        r.get('content', '')[:100]
    ] for r in results]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Research Document Manager")
    
    with gr.Tab("Upload Document"):
        agent_id = gr.Textbox(label="Agent ID")
        title = gr.Textbox(label="Document Title")
        author = gr.Textbox(label="Author")
        year = gr.Number(label="Year", precision=0)
        content = gr.Textbox(label="Content", lines=10)
        upload_btn = gr.Button("Upload", variant="primary")
        upload_result = gr.Textbox(label="Result")
        
        upload_btn.click(
            upload_research_doc,
            inputs=[agent_id, title, content, author, year],
            outputs=upload_result
        )
    
    with gr.Tab("Search Documents"):
        search_input = gr.Textbox(label="Search Query")
        search_btn = gr.Button("Search")
        results_table = gr.Dataframe(
            headers=["Title", "Author", "Preview"],
            label="Results"
        )
        
        search_btn.click(
            search_research,
            inputs=search_input,
            outputs=results_table
        )

demo.launch(server_port=7863)
```

---

## Deployment

### Local Network Access

```python
# In your UI script, change launch to:
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    share=False
)

# Access from other devices:
# http://<your-ip>:7860
```

### Public Sharing (Gradio Share)

```python
# Enable temporary public URL
demo.launch(share=True)

# Gradio will provide a public URL like:
# https://1234abcd.gradio.live
```

**⚠️ Security Warning**: Share links are public. Don't use for sensitive data.

### Production Deployment

For production, use a reverse proxy:

```nginx
# nginx config
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Troubleshooting

### UI Won't Start

```bash
# Check if port is in use
netstat -ano | findstr :7860  # Windows
lsof -i :7860                 # Linux/Mac

# Change port in script or via command line
python examples/enhanced_gradio_ui.py --server-port 7861
```

### Can't Connect to API

```bash
# Verify API is running
curl http://localhost:8000/

# Check API_BASE in UI script
# Make sure it matches your API address
```

### Ollama Not Available

```bash
# Install Ollama
# Visit: https://ollama.ai

# Pull model
ollama pull qwen2.5-coder:3b

# Start Ollama service
ollama serve
```

### Slow Chat Responses

- Normal: 5-30 seconds depending on message complexity
- Check Ollama is using GPU (if available)
- Reduce message length
- Ensure sufficient RAM (8GB+ recommended)

---

## Best Practices

1. **Always start API first** before launching UI
2. **Use Enhanced UI for production**, Simple UI for testing
3. **Keep browser tab active** during long AI responses
4. **Save agent IDs** - copy them after creation
5. **Use partial IDs** (first 8-12 chars) for convenience
6. **Monitor system stats** regularly to track growth
7. **Test Ollama separately** before using chat: `ollama run qwen2.5-coder:3b "Hello"`

---

## See Also

- **Python Library**: [LIBRARY.md](LIBRARY.md) - For scripting
- **REST API**: [API.md](API.md) - HTTP endpoints
- **CLI Tools**: [CLI.md](CLI.md) - Command-line utilities
- **Testing**: [TESTING.md](TESTING.md) - Test suite documentation
