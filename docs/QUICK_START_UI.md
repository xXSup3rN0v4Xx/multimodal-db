# 🚀 Quick Start Guide - Multimodal-DB with Gradio UI

## TL;DR - Get Running in 3 Steps

```powershell
# 1. Start the API
python multimodal-db/api/run_api.py

# 2. In another terminal, start the UI
python examples/multimodal_gradio_ui_v2.py

# 3. Open your browser
# UI: http://localhost:7860
# API: http://localhost:8000/docs
```

## ✨ Features Available in v2.0 UI

### 📊 Dashboard Tab
- System health monitoring
- Real-time statistics
- Database metrics
- Vector collections status

### 🤖 Agents Tab
**Create Agents:**
- **CoreCoder**: Specialized for coding tasks with qwen2.5-coder:3b
- **Multimodal**: Handles text, images, audio, video, documents

**Manage Agents:**
- List all agents with details
- View full agent configuration
- Export agents (full with content or config only)
- Partial ID matching (copy just the first few characters!)

### 📝 Content Tab
**Upload Content:**
- Select agent to associate content with
- Choose media type (text, document, embedding, image, audio, video)
- Add custom metadata (JSON format)

**Browse Content:**
- Filter by agent
- Filter by media type
- View all stored content

### 🔍 Search Tab
**Powerful Search:**
- Search across all content
- Filter by agent (optional)
- Filter by media type (optional)
- Control result limits

**Vector Collections:**
- View all Qdrant collections
- Check collection statistics
- Monitor vector database health

### 💬 Chat Tab
**Real-time Chat:**
- Select any agent
- Have conversations
- Clear chat history
- Multi-turn conversations

## 🎯 Common Tasks

### Creating Your First Agent

1. Go to **Agents** tab
2. Fill in:
   - **Name**: `my_coding_assistant`
   - **Type**: `corecoder`
   - **Description**: `My personal coding helper`
   - **Tags**: `python, coding, helper`
3. Click **Create Agent**
4. Copy the Agent ID from the table

### Uploading Content

1. Go to **Content** tab
2. Click **Refresh Agents** if needed
3. Select your agent from dropdown
4. Enter content (e.g., Python code, documentation)
5. Choose media type (usually `text` or `document`)
6. Optional: Add metadata like `{"language": "python", "topic": "algorithms"}`
7. Click **Upload**

### Chatting with an Agent

1. Go to **Chat** tab
2. Select your agent
3. Type a message
4. Press **Send** or hit Enter
5. See the agent's response

### Searching Content

1. Go to **Search** tab
2. Enter search query (e.g., "Python optimization")
3. Optional: Filter by specific agent or media type
4. Adjust max results slider
5. Click **Search**

## 🔧 Troubleshooting

### API Not Responding
```
⚠️  WARNING: API not responding!
```

**Solution:**
```powershell
# Make sure API is running
python multimodal-db/api/run_api.py

# Check if port 8000 is available
netstat -an | findstr "8000"
```

### No Agents Found
**In UI:** You'll see "No agents found" in dropdowns

**Solution:**
1. Go to **Agents** tab
2. Create at least one agent
3. Go back to other tabs and click **Refresh**

### Agent ID Too Long
**Tip:** You only need the first 16 characters!
- Full ID: `a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6`
- Use: `a1b2c3d4-e5f6-g7...` (copied from table)

The UI automatically finds the full ID for you!

### Port Already in Use
```
Error: Port 7860 already in use
```

**Solution:**
```powershell
# Find and kill the process
netstat -ano | findstr "7860"
taskkill /PID <PID> /F

# Or change the port in the code:
# In multimodal_gradio_ui_v2.py, line ~780:
# app.launch(server_port=7861)  # Use different port
```

## 📱 UI Layout

```
┌─────────────────────────────────────┐
│  🗾 Multimodal-DB Control Center   │
│  Razor-Sharp Data Management        │
│  API: ONLINE ✅                      │
├─────────────────────────────────────┤
│  📊│🤖│📝│🔍│💬                      │
│  ────────────────────────────────   │
│  [Current Tab Content]              │
│                                     │
│  [Forms, Tables, Inputs]            │
│                                     │
│  [Action Buttons]                   │
└─────────────────────────────────────┘
```

## 🎨 Theme

- **Background**: Dark Grey (#181818)
- **Accent**: Banana Yellow (#ffe135)
- **Highlight**: Gold (#ffd700)
- **Professional dark theme optimized for long coding sessions**

## 📊 Workflow Example

### Complete Workflow: Create Agent → Add Content → Search → Chat

```powershell
# 1. Start services
python multimodal-db/api/run_api.py     # Terminal 1
python examples/multimodal_gradio_ui_v2.py  # Terminal 2

# 2. In Browser (http://localhost:7860)
# → Agents Tab:
#    Create "code_helper" (corecoder)

# → Content Tab:
#    Upload Python code snippets
#    Upload documentation
#    Upload examples

# → Search Tab:
#    Search for "sorting algorithms"
#    Find relevant content

# → Chat Tab:
#    Ask: "Explain the binary search algorithm"
#    Get response from agent
```

## 🔒 Production Notes

**For Production Use:**
- Change `allow_origins` in `api/main.py` CORS settings
- Set proper authentication
- Use environment variables for config
- Enable HTTPS
- Set up proper logging
- Use production WSGI server (gunicorn/uvicorn)

## 📚 Next Steps

1. **Read**: `GRADIO_UI_UPGRADE.md` for detailed architecture
2. **Explore**: API docs at http://localhost:8000/docs
3. **Test**: Run `tests/System_Test_Demo.ipynb` notebook
4. **Integrate**: Connect with chatbot-python-core for real AI

## 🤝 Support

- **Issues**: Check `GRADIO_UI_UPGRADE.md` for common confusion
- **API Docs**: http://localhost:8000/docs
- **Architecture**: See `README.md` in project root

---

**Built with ❤️ for the AI Agent Community**
