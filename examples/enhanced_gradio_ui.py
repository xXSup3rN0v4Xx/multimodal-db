"""
Enhanced Gradio UI for Multimodal-DB API
Full-featured interface with agent details, deletion, and improved chat.
"""
import gradio as gr
import requests
from typing import Dict, Any, List
import json

API_BASE = "http://localhost:8000"

# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is online."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            collections = data.get('components', {}).get('collections', 0)
            return f"✅ API Online | Collections: {collections}"
        return f"⚠️ API returned {response.status_code}"
    except:
        return "❌ API is offline! Start with: python multimodal-db/api/run_api.py"

def check_ollama_status():
    """Check Ollama status."""
    try:
        response = requests.get(f"{API_BASE}/chat/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('ollama_available'):
                return f"✅ Ollama Ready | Model: {data.get('model', 'N/A')}"
            return f"⚠️ {data.get('message', 'Ollama unavailable')}"
        return f"⚠️ Status check failed"
    except:
        return "❌ Cannot check Ollama status"

def get_system_stats():
    """Get enhanced system statistics."""
    try:
        response = requests.get(f"{API_BASE}/admin/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def create_agent(name: str, agent_type: str, description: str = "", tags: str = ""):
    """Create a new agent."""
    try:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        payload = {
            "name": name,
            "agent_type": agent_type,
            "description": description or None,
            "tags": tag_list or None
        }
        response = requests.post(f"{API_BASE}/agents/", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ Created: {data.get('name', 'N/A')} | ID: {data.get('agent_id', 'N/A')[:24]}..."
        return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_agents_simple():
    """List agents in simple table format."""
    try:
        response = requests.get(f"{API_BASE}/agents/", timeout=5)
        if response.status_code == 200:
            agents = response.json()
            if not agents:
                return [["No agents found", "", "", "", ""]]
            return [
                [
                    a.get('agent_id', '')[:16] + '...',
                    a.get('name', ''),
                    a.get('description', '')[:50] + '...' if a.get('description') else '',
                    ', '.join(a.get('tags', [])) if a.get('tags') else '',
                    a.get('created_at', '')[:19] if a.get('created_at') else ''
                ]
                for a in agents
            ]
        return [["Error", str(response.status_code), "", "", ""]]
    except Exception as e:
        return [["Error", str(e), "", "", ""]]

def list_agents_detailed():
    """List agents with full configuration details."""
    try:
        response = requests.get(f"{API_BASE}/agents/?include_full=true", timeout=5)
        if response.status_code == 200:
            agents = response.json()
            if not agents:
                return "No agents found"
            
            output = []
            for i, agent in enumerate(agents, 1):
                output.append(f"\n{'='*80}")
                output.append(f"Agent {i}: {agent.get('name', 'N/A')}")
                output.append(f"{'='*80}")
                output.append(f"ID: {agent.get('agent_id', 'N/A')}")
                output.append(f"Description: {agent.get('description', 'None')}")
                output.append(f"Tags: {', '.join(agent.get('tags', []))}")
                output.append(f"Created: {agent.get('created_at', 'N/A')[:19]}")
                
                # System prompt
                sys_prompt = agent.get('system_prompt', '')
                if sys_prompt:
                    output.append(f"\nSystem Prompt:")
                    output.append(f"  {sys_prompt[:200]}..." if len(sys_prompt) > 200 else f"  {sys_prompt}")
                
                # Helper prompts
                helper_prompts = agent.get('helper_prompts', {})
                if helper_prompts:
                    output.append(f"\nHelper Prompts ({len(helper_prompts)}):")
                    for name, prompt in list(helper_prompts.items())[:3]:  # Show first 3
                        output.append(f"  • {name}: {prompt[:100]}..." if len(prompt) > 100 else f"  • {name}: {prompt}")
                else:
                    output.append("\nHelper Prompts: None")
                
                # Enabled models
                enabled_models = agent.get('enabled_models', {})
                if enabled_models:
                    try:
                        models_dict = json.loads(enabled_models) if isinstance(enabled_models, str) else enabled_models
                        output.append(f"\nEnabled Models:")
                        for model_type, model_name in models_dict.items():
                            output.append(f"  • {model_type}: {model_name}")
                    except:
                        output.append(f"\nEnabled Models: {enabled_models}")
            
            return "\n".join(output)
        return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def delete_agent(agent_id: str):
    """Delete an agent by ID."""
    if not agent_id or len(agent_id) < 10:
        return "❌ Invalid agent ID"
    
    try:
        # First check if agent exists
        response = requests.get(f"{API_BASE}/agents/{agent_id}", timeout=5)
        if response.status_code == 404:
            return f"❌ Agent not found: {agent_id[:24]}..."
        
        # Delete the agent
        response = requests.delete(f"{API_BASE}/agents/{agent_id}", timeout=5)
        if response.status_code == 200:
            return f"✅ Deleted agent: {agent_id[:24]}..."
        elif response.status_code == 404:
            return f"❌ Agent not found"
        else:
            return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def upload_content(agent_id: str, content: str, media_type: str, metadata: str = ""):
    """Upload content for an agent."""
    try:
        meta_dict = json.loads(metadata) if metadata else {}
        
        payload = {
            "agent_id": agent_id,
            "content": content,
            "media_type": media_type,
            "metadata": meta_dict or None
        }
        response = requests.post(f"{API_BASE}/content/", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ Uploaded! Content ID: {data.get('content_id', 'N/A')[:16]}..."
        return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_content():
    """List all content."""
    try:
        response = requests.get(f"{API_BASE}/content/", timeout=5)
        if response.status_code == 200:
            items = response.json()
            if not items:
                return [["No content found", "", "", "", ""]]
            return [
                [
                    item.get('content_id', '')[:16] + '...',
                    item.get('agent_id', '')[:16] + '...',
                    item.get('media_type', ''),
                    str(item.get('content', ''))[:60] + '...',
                    item.get('created_at', '')[:19] if item.get('created_at') else ''
                ]
                for item in items
            ]
        return [["Error", str(response.status_code), "", "", ""]]
    except Exception as e:
        return [["Error", str(e), "", "", ""]]

def send_chat(agent_id: str, message: str, history):
    """Send a chat message with real AI response."""
    if not agent_id or not message:
        return history, ""
    
    try:
        payload = {
            "agent_id": agent_id,
            "message": message
        }
        response = requests.post(f"{API_BASE}/chat/message", json=payload, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            ai_response = data.get('ai_response', 'No response')
            status = data.get('status', 'unknown')
            
            # Add status indicator to response if not success
            if status != 'success':
                ai_response = f"[{status.upper()}] {ai_response}"
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ai_response})
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ API Error {response.status_code}: {response.text[:200]}"})
        
        return history, ""
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

# ============================================================================
# GRADIO UI
# ============================================================================

def build_ui():
    """Build the enhanced Gradio interface."""
    
    with gr.Blocks(title="Multimodal-DB Enhanced UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🗾 Multimodal-DB Enhanced UI")
        gr.Markdown("Full-featured interface for agent management and AI chat")
        
        # Status Bar
        with gr.Row():
            status_api = gr.Textbox(label="API Status", value=check_api_health(), interactive=False, scale=2)
            status_ollama = gr.Textbox(label="Ollama Status", value=check_ollama_status(), interactive=False, scale=2)
            refresh_status_btn = gr.Button("🔄 Refresh", scale=1)
        
        refresh_status_btn.click(
            lambda: (check_api_health(), check_ollama_status()),
            outputs=[status_api, status_ollama]
        )
        
        # Main Tabs
        with gr.Tabs():
            # Stats Tab
            with gr.Tab("📊 System Stats"):
                gr.Markdown("### Comprehensive System Statistics")
                stats_json = gr.JSON(label="System Metrics")
                refresh_stats_btn = gr.Button("📊 Refresh Stats", variant="primary")
                refresh_stats_btn.click(get_system_stats, outputs=stats_json)
            
            # Agents Tab
            with gr.Tab("🤖 Agent Management"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Create New Agent")
                        agent_name = gr.Textbox(label="Agent Name", placeholder="my_agent")
                        agent_type = gr.Radio(["corecoder", "multimodal"], label="Agent Type", value="corecoder")
                        agent_desc = gr.Textbox(label="Description", placeholder="Optional description", lines=2)
                        agent_tags = gr.Textbox(label="Tags", placeholder="tag1, tag2, tag3")
                        create_btn = gr.Button("✨ Create Agent", variant="primary")
                        create_output = gr.Textbox(label="Result", lines=2)
                        
                        create_btn.click(
                            create_agent,
                            inputs=[agent_name, agent_type, agent_desc, agent_tags],
                            outputs=create_output
                        )
                        
                        gr.Markdown("### Delete Agent")
                        delete_id = gr.Textbox(label="Agent ID to Delete", placeholder="Paste full agent ID here")
                        delete_btn = gr.Button("🗑️ Delete Agent", variant="stop")
                        delete_output = gr.Textbox(label="Result", lines=2)
                        
                        delete_btn.click(delete_agent, inputs=delete_id, outputs=delete_output)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### All Agents (Simple View)")
                        agents_table = gr.Dataframe(
                            headers=["Agent ID", "Name", "Description", "Tags", "Created"],
                            label="Agents",
                            wrap=True
                        )
                        list_agents_btn = gr.Button("📋 Refresh Agent List")
                        list_agents_btn.click(list_agents_simple, outputs=agents_table)
                        
                        gr.Markdown("### Detailed Agent View")
                        agents_detailed = gr.Textbox(label="Full Agent Details", lines=20, max_lines=30)
                        list_detailed_btn = gr.Button("📖 Show Full Details")
                        list_detailed_btn.click(list_agents_detailed, outputs=agents_detailed)
            
            # Content Tab
            with gr.Tab("📝 Content Management"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Content")
                        content_agent_id = gr.Textbox(label="Agent ID", placeholder="Paste agent ID")
                        content_text = gr.Textbox(label="Content", lines=8, placeholder="Enter content...")
                        content_media = gr.Radio(
                            ["text", "document", "embedding", "image", "audio", "video"],
                            label="Media Type",
                            value="text"
                        )
                        content_metadata = gr.Textbox(label="Metadata (JSON)", placeholder='{"key": "value"}')
                        upload_btn = gr.Button("📤 Upload Content", variant="primary")
                        upload_output = gr.Textbox(label="Result")
                        
                        upload_btn.click(
                            upload_content,
                            inputs=[content_agent_id, content_text, content_media, content_metadata],
                            outputs=upload_output
                        )
                    
                    with gr.Column():
                        gr.Markdown("### All Content")
                        content_table = gr.Dataframe(
                            headers=["Content ID", "Agent ID", "Type", "Preview", "Created"],
                            label="Content"
                        )
                        list_content_btn = gr.Button("📋 Refresh Content List")
                        list_content_btn.click(list_content, outputs=content_table)
            
            # Chat Tab
            with gr.Tab("💬 AI Chat"):
                gr.Markdown("### Chat with Your Agents")
                gr.Markdown("⚠️ **Requires Ollama** with model `qwen2.5-coder:3b` running")
                
                chat_agent_id = gr.Textbox(
                    label="Agent ID",
                    placeholder="Paste agent ID from the Agents tab",
                    info="Copy the full agent ID from the agent list above"
                )
                
                chatbox = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=3,
                        scale=4
                    )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("📤 Send", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear Chat", size="lg")
                
                send_btn.click(
                    send_chat,
                    inputs=[chat_agent_id, chat_input, chatbox],
                    outputs=[chatbox, chat_input]
                )
                
                chat_input.submit(
                    send_chat,
                    inputs=[chat_agent_id, chat_input, chatbox],
                    outputs=[chatbox, chat_input]
                )
                
                clear_btn.click(lambda: [], outputs=chatbox)
                
                gr.Markdown("""
                ### 💡 Tips:
                - Create an agent first in the **Agent Management** tab
                - Copy the full Agent ID (the long UUID string)
                - Paste it in the "Agent ID" field above
                - Start chatting! The AI will use the agent's configuration and prompts
                """)
        
        gr.Markdown("---")
        gr.Markdown("**Multimodal-DB v1.0** | [API Docs](http://localhost:8000/docs) | Enhanced UI with Full Features")
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🗾 Multimodal-DB Enhanced UI")
    print("📍 UI: http://localhost:7860")
    print("📍 API: http://localhost:8000")
    print()
    print("Status Check:")
    print(f"  {check_api_health()}")
    print(f"  {check_ollama_status()}")
    print()
    print("🚀 Launching enhanced interface...")
    
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
