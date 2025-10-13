"""
Simple Gradio UI for Multimodal-DB API Testing
Minimal, functional interface to test all API endpoints.
"""
import gradio as gr
import requests
from typing import Dict, Any

API_BASE = "http://localhost:8000"

# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is online."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=2)
        if response.status_code == 200:
            return "✅ API is online!"
        return f"⚠️ API returned {response.status_code}"
    except:
        return "❌ API is offline!"

def get_system_stats():
    """Get system statistics."""
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
            return f"✅ Created agent: {data.get('agent_id', 'N/A')[:16]}..."
        return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_agents():
    """List all agents."""
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
                    a.get('agent_type', ''),
                    a.get('description', '')[:50] + '...' if a.get('description') else '',
                    a.get('created_at', '')[:19] if a.get('created_at') else ''
                ]
                for a in agents
            ]
        return [["Error", str(response.status_code), "", "", ""]]
    except Exception as e:
        return [["Error", str(e), "", "", ""]]

def upload_content(agent_id: str, content: str, media_type: str, metadata: str = ""):
    """Upload content for an agent."""
    try:
        import json
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
    """Send a chat message."""
    try:
        payload = {
            "agent_id": agent_id,
            "message": message
        }
        response = requests.post(f"{API_BASE}/chat/", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            reply = data.get('ai_response', 'No response')
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return history, ""
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {response.status_code}"})
            return history, ""
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

# ============================================================================
# GRADIO UI
# ============================================================================

def build_ui():
    """Build the Gradio interface."""
    
    with gr.Blocks(title="Multimodal-DB API Tester") as demo:
        gr.Markdown("# 🗾 Multimodal-DB API Tester")
        gr.Markdown("Simple interface to test all API endpoints")
        
        # Status Tab
        with gr.Tab("📊 Status"):
            gr.Markdown("### System Status")
            status_text = gr.Textbox(label="API Status", value=check_api_health())
            refresh_status = gr.Button("🔄 Refresh Status")
            refresh_status.click(check_api_health, outputs=status_text)
            
            gr.Markdown("### System Statistics")
            stats_json = gr.JSON(label="Stats")
            refresh_stats = gr.Button("📊 Get Stats")
            refresh_stats.click(get_system_stats, outputs=stats_json)
        
        # Agents Tab
        with gr.Tab("🤖 Agents"):
            gr.Markdown("### Create Agent")
            with gr.Row():
                agent_name = gr.Textbox(label="Name", placeholder="my_agent")
                agent_type = gr.Radio(["corecoder", "multimodal"], label="Type", value="corecoder")
            
            agent_desc = gr.Textbox(label="Description (optional)", placeholder="Agent description")
            agent_tags = gr.Textbox(label="Tags (optional)", placeholder="tag1, tag2")
            create_btn = gr.Button("✨ Create Agent", variant="primary")
            create_output = gr.Textbox(label="Result")
            
            create_btn.click(
                create_agent,
                inputs=[agent_name, agent_type, agent_desc, agent_tags],
                outputs=create_output
            )
            
            gr.Markdown("### List Agents")
            agents_table = gr.Dataframe(
                headers=["Agent ID", "Name", "Type", "Description", "Created"],
                label="Agents"
            )
            list_agents_btn = gr.Button("📋 List All Agents")
            list_agents_btn.click(list_agents, outputs=agents_table)
        
        # Content Tab
        with gr.Tab("📝 Content"):
            gr.Markdown("### Upload Content")
            content_agent_id = gr.Textbox(label="Agent ID", placeholder="Paste agent ID here")
            content_text = gr.Textbox(label="Content", lines=5, placeholder="Enter content...")
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
            
            gr.Markdown("### List Content")
            content_table = gr.Dataframe(
                headers=["Content ID", "Agent ID", "Media Type", "Preview", "Created"],
                label="Content"
            )
            list_content_btn = gr.Button("📋 List All Content")
            list_content_btn.click(list_content, outputs=content_table)
        
        # Chat Tab
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Chat with Agent")
            chat_agent_id = gr.Textbox(label="Agent ID", placeholder="Paste agent ID here")
            chatbox = gr.Chatbot(label="Conversation", height=400, type="messages")
            
            with gr.Row():
                chat_input = gr.Textbox(label="Message", placeholder="Type message...", scale=4)
                send_btn = gr.Button("📤 Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear Chat")
            
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
        
        gr.Markdown("---")
        gr.Markdown("**Multimodal-DB v1.0** | [API Docs](http://localhost:8000/docs)")
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🗾 Multimodal-DB Simple UI")
    print("📍 UI: http://localhost:7860")
    print("📍 API: http://localhost:8000")
    print()
    print(check_api_health())
    print()
    print("🚀 Launching...")
    
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
