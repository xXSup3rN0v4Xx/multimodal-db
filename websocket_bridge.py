"""
WebSocket Bridge Server
Connects Next.js WebUI with Chatbot-Python-Core and Multimodal-DB
Port: 2020
"""

import asyncio
import json
import httpx
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="WebSocket Bridge Server", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
CHATBOT_CORE_API = "http://localhost:8000"
MULTIMODAL_DB_API = "http://localhost:8001"


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, agent_id: str, websocket: WebSocket):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        self.active_connections[agent_id] = websocket
        print(f"Agent {agent_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, agent_id: str):
        """Remove WebSocket connection"""
        if agent_id in self.active_connections:
            del self.active_connections[agent_id]
            print(f"Agent {agent_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, agent_id: str, message: dict):
        """Send message to specific WebSocket"""
        if agent_id in self.active_connections:
            try:
                await self.active_connections[agent_id].send_json(message)
            except Exception as e:
                print(f"Error sending message to {agent_id}: {e}")
                self.disconnect(agent_id)


manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "WebSocket Bridge Server",
        "status": "running",
        "active_connections": len(manager.active_connections)
    }


@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """
    Main WebSocket endpoint for chat communication
    
    Message Types (Client -> Server):
    - chat_message: User sends a chat message
    - command: Execute special commands
    
    Message Types (Server -> Client):
    - chat_response: AI response (streaming or complete)
    - error: Error message
    - command_result: Command execution result
    """
    await manager.connect(agent_id, websocket)
    
    try:
        # Get or create agent config from multimodal-db
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                agent_response = await client.get(
                    f"{MULTIMODAL_DB_API}/agents/{agent_id}"
                )
                if agent_response.status_code == 404:
                    # Create default agent if doesn't exist
                    print(f"Creating new agent: {agent_id}")
                    create_response = await client.post(
                        f"{MULTIMODAL_DB_API}/agents/",  # Added trailing slash
                        json={
                            "name": agent_id,
                            "agent_type": "corecoder",
                            "description": f"WebUI Agent {agent_id[:8]}"
                        }
                    )
                    create_response.raise_for_status()
                    agent_config = create_response.json()  # POST now returns full config
                    print(f"Agent created: {agent_config.get('agent_id')}")
                else:
                    agent_response.raise_for_status()
                    agent_config = agent_response.json()
                    print(f"Agent loaded: {agent_config.get('agent_id')}")
            except httpx.ConnectError:
                print(f"Warning: Could not connect to Multimodal-DB API")
                agent_config = {
                    "agent_id": agent_id,
                    "models": {
                        "large_language_model": {
                            "ollama": {
                                "enabled": True,
                                "instances": [{"model": "llama3.2"}]
                            }
                        }
                    }
                }
            except Exception as e:
                print(f"Error getting agent config: {e}")
                agent_config = {"agent_id": agent_id}
        
        while True:
            # Receive message from WebUI
            data = await websocket.receive_json()
            message_type = data.get("type")
            content = data.get("content")
            
            print(f"Received {message_type} from {agent_id}: {str(content)[:100]}")
            
            if message_type == "chat_message":
                await handle_chat_message(agent_id, content, agent_config)
            
            elif message_type == "command":
                await handle_command(agent_id, content)
            
            elif message_type == "vision_detect":
                await handle_vision_detection(agent_id, content)
            
            elif message_type == "speech_to_text":
                await handle_speech_to_text(agent_id, content)
            
            elif message_type == "text_to_speech":
                await handle_text_to_speech(agent_id, content)
            
            elif message_type == "image_generate":
                await handle_image_generation(agent_id, content)
            
            else:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(agent_id)
    except Exception as e:
        print(f"WebSocket error for {agent_id}: {e}")
        manager.disconnect(agent_id)


async def handle_chat_message(agent_id: str, content: str, agent_config: dict):
    """Process chat message with conversation history and AI response"""
    try:
        # Store user message in multimodal-db
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.post(
                    f"{MULTIMODAL_DB_API}/conversations/{agent_id}/messages",
                    json={
                        "role": "user",
                        "content": content
                    }
                )
            except httpx.ConnectError:
                print("Warning: Could not store message in Multimodal-DB")
        
        # Get conversation history for context
        history = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                history_response = await client.get(
                    f"{MULTIMODAL_DB_API}/conversations/{agent_id}/messages",
                    params={"limit": 10}
                )
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    history = history_data.get("messages", [])
            except Exception as e:
                print(f"Warning: Could not get conversation history: {e}")
        
        # Get model configuration
        model_config = agent_config.get("models", {}).get("large_language_model", {}).get("ollama", {})
        model_name = "llama3.2"
        if model_config.get("enabled") and model_config.get("instances"):
            model_name = model_config["instances"][0].get("model", "llama3.2")
        
        print(f"Using model: {model_name}")
        
        # Send to chatbot-python-core for processing
        try:
            # Build conversation history as a single prompt
            conversation_context = ""
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg['content']}\n\n"
            conversation_context += f"User: {content}\n\nAssistant:"
            
            # Stream response from Ollama via chatbot-core
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{CHATBOT_CORE_API}/api/v1/ollama/chat",
                    json={
                        "model": model_name,
                        "prompt": conversation_context,
                        "stream": False  # We'll handle streaming differently if needed
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("success") and result.get("response"):
                    full_response = result["response"]
                    
                    # Send complete message to WebUI
                    await manager.send_message(agent_id, {
                        "type": "chat_response",
                        "content": full_response,
                        "is_stream": False
                    })
                    
                    print(f"Response sent to {agent_id}: {len(full_response)} chars")
            
            # Store assistant response in multimodal-db
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    await client.post(
                        f"{MULTIMODAL_DB_API}/conversations/{agent_id}/messages",
                        json={
                            "role": "assistant",
                            "content": full_response
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not store assistant message: {e}")
        
        except httpx.ConnectError:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": "Could not connect to Chatbot-Python-Core. Is it running on port 8000?"
            })
        except Exception as e:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": f"Error processing message: {str(e)}"
            })
    
    except Exception as e:
        print(f"Error in handle_chat_message: {e}")
        await manager.send_message(agent_id, {
            "type": "error",
            "content": f"Internal error: {str(e)}"
        })


async def handle_command(agent_id: str, content: str):
    """Handle special commands"""
    print(f"Processing command for {agent_id}: {content}")
    
    # Parse command
    if content.startswith("/"):
        parts = content.split()
        command = parts[0]
        
        if command == "/help":
            help_text = """
Available commands:
- /speech_rec on/off - Toggle speech recognition (Whisper)
- /speech_gen on/off - Toggle speech generation (Kokoro/VibeVoice)
- /vision_on - Enable vision detection (YOLO)
- /vision_off - Disable vision detection
- /image_gen_on - Enable image generation (SDXL)
- /image_gen_off - Disable image generation
- /avatar_sync_on - Enable avatar lip sync (SadTalker)
- /avatar_sync_off - Disable avatar lip sync
- /clear - Clear conversation history
- /models - List available models
- /help - Show this help message
"""
            await manager.send_message(agent_id, {
                "type": "command_result",
                "content": help_text
            })
        
        elif command == "/clear":
            # Clear conversation history in multimodal-db
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.delete(
                        f"{MULTIMODAL_DB_API}/api/v1/conversations/{agent_id}"
                    )
                await manager.send_message(agent_id, {
                    "type": "command_result",
                    "content": "✓ Conversation history cleared"
                })
            except Exception as e:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Failed to clear history: {str(e)}"
                })
        
        elif command == "/models":
            # Get available models from both services
            try:
                models_info = []
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Get Ollama models
                    try:
                        response = await client.get(f"{CHATBOT_CORE_API}/api/v1/ollama/models")
                        if response.status_code == 200:
                            data = response.json()
                            models_info.append(f"📝 Ollama Models: {', '.join(data.get('models', []))}")
                    except:
                        models_info.append("⚠ Ollama API unavailable")
                
                await manager.send_message(agent_id, {
                    "type": "command_result",
                    "content": "\n".join(models_info)
                })
            except Exception as e:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Failed to get models: {str(e)}"
                })
        
        elif command in ["/speech_rec", "/speech_gen", "/vision_on", "/vision_off", 
                        "/image_gen_on", "/image_gen_off", "/avatar_sync_on", "/avatar_sync_off"]:
            # Store feature state in agent config
            feature_map = {
                "/speech_rec": "speech_recognition",
                "/speech_gen": "speech_generation",
                "/vision_on": "vision_detection",
                "/vision_off": "vision_detection",
                "/image_gen_on": "image_generation",
                "/image_gen_off": "image_generation",
                "/avatar_sync_on": "avatar_lip_sync",
                "/avatar_sync_off": "avatar_lip_sync"
            }
            
            feature = feature_map.get(command)
            enabled = not command.endswith("_off")
            
            # Update agent config (simplified - just acknowledge)
            try:
                # TODO: Implement agent feature updates in multimodal-db API
                status = "enabled" if enabled else "disabled"
                await manager.send_message(agent_id, {
                    "type": "command_result",
                    "content": f"✓ {feature.replace('_', ' ').title()} {status} (note: agent config updates not yet implemented)"
                })
            except Exception as e:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Failed to update feature: {str(e)}"
                })
        
        else:
            await manager.send_message(agent_id, {
                "type": "command_result",
                "content": f"Command executed: {content}"
            })


async def handle_vision_detection(agent_id: str, content: dict):
    """Handle vision detection requests"""
    print(f"Processing vision detection for {agent_id}")
    
    try:
        # Extract image and model
        image = content.get("image")
        model = content.get("model", "yolov8n")
        
        if not image:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": "No image provided"
            })
            return
        
        # Call chatbot-python-core vision API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CHATBOT_CORE_API}/api/v1/vision/detect",
                json={
                    "image": image,
                    "model": model,
                    "confidence_threshold": 0.5
                }
            )
            
            if response.status_code == 200:
                detections = response.json()
                
                # Store detections in multimodal-db
                try:
                    await client.post(
                        f"{MULTIMODAL_DB_API}/api/v1/detections/store",
                        json={
                            "agent_id": agent_id,
                            "detections": detections.get("detections", []),
                            "model": model
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not store detections: {e}")
                
                # Send results to WebUI
                await manager.send_message(agent_id, {
                    "type": "vision_results",
                    "content": detections
                })
            else:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Vision detection failed: {response.text}"
                })
    
    except Exception as e:
        await manager.send_message(agent_id, {
            "type": "error",
            "content": f"Vision detection error: {str(e)}"
        })


async def handle_speech_to_text(agent_id: str, content: dict):
    """Handle speech-to-text (Whisper) requests"""
    print(f"Processing speech-to-text for {agent_id}")
    
    try:
        audio_data = content.get("audio")
        model = content.get("model", "whisper")
        
        if not audio_data:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": "No audio data provided"
            })
            return
        
        # Call chatbot-python-core speech recognition API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CHATBOT_CORE_API}/api/v1/audio/stt",
                json={
                    "audio": audio_data,
                    "model": model
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get("text", "")
                
                # Send transcribed text back to WebUI
                await manager.send_message(agent_id, {
                    "type": "speech_to_text_result",
                    "content": transcribed_text
                })
                
                # Optionally auto-send as chat message
                if content.get("auto_send", True):
                    await handle_chat_message(agent_id, transcribed_text, {})
            else:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Speech recognition failed: {response.text}"
                })
    
    except Exception as e:
        await manager.send_message(agent_id, {
            "type": "error",
            "content": f"Speech-to-text error: {str(e)}"
        })


async def handle_text_to_speech(agent_id: str, content: dict):
    """Handle text-to-speech (Kokoro/VibeVoice) requests"""
    print(f"Processing text-to-speech for {agent_id}")
    
    try:
        text = content.get("text")
        model = content.get("model", "kokoro")
        voice = content.get("voice", "af_sarah")
        
        if not text:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": "No text provided"
            })
            return
        
        # Call chatbot-python-core TTS API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CHATBOT_CORE_API}/api/v1/audio/tts",
                json={
                    "text": text,
                    "model": model,
                    "voice": voice
                }
            )
            
            if response.status_code == 200:
                # Response should be audio file or base64 encoded audio
                audio_data = response.content
                
                # Send audio back to WebUI
                await manager.send_message(agent_id, {
                    "type": "text_to_speech_result",
                    "content": {
                        "audio": audio_data.decode() if isinstance(audio_data, bytes) else audio_data,
                        "format": "wav"
                    }
                })
            else:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Text-to-speech failed: {response.text}"
                })
    
    except Exception as e:
        await manager.send_message(agent_id, {
            "type": "error",
            "content": f"Text-to-speech error: {str(e)}"
        })


async def handle_image_generation(agent_id: str, content: dict):
    """Handle image generation (SDXL) requests"""
    print(f"Processing image generation for {agent_id}")
    
    try:
        prompt = content.get("prompt")
        
        if not prompt:
            await manager.send_message(agent_id, {
                "type": "error",
                "content": "No prompt provided"
            })
            return
        
        # Notify user that generation is starting (can take a while)
        await manager.send_message(agent_id, {
            "type": "image_generation_status",
            "content": "🎨 Generating image... This may take 30-60 seconds."
        })
        
        # Call chatbot-python-core image generation API
        async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for image generation
            response = await client.post(
                f"{CHATBOT_CORE_API}/api/v1/image/generate",
                json={
                    "prompt": prompt,
                    "steps": content.get("steps", 30),
                    "guidance_scale": content.get("guidance_scale", 7.5)
                }
            )
            
            if response.status_code == 200:
                # Response should be image file or base64 encoded image
                image_data = response.content
                
                # Store image in multimodal-db
                try:
                    async with httpx.AsyncClient(timeout=10.0) as db_client:
                        await db_client.post(
                            f"{MULTIMODAL_DB_API}/api/v1/media/store",
                            json={
                                "agent_id": agent_id,
                                "media_type": "image",
                                "data": image_data.decode() if isinstance(image_data, bytes) else image_data,
                                "prompt": prompt
                            }
                        )
                except Exception as e:
                    print(f"Warning: Could not store image: {e}")
                
                # Send image back to WebUI
                await manager.send_message(agent_id, {
                    "type": "image_generation_result",
                    "content": {
                        "image": image_data.decode() if isinstance(image_data, bytes) else image_data,
                        "prompt": prompt
                    }
                })
            else:
                await manager.send_message(agent_id, {
                    "type": "error",
                    "content": f"Image generation failed: {response.text}"
                })
    
    except Exception as e:
        await manager.send_message(agent_id, {
            "type": "error",
            "content": f"Image generation error: {str(e)}"
        })


@app.websocket("/audio-stream/{agent_id}")
async def audio_stream_endpoint(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for audio streaming
    
    Sends real-time audio data for visualization:
    - user_audio_data: Audio from user microphone
    - llm_audio_data: Audio from LLM speech generation
    """
    await websocket.accept()
    print(f"Audio stream connected for {agent_id}")
    
    try:
        while True:
            # TODO: Integrate with actual audio processing
            # For now, send dummy data for visualization
            audio_data = {
                "user_audio_data": [0] * 128,  # Replace with actual user audio
                "llm_audio_data": [0] * 128    # Replace with actual LLM audio
            }
            await websocket.send_json(audio_data)
            await asyncio.sleep(0.05)  # 20 FPS
    
    except WebSocketDisconnect:
        print(f"Audio stream disconnected for {agent_id}")
    except Exception as e:
        print(f"Audio stream error for {agent_id}: {e}")


@app.get("/available_models")
async def get_available_models():
    """Get available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{CHATBOT_CORE_API}/api/v1/ollama/models")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error getting models: {e}")
    
    # Return default models if API unavailable
    return {
        "models": [
            "llama3.2",
            "llama3.2:1b",
            "mistral",
            "codellama",
            "qwen2.5-coder:7b"
        ]
    }


@app.post("/set_model")
async def set_model(data: dict):
    """Set active model for an agent"""
    agent_id = data.get("agent_id")
    model = data.get("model")
    
    if not agent_id or not model:
        return {"status": "error", "message": "Missing agent_id or model"}
    
    try:
        # TODO: Implement agent model updates in multimodal-db API
        # For now, just return success - model selection happens in chatbot-core
        return {"status": "success", "model": model, "note": "Model selection handled by chatbot-core API"}
    except Exception as e:
        print(f"Error setting model: {e}")
    
    return {"status": "error", "message": "Could not update model"}


if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Bridge Server Starting...")
    print("=" * 60)
    print(f"Port: 2020")
    print(f"WebSocket endpoints:")
    print(f"  - ws://localhost:2020/ws/{{agent_id}}")
    print(f"  - ws://localhost:2020/audio-stream/{{agent_id}}")
    print(f"HTTP endpoints:")
    print(f"  - http://localhost:2020/available_models")
    print(f"  - http://localhost:2020/set_model")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=2020,
        log_level="info"
    )
