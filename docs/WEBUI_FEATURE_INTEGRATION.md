# Next.js WebUI Feature Integration Guide

**Date:** October 15, 2025  
**Version:** 2.0.0

This document explains which features are connected between the Next.js WebUI, WebSocket Bridge, Chatbot-Python-Core, and Multimodal-DB.

---

## 🎯 Feature Status Overview

| Feature | UI Button | Backend Connected | Storage | Status |
|---------|-----------|-------------------|---------|--------|
| **Chat (Ollama)** | Main chat box | ✅ Full | ✅ Yes | 🟢 **WORKING** |
| **Speech Recognition** | Mic button (side panel) | ✅ Full | ✅ Yes | 🟢 **READY** |
| **Speech Generation** | Volume button (side panel) | ✅ Full | ✅ Yes | 🟢 **READY** |
| **Vision Detection** | Eye button (side panel) | ✅ Full | ✅ Yes | 🟢 **READY** |
| **Image Generation** | Image button (side panel) | ✅ Full | ✅ Yes | 🟢 **READY** |
| **Avatar Lip Sync** | Smile button (side panel) | ⚠️ Partial | ❌ No | 🟡 **PLANNED** |
| **Audio Visualization** | Waveform display | ⚠️ Dummy data | ❌ No | 🟡 **NEEDS WORK** |

**Legend:**
- 🟢 **WORKING** - Fully functional, tested
- 🟢 **READY** - Backend implemented, needs frontend integration
- 🟡 **PLANNED** - Partially implemented
- 🔴 **NOT CONNECTED** - UI only

---

## ✅ Fully Working Features

### 1. Chat with Ollama (Main Feature)

**What it does:**
- User types message in chat box
- Message sent to Ollama via Chatbot-Python-Core
- Response streamed back in real-time
- Full conversation history stored in Multimodal-DB

**How to use:**
1. Type message in the main chat input
2. Press Enter or click Send
3. Watch AI response appear in real-time
4. All conversation automatically saved

**Technical flow:**
```
User Input → WebSocket → Bridge → Multimodal-DB (store) → 
Bridge → Chatbot-Core (Ollama) → Bridge (stream response) → 
Multimodal-DB (store) → WebSocket → User Display
```

**Message format:**
```javascript
// Frontend sends:
ws.send(JSON.stringify({
  type: "chat_message",
  content: "Hello, AI!"
}))

// Backend streams:
{
  type: "chat_response",
  content: "Hello! How can I help you?",
  is_stream: true/false
}
```

---

## 🟢 Ready Features (Backend Implemented)

These features have full backend support but need frontend integration.

### 2. Speech Recognition (Whisper)

**Status:** Backend ready, frontend needs audio capture

**What it does:**
- Capture audio from microphone
- Send to Whisper for transcription
- Transcribed text auto-sent as chat message
- Audio and transcription stored

**Backend message format:**
```javascript
// Frontend should send:
ws.send(JSON.stringify({
  type: "speech_to_text",
  content: {
    audio: "<base64_encoded_audio>",
    model: "whisper",
    auto_send: true  // Auto-send transcribed text as chat
  }
}))

// Backend responds:
{
  type: "speech_to_text_result",
  content: "transcribed text here"
}
```

**Frontend TODO:**
```javascript
// Need to add in WebUI:
1. Capture audio from microphone when mic button clicked
2. Convert to base64 or send as blob
3. Send via WebSocket with type "speech_to_text"
4. Display transcribed text
5. Optionally auto-send to chat
```

### 3. Speech Generation (Kokoro/VibeVoice)

**Status:** Backend ready, frontend needs audio playback

**What it does:**
- Convert AI response text to speech
- Send audio back to frontend
- Play audio through speakers
- Audio files stored

**Backend message format:**
```javascript
// Frontend should send:
ws.send(JSON.stringify({
  type: "text_to_speech",
  content: {
    text: "Text to speak",
    model: "kokoro",  // or "vibevoice" or "f5"
    voice: "af_sarah"
  }
}))

// Backend responds:
{
  type: "text_to_speech_result",
  content: {
    audio: "<base64_encoded_audio>",
    format: "wav"
  }
}
```

**Frontend TODO:**
```javascript
// Need to add in WebUI:
1. When volume button toggled on, auto-convert AI responses to speech
2. Receive audio from WebSocket
3. Decode base64 audio
4. Play through Audio API
5. Update audio visualizer with real data
```

### 4. Vision Detection (YOLO)

**Status:** Backend ready, frontend needs image capture

**What it does:**
- Capture image from camera or upload file
- Send to YOLO for object detection
- Display detected objects with bounding boxes
- Detections stored in database

**Backend message format:**
```javascript
// Frontend should send:
ws.send(JSON.stringify({
  type: "vision_detect",
  content: {
    image: "<base64_encoded_image>",
    model: "yolov8n"
  }
}))

// Backend responds:
{
  type: "vision_results",
  content: {
    detections: [
      {
        class_name: "person",
        confidence: 0.95,
        bbox: [x, y, width, height]
      },
      // ... more detections
    ]
  }
}
```

**Frontend TODO:**
```javascript
// Need to add in WebUI:
1. Add image upload or camera capture
2. Convert image to base64
3. Send via WebSocket with type "vision_detect"
4. Display detection results
5. Draw bounding boxes on image
6. Show detected object names and confidence scores
```

### 5. Image Generation (SDXL)

**Status:** Backend ready, frontend needs UI integration

**What it does:**
- User provides text prompt
- SDXL generates image (takes 30-60 seconds)
- Image displayed in chat
- Image stored in multimodal-db

**Backend message format:**
```javascript
// Frontend should send:
ws.send(JSON.stringify({
  type: "image_generate",
  content: {
    prompt: "A sunset over mountains",
    steps: 30,
    guidance_scale: 7.5
  }
}))

// Backend sends status:
{
  type: "image_generation_status",
  content: "🎨 Generating image... This may take 30-60 seconds."
}

// Then sends result:
{
  type: "image_generation_result",
  content: {
    image: "<base64_encoded_image>",
    prompt: "original prompt"
  }
}
```

**Frontend TODO:**
```javascript
// Need to add in WebUI:
1. Add image generation prompt input (could be in chat)
2. Parse messages starting with "/image" or use button
3. Send via WebSocket with type "image_generate"
4. Show "generating..." loading state
5. Display generated image when received
6. Allow download/save
```

---

## 🟡 Partially Implemented Features

### 6. Avatar Lip Sync (SadTalker)

**Status:** Command handling only, no actual integration yet

**What it would do:**
- Sync avatar mouth movements with generated speech
- Requires SadTalker model integration
- Needs video generation pipeline

**Current state:**
- Command `/avatar_sync_on` toggles feature flag
- No actual video generation yet
- Planned for future release

### 7. Audio Visualization

**Status:** UI exists but shows dummy data

**Current behavior:**
- Waveform components display in UI
- Currently shows zeros/dummy data
- Need real audio stream from mic and TTS

**What's needed:**
```javascript
// WebSocket audio endpoint exists at:
ws://localhost:2020/audio-stream/{agent_id}

// Currently sends dummy data:
{
  user_audio_data: [0, 0, 0, ...],  // Float32Array of audio samples
  llm_audio_data: [0, 0, 0, ...]
}

// TODO: Replace with real audio data from:
// - User microphone input (when speech_rec enabled)
// - Generated speech output (when speech_gen enabled)
```

---

## 🎮 Side Panel Buttons

The WebUI has a side panel with toggle buttons. Here's what they do:

### User Button (Top)
- Opens user profile dialog
- Configure user name
- Save preferences

### Mic Button (Speech Recognition)
**Status:** 🟢 Backend ready

**What it should do:**
1. Click to toggle speech recognition on/off
2. When ON: Capture audio from microphone
3. Send audio chunks to `/speech_to_text` endpoint
4. Display transcribed text
5. Auto-send as chat message
6. Button glows green when active

**Current behavior:**
- Button toggles state
- Sends command to backend
- Backend ready to process audio
- ❌ Frontend doesn't capture audio yet

### Volume Button (Speech Generation)
**Status:** 🟢 Backend ready

**What it should do:**
1. Click to toggle text-to-speech on/off
2. When ON: Automatically convert all AI responses to speech
3. Play speech through speakers
4. Update audio visualizer
5. Button glows green when active

**Current behavior:**
- Button toggles state
- Sends command to backend
- Backend ready to generate speech
- ❌ Frontend doesn't request/play audio yet

### Eye Button (Vision Detection)
**Status:** 🟢 Backend ready

**What it should do:**
1. Click to enable vision detection
2. Opens camera or file picker
3. Continuously or on-demand detect objects
4. Display bounding boxes and labels
5. Store detections in database

**Current behavior:**
- Button toggles state
- Sends command to backend
- Backend ready to process images
- ❌ Frontend doesn't capture/send images yet

### Image Button (Image Generation)
**Status:** 🟢 Backend ready

**What it should do:**
1. Click to enable image generation mode
2. Parse chat messages for prompts
3. Generate images with SDXL
4. Display in chat
5. Allow save/download

**Current behavior:**
- Button toggles state
- Sends command to backend
- Backend ready to generate images
- ❌ Frontend doesn't send prompts yet

### Smile Button (Avatar Lip Sync)
**Status:** 🟡 Planned

**What it should do:**
1. Enable avatar mouth sync with speech
2. Generate talking head video
3. Display animated avatar

**Current behavior:**
- Button toggles state
- ❌ Feature not fully implemented

### Settings Button (Bottom)
**Status:** ✅ Working

- Opens settings dialog
- Toggle dark mode
- Configure API URLs
- Add/remove UI components

---

## 📡 WebSocket Message Types Reference

### From Frontend to Bridge

| Type | Purpose | Content Format |
|------|---------|----------------|
| `chat_message` | Send chat message | `string` |
| `command` | Execute command | `string` (e.g., "/help") |
| `vision_detect` | Detect objects | `{image, model}` |
| `speech_to_text` | Transcribe audio | `{audio, model, auto_send}` |
| `text_to_speech` | Generate speech | `{text, model, voice}` |
| `image_generate` | Generate image | `{prompt, steps, guidance_scale}` |

### From Bridge to Frontend

| Type | Purpose | Content Format |
|------|---------|----------------|
| `chat_response` | AI response | `{content, is_stream}` |
| `error` | Error message | `string` |
| `command_result` | Command output | `string` |
| `vision_results` | Detection results | `{detections: [...]}` |
| `speech_to_text_result` | Transcription | `string` |
| `text_to_speech_result` | Audio data | `{audio, format}` |
| `image_generation_status` | Progress update | `string` |
| `image_generation_result` | Generated image | `{image, prompt}` |

---

## 🔧 Frontend Integration TODOs

### High Priority (Core Multimodal Features)

1. **Speech Recognition Integration**
   ```javascript
   // Add to ChatSection component:
   const startRecording = async () => {
     const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
     const mediaRecorder = new MediaRecorder(stream)
     // ... record audio, send to WebSocket
   }
   ```

2. **Speech Generation Integration**
   ```javascript
   // Add to WebSocket message handler:
   if (data.type === 'text_to_speech_result') {
     const audioBlob = base64ToBlob(data.content.audio)
     const audioUrl = URL.createObjectURL(audioBlob)
     const audio = new Audio(audioUrl)
     audio.play()
   }
   ```

3. **Vision Detection Integration**
   ```javascript
   // Add camera capture or file upload:
   const captureImage = async () => {
     const video = document.querySelector('video')
     const canvas = document.createElement('canvas')
     canvas.getContext('2d').drawImage(video, 0, 0)
     const base64 = canvas.toDataURL().split(',')[1]
     
     ws.send(JSON.stringify({
       type: 'vision_detect',
       content: { image: base64, model: 'yolov8n' }
     }))
   }
   ```

4. **Image Generation Integration**
   ```javascript
   // Parse chat messages for image prompts:
   if (message.startsWith('/image ')) {
     const prompt = message.substring(7)
     ws.send(JSON.stringify({
       type: 'image_generate',
       content: { prompt, steps: 30 }
     }))
   }
   ```

### Medium Priority

5. **Real Audio Visualization**
   - Connect microphone input to AudioVisualizer
   - Connect TTS output to AudioVisualizer
   - Replace dummy data with real audio samples

6. **Detection Results Display**
   - Create bounding box overlay component
   - Show detection labels and confidence
   - Allow clicking detections for details

7. **Image Gallery**
   - Display generated images in chat
   - Allow saving/downloading
   - Show generation parameters

### Low Priority

8. **Avatar Lip Sync** (requires SadTalker implementation)
9. **Advanced Settings** (model parameters, voices, etc.)
10. **History Browsing** (view past detections, images, transcriptions)

---

## 🎓 Example Frontend Code

### Complete Speech Recognition Example

```javascript
// components/SpeechRecognition.jsx
import { useState, useRef } from 'react'

export default function SpeechRecognition({ ws, enabled }) {
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' })
        const base64 = await blobToBase64(audioBlob)
        
        // Send to backend
        ws.send(JSON.stringify({
          type: 'speech_to_text',
          content: {
            audio: base64,
            model: 'whisper',
            auto_send: true
          }
        }))
        
        chunksRef.current = []
      }
      
      mediaRecorder.start()
      mediaRecorderRef.current = mediaRecorder
      setIsRecording(true)
    } catch (error) {
      console.error('Failed to start recording:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
      setIsRecording(false)
    }
  }

  const blobToBase64 = (blob) => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result.split(',')[1])
      reader.readAsDataURL(blob)
    })
  }

  return (
    <button 
      onClick={isRecording ? stopRecording : startRecording}
      disabled={!enabled}
    >
      {isRecording ? '⏹ Stop' : '🎤 Record'}
    </button>
  )
}
```

### Complete Vision Detection Example

```javascript
// components/VisionDetection.jsx
import { useState, useRef } from 'react'

export default function VisionDetection({ ws, enabled }) {
  const [detections, setDetections] = useState([])
  const [image, setImage] = useState(null)
  const videoRef = useRef(null)

  const captureFromCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    videoRef.current.srcObject = stream
    
    // Capture frame after 1 second
    setTimeout(() => {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0)
      
      const base64 = canvas.toDataURL('image/jpeg').split(',')[1]
      setImage(canvas.toDataURL('image/jpeg'))
      
      // Send to backend
      ws.send(JSON.stringify({
        type: 'vision_detect',
        content: {
          image: base64,
          model: 'yolov8n'
        }
      }))
      
      // Stop camera
      stream.getTracks().forEach(track => track.stop())
    }, 1000)
  }

  // Listen for results
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type === 'vision_results') {
      setDetections(data.content.detections)
    }
  }

  return (
    <div>
      <button onClick={captureFromCamera} disabled={!enabled}>
        📷 Detect Objects
      </button>
      
      <video ref={videoRef} style={{ display: 'none' }} autoPlay />
      
      {image && (
        <div style={{ position: 'relative' }}>
          <img src={image} alt="Captured" />
          {detections.map((det, idx) => (
            <div
              key={idx}
              style={{
                position: 'absolute',
                left: det.bbox[0],
                top: det.bbox[1],
                width: det.bbox[2],
                height: det.bbox[3],
                border: '2px solid red',
                color: 'red',
                fontSize: '12px',
                background: 'rgba(0,0,0,0.5)'
              }}
            >
              {det.class_name} ({(det.confidence * 100).toFixed(1)}%)
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

---

## 🚀 Getting Started

### For Backend Developers

1. ✅ WebSocket Bridge is ready
2. ✅ All multimodal endpoints implemented
3. ✅ Conversation storage working
4. ✅ Detection storage ready
5. Just restart the bridge: `python websocket_bridge.py`

### For Frontend Developers

1. Read this document thoroughly
2. Pick a feature to integrate (start with Speech Recognition)
3. Add message sending logic (see examples above)
4. Add response handling logic
5. Test with backend
6. Move to next feature

### Testing

```bash
# Test speech recognition endpoint
curl -X POST http://localhost:8000/api/v1/audio/stt \
  -H "Content-Type: application/json" \
  -d '{"audio": "base64_data", "model": "whisper"}'

# Test vision detection
curl -X POST http://localhost:8000/api/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image", "model": "yolov8n"}'

# Test image generation
curl -X POST http://localhost:8000/api/v1/image/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A sunset", "steps": 30}'
```

---

## 📚 Additional Resources

- [Chatbot-Python-Core API Docs](http://localhost:8000/docs)
- [Multimodal-DB API Docs](http://localhost:8001/docs)
- [WebSocket Bridge Source](../websocket_bridge.py)
- [Next.js WebUI Source](../../chatbot-nextjs-webui/)

---

**Ready to integrate? Start with Speech Recognition - it's the easiest!** 🎤
