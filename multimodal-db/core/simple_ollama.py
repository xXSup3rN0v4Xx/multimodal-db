"""
Simple Ollama Integration for multimodal-db
Minimal, fast, focused on qwen2.5-coder:3b
"""
import subprocess
from typing import Dict, Any


class SimpleOllamaClient:
    """Lightweight Ollama client optimized for qwen2.5-coder:3b."""
    
    def __init__(self, model: str = "qwen2.5-coder:3b", timeout: int = 30):
        self.model = model
        self.timeout = timeout
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Fast availability check."""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and self.model in result.stdout
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Generate response with optional system prompt."""
        if not self.available:
            return {"content": "Model not available", "success": False}
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model, full_prompt],
                capture_output=True, text=True, timeout=self.timeout
            )
            
            if result.returncode == 0:
                return {"content": result.stdout.strip(), "success": True}
            else:
                return {"content": f"Error: {result.stderr}", "success": False}
        except subprocess.TimeoutExpired:
            return {"content": "Timeout", "success": False}
        except Exception as e:
            return {"content": f"Error: {e}", "success": False}