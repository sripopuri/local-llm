"""
LM Studio API Client
A Python module to interact with locally hosted LLM via LM Studio API
"""

import requests
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a chat message"""
    role: str  # "user", "assistant", "system"
    content: str


class LMStudioClient:
    """Client for interacting with LM Studio locally hosted LLM"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234"):
        """
        Initialize the LM Studio client
        
        Args:
            base_url: The base URL of the LM Studio API (default: http://127.0.0.1:1234)
        """
        self.base_url = base_url
        self.model = None
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify that the LM Studio API is reachable"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            print("✓ Connected to LM Studio API")
            return True
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to LM Studio API. Make sure LM Studio is running.")
            return False
        except Exception as e:
            print(f"✗ Error connecting to LM Studio API: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            print(f"Available models: {len(models)}")
            for model in models:
                print(f"  - {model.get('id', 'Unknown')}")
            return models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def set_model(self, model_id: str) -> None:
        """Set the active model to use"""
        self.model = model_id
        print(f"Model set to: {model_id}")
    
    def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request
        
        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
        
        Returns:
            Response from the API
        """
        if not self.model:
            print("Error: No model set. Use set_model() first.")
            return {}
        
        try:
            # Convert Message objects to dicts
            messages_list = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            payload = {
                "model": self.model,
                "messages": messages_list,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return {}
    
    def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a text completion request
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
        
        Returns:
            Response from the API
        """
        if not self.model:
            print("Error: No model set. Use set_model() first.")
            return {}
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error in completion: {e}")
            return {}
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Get embeddings for a text
        
        Args:
            text: The text to embed
        
        Returns:
            List of embedding values
        """
        if not self.model:
            print("Error: No model set. Use set_model() first.")
            return None
        
        try:
            payload = {
                "model": self.model,
                "input": text
            }
            
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("data", [{}])[0].get("embedding", [])
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None


def main():
    """Example usage of the LM Studio client"""
    
    # Initialize the client
    client = LMStudioClient()
    
    # Get available models
    models = client.get_available_models()
    
    if models:
        # Set the first available model
        model_id = models[0].get("id")
        client.set_model(model_id)
        
        # Example 1: Simple chat completion
        print("\n--- Chat Completion Example ---")
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="How many states are there in the US?")
        ]
        
        response = client.chat_completion(messages, max_tokens=500)
        
        if response and "choices" in response:
            print(f"Assistant: {response['choices'][0]['message']['content']}")
        
        # # Example 2: Text completion
        # print("\n--- Text Completion Example ---")
        # prompt = "The future of AI is"
        # response = client.completion(prompt, max_tokens=50)
        
        # if response and "choices" in response:
        #     print(f"Completion: {response['choices'][0]['text']}")


if __name__ == "__main__":
    main()
