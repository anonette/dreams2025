"""
Multi-model LLM interface for dream generation.
Supports various LLM providers and models, including OpenAI, Anthropic, and OpenRouter.
"""

from typing import Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

@dataclass
class GenerationConfig:
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class LLMInterface:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.setup_clients()
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def setup_clients(self):
        """Initialize API clients for different providers."""
        if 'openai' in self.api_keys and OPENAI_AVAILABLE:
            openai.api_key = self.api_keys['openai']
        if 'anthropic' in self.api_keys and ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
        # OpenRouter uses HTTP API, no client needed
        
    async def generate_dream(self, prompt: str, config: GenerationConfig) -> str:
        """Generate dream narrative using specified model and parameters."""
        try:
            if 'gpt' in config.model.lower():
                return self._generate_openai(prompt, config)
            elif 'claude' in config.model.lower():
                return await self._generate_anthropic(prompt, config)
            elif 'openrouter' in config.model.lower() or any(x in config.model.lower() for x in ['mistral', 'deepseek']):
                return await self._generate_openrouter(prompt, config)
            else:
                raise ValueError(f"Unsupported model: {config.model}")
        except Exception as e:
            logging.error(f"Error generating dream: {e}")
            return ""
    
    def _generate_openai(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using OpenAI models (openai>=1.0.0)."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        # Prepare parameters, only include max_tokens if specified
        params = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }
        
        # Only add max_tokens if it's specified
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        
        # Use the synchronous API for chat completions
        response = openai.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    
    async def _generate_anthropic(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using Anthropic Claude models."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        # Prepare parameters, only include max_tokens if specified
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Only add max_tokens if it's specified
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        
        response = await self.anthropic_client.messages.create(**params)
        return response.content[0].text.strip()

    async def _generate_openrouter(self, prompt: str, config: GenerationConfig) -> str:
        """
        Generate using OpenRouter API (supports models like mistral, deepseek, etc).
        Requires 'openrouter' key in api_keys dict.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx library not available. Install with: pip install httpx")
        
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('openrouter', '')}",
            "Content-Type": "application/json"
        }
        
        # Prepare data, only include max_tokens if specified
        data = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        # Only add max_tokens if it's specified
        if config.max_tokens is not None:
            data["max_tokens"] = config.max_tokens
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.openrouter_url, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            # OpenRouter returns choices like OpenAI
            return result['choices'][0]['message']['content'].strip()
