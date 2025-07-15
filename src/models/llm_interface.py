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
    
    def setup_google_gemini_client(self):
        """Setup OpenAI client to use Google's Gemini API endpoint."""
        if OPENAI_AVAILABLE and 'openai' in self.api_keys:
            # Create OpenAI client configured for Google's endpoint
            self.openai_client = openai.OpenAI(
                api_key=self.api_keys['openai'],  # This will be the Gemini API key
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        
    async def generate_dream(self, prompt: str, config: GenerationConfig, system_message: Optional[str] = None) -> str:
        """Generate dream narrative using specified model and parameters."""
        try:
            # Route models based on their provider prefix or characteristics
            if 'gemini' in config.model.lower() and not config.model.startswith(('google/', 'openai/')):
                return await self._generate_openai(prompt, config, system_message)  # Use OpenAI-compatible endpoint
            elif config.model.startswith('openai/') or 'gpt' in config.model.lower():
                return await self._generate_openrouter(prompt, config, system_message)  # OpenRouter for OpenAI models
            elif config.model.startswith('anthropic/') or 'claude' in config.model.lower():
                return await self._generate_openrouter(prompt, config, system_message)  # OpenRouter for Anthropic models
            elif any(config.model.startswith(prefix) for prefix in ['mistralai/', 'google/', 'meta-llama/', 'qwen/', 'deepseek/']):
                return await self._generate_openrouter(prompt, config, system_message)  # OpenRouter for other providers
            elif 'openrouter' in config.model.lower() or any(x in config.model.lower() for x in ['mistral', 'deepseek']):
                return await self._generate_openrouter(prompt, config, system_message)
            else:
                # Default to OpenRouter for unknown models (likely they're available there)
                return await self._generate_openrouter(prompt, config, system_message)
        except Exception as e:
            logging.error(f"Error generating dream: {e}")
            raise e  # Re-raise to see the actual error
    
    async def _generate_openai(self, prompt: str, config: GenerationConfig, system_message: Optional[str] = None) -> str:
        """Generate using OpenAI models (openai>=1.0.0) or Google Gemini via OpenAI compatibility."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        def _sync_call():
            # Prepare messages with optional system message
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters, only include max_tokens if specified
            params = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty
            }
            
            # Only add max_tokens if it's specified
            if config.max_tokens is not None:
                params["max_tokens"] = config.max_tokens

            # For Gemini, remove unsupported parameters
            if 'gemini' in config.model.lower():
                params.pop("frequency_penalty", None)
                params.pop("presence_penalty", None)
            
            # Use custom client if available (for Gemini), otherwise use default OpenAI
            if hasattr(self, 'openai_client') and self.openai_client:
                response = self.openai_client.chat.completions.create(**params)
            else:
                response = openai.chat.completions.create(**params)
            
            choice = response.choices[0]
            if choice.message.content is None:
                logging.warning(f"Dream generation resulted in empty content. Finish reason: {choice.finish_reason}")
                return ""
            
            return choice.message.content.strip()
        
        # Run the synchronous call in a thread pool to make it properly async
        return await asyncio.to_thread(_sync_call)
    
    async def _generate_anthropic(self, prompt: str, config: GenerationConfig, system_message: Optional[str] = None) -> str:
        """Generate using Anthropic Claude models."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        # Prepare messages with optional system message
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare parameters, only include max_tokens if specified
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "messages": messages
        }
        
        # Only add max_tokens if it's specified
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        
        response = await self.anthropic_client.messages.create(**params)
        return response.content[0].text.strip()

    async def _generate_openrouter(self, prompt: str, config: GenerationConfig, system_message: Optional[str] = None) -> str:
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
        
        # Prepare messages with optional system message
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare data, only include max_tokens if specified
        data = {
            "model": config.model,
            "messages": messages,
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
