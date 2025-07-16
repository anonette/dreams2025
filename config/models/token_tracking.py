"""
Token tracking configuration for LLM providers
Tracks token usage, costs, and provides monitoring capabilities
"""

from dataclasses import dataclass
from typing import Dict, Optional
import tiktoken

@dataclass
class TokenPricing:
    """Token pricing per provider and model"""
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    currency: str = "USD"

# Token pricing as of 2025 (approximate)
TOKEN_PRICING = {
    # OpenAI models via OpenRouter
    "openai/gpt-4o": TokenPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015),
    "openai/gpt-4o-mini": TokenPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    
    # Anthropic models via OpenRouter
    "anthropic/claude-3.5-sonnet": TokenPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    
    # Google models via OpenRouter
    "google/gemini-pro-1.5": TokenPricing(input_cost_per_1k=0.00125, output_cost_per_1k=0.005),
    
    # Mistral models via OpenRouter
    "mistralai/mistral-nemo": TokenPricing(input_cost_per_1k=0.0003, output_cost_per_1k=0.0003),
    
    # Meta models via OpenRouter
    "meta-llama/llama-3.1-70b-instruct": TokenPricing(input_cost_per_1k=0.0009, output_cost_per_1k=0.0009),
    
    # Qwen models via OpenRouter
    "qwen/qwen-2.5-72b-instruct": TokenPricing(input_cost_per_1k=0.0009, output_cost_per_1k=0.0009),
    
    # DeepSeek models via OpenRouter
    "deepseek/deepseek-chat": TokenPricing(input_cost_per_1k=0.00014, output_cost_per_1k=0.00028),
}

class TokenCounter:
    """Token counting utility for different models"""
    
    def __init__(self):
        # Initialize tokenizers for different model families
        try:
            self.gpt_tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            # Fallback to cl100k_base encoding
            self.gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens for given text and model"""
        try:
            if "gpt" in model.lower() or "openai" in model.lower():
                return len(self.gpt_tokenizer.encode(text))
            elif "claude" in model.lower() or "anthropic" in model.lower():
                # Claude uses similar tokenization to GPT
                return len(self.gpt_tokenizer.encode(text))
            else:
                # For other models, use GPT tokenizer as approximation
                return len(self.gpt_tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for given token usage"""
        pricing = TOKEN_PRICING.get(model)
        if not pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        return input_cost + output_cost

@dataclass
class TokenUsageStats:
    """Token usage statistics for monitoring"""
    model: str
    language: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    estimated_cost: float = 0.0
    
    def add_usage(self, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Add token usage to stats"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        self.estimated_cost += cost
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def avg_tokens_per_request(self) -> float:
        return self.total_tokens / self.total_requests if self.total_requests > 0 else 0

class TokenTracker:
    """Global token usage tracker"""
    
    def __init__(self):
        self.counter = TokenCounter()
        self.usage_stats: Dict[str, Dict[str, TokenUsageStats]] = {}
    
    def track_usage(self, model: str, language: str, prompt: str, response: str, system_message: str = ""):
        """Track token usage for a request"""
        # Initialize if needed
        if model not in self.usage_stats:
            self.usage_stats[model] = {}
        if language not in self.usage_stats[model]:
            self.usage_stats[model][language] = TokenUsageStats(model=model, language=language)
        
        # Count tokens
        input_text = f"{system_message}\n{prompt}" if system_message else prompt
        input_tokens = self.counter.count_tokens(input_text, model)
        output_tokens = self.counter.count_tokens(response, model)
        
        # Estimate cost
        cost = self.counter.estimate_cost(input_tokens, output_tokens, model)
        
        # Update stats
        self.usage_stats[model][language].add_usage(input_tokens, output_tokens, cost)
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'estimated_cost': cost
        }
    
    def get_model_stats(self, model: str) -> Dict[str, TokenUsageStats]:
        """Get token stats for a specific model"""
        return self.usage_stats.get(model, {})
    
    def get_total_stats(self) -> Dict:
        """Get total token usage across all models and languages"""
        total_input = 0
        total_output = 0
        total_cost = 0.0
        total_requests = 0
        
        for model_stats in self.usage_stats.values():
            for lang_stats in model_stats.values():
                total_input += lang_stats.total_input_tokens
                total_output += lang_stats.total_output_tokens
                total_cost += lang_stats.estimated_cost
                total_requests += lang_stats.total_requests
        
        return {
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_input + total_output,
            'total_requests': total_requests,
            'estimated_total_cost': total_cost,
            'avg_tokens_per_request': (total_input + total_output) / total_requests if total_requests > 0 else 0
        }
    
    def format_stats_summary(self) -> str:
        """Format token usage summary for display"""
        total_stats = self.get_total_stats()
        
        summary = f"""
🔢 TOKEN USAGE SUMMARY
{'='*50}
📊 Total Requests: {total_stats['total_requests']:,}
📥 Input Tokens: {total_stats['total_input_tokens']:,}
📤 Output Tokens: {total_stats['total_output_tokens']:,}
🔢 Total Tokens: {total_stats['total_tokens']:,}
💰 Estimated Cost: ${total_stats['estimated_total_cost']:.4f}
📈 Avg Tokens/Request: {total_stats['avg_tokens_per_request']:.1f}

📋 BY MODEL:
"""
        
        for model, model_stats in self.usage_stats.items():
            model_total_tokens = sum(stats.total_tokens for stats in model_stats.values())
            model_total_cost = sum(stats.estimated_cost for stats in model_stats.values())
            model_short = model.split('/')[-1]
            
            summary += f"\n🤖 {model_short}:\n"
            summary += f"   🔢 Tokens: {model_total_tokens:,}\n"
            summary += f"   💰 Cost: ${model_total_cost:.4f}\n"
            
            for language, stats in model_stats.items():
                summary += f"   📝 {language}: {stats.total_tokens:,} tokens (${stats.estimated_cost:.4f})\n"
        
        return summary
