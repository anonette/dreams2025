# Available Models for Dream Research

This document lists all the models you can use with the dream research system, organized by provider.

## Current Configuration

The system is currently configured to use these models:
- `gpt-4`
- `gpt-3.5-turbo` 
- `claude-3-sonnet-20240229`

## OpenAI Models

### GPT-4 Series
- `gpt-4o` - Latest and most capable model, fastest GPT-4 variant
- `gpt-4o-mini` - Faster and more cost-effective than GPT-4o
- `gpt-4` - Most capable model, best for complex reasoning
- `gpt-4-turbo` - Faster and more cost-effective than GPT-4
- `gpt-4-turbo-preview` - Latest preview version
- `gpt-4-1106-preview` - GPT-4 Turbo with vision
- `gpt-4-vision-preview` - GPT-4 with image understanding

### GPT-3.5 Series
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-3.5-turbo-16k` - Same as turbo but with 16k context
- `gpt-3.5-turbo-1106` - Latest GPT-3.5 Turbo version

**API Key Required**: `OPENAI_API_KEY`

## Anthropic Models

### Claude 3 Series
- `claude-3-opus-20240229` - Most capable Claude model
- `claude-3-sonnet-20240229` - Balanced performance and speed
- `claude-3-haiku-20240307` - Fastest and most cost-effective

### Claude 2 Series
- `claude-2.1` - Previous generation Claude
- `claude-2.0` - Earlier version

**API Key Required**: `ANTHROPIC_API_KEY`

## OpenRouter Models

OpenRouter provides access to many different model providers through a single API.

### Mistral Models
- `mistral-7b-instruct` - 7B parameter model
- `mistral-7b-instruct-v0.2` - Updated version
- `mistral-large` - Larger, more capable model
- `mistral-medium` - Balanced performance

### DeepSeek Models
- `deepseek-chat` - General purpose chat model
- `deepseek-coder` - Specialized for coding
- `deepseek-coder-33b-instruct` - 33B parameter coding model

### Other Popular Models via OpenRouter
- `meta-llama/llama-2-70b-chat` - Meta's Llama 2
- `google/gemini-pro` - Google's Gemini Pro
- `meta-llama/llama-2-13b-chat` - Smaller Llama 2
- `anthropic/claude-3-opus` - Claude 3 Opus via OpenRouter
- `anthropic/claude-3-sonnet` - Claude 3 Sonnet via OpenRouter
- `openai/gpt-4` - GPT-4 via OpenRouter
- `openai/gpt-3.5-turbo` - GPT-3.5 via OpenRouter

**API Key Required**: `OPENROUTER_API_KEY`

## How to Use Different Models

### 1. Update the Pipeline Configuration

Edit `src/pipeline/dream_generator.py` and modify the `models` list:

```python
self.models = [
    'gpt-4o',                    # Latest OpenAI model
    'gpt-4o-mini',               # Cost-effective GPT-4o
    'claude-3-sonnet-20240229',
    'mistral-7b-instruct',       # OpenRouter model
    'deepseek-chat',             # OpenRouter model
    'meta-llama/llama-2-70b-chat'  # OpenRouter model
]
```

### 2. Set Up API Keys

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENROUTER_API_KEY=your-openrouter-key
```

### 3. Run with Specific Models

You can also specify models via command line by modifying the main script:

```bash
python main.py --models gpt-4o,claude-3-sonnet-20240229,mistral-7b-instruct
```

## Model Selection Recommendations

### For Research Purposes
- **Best Quality**: `gpt-4o`, `gpt-4`, `claude-3-opus-20240229`
- **Good Balance**: `gpt-4o-mini`, `gpt-4-turbo`, `claude-3-sonnet-20240229`
- **Cost-Effective**: `gpt-3.5-turbo`, `claude-3-haiku-20240307`

### For Cross-Linguistic Analysis
- **Multilingual**: `gpt-4o`, `gpt-4`, `claude-3-sonnet-20240229`, `mistral-7b-instruct`
- **Cultural Sensitivity**: `gpt-4o`, `claude-3-opus-20240229`, `gpt-4`

### For Cost Optimization
- **Free Tier**: Use OpenRouter's free tier with smaller models
- **Budget**: `gpt-4o-mini`, `gpt-3.5-turbo`, `claude-3-haiku-20240307`
- **Performance**: `gpt-4o`, `mistral-7b-instruct`, `deepseek-chat`

## Pricing Considerations

### OpenAI
- GPT-4o: ~$0.005 per 1K tokens (input), ~$0.015 per 1K tokens (output)
- GPT-4o-mini: ~$0.00015 per 1K tokens (input), ~$0.0006 per 1K tokens (output)
- GPT-4: ~$0.03 per 1K tokens
- GPT-4 Turbo: ~$0.01 per 1K tokens
- GPT-3.5 Turbo: ~$0.002 per 1K tokens

### Anthropic
- Claude 3 Opus: ~$0.015 per 1K tokens
- Claude 3 Sonnet: ~$0.003 per 1K tokens
- Claude 3 Haiku: ~$0.00025 per 1K tokens

### OpenRouter
- Varies by model, generally competitive pricing
- Some models available on free tier
- Check [OpenRouter pricing](https://openrouter.ai/pricing) for current rates

## Testing Models

To test if a model works with your setup:

```python
from src.models.llm_interface import LLMInterface, GenerationConfig

# Initialize with your API keys
api_keys = {
    'openai': 'your-key',
    'anthropic': 'your-key', 
    'openrouter': 'your-key'
}

interface = LLMInterface(api_keys)

# Test a model
config = GenerationConfig(model='gpt-4o', temperature=0.7)
result = await interface.generate_dream("Finish: Last night I dream of...", config)
print(result)
```

## Model Capabilities Comparison

| Model | Multilingual | Creativity | Reasoning | Cost | Speed |
|-------|-------------|------------|-----------|------|-------|
| GPT-4o | Excellent | High | Excellent | Low | Very Fast |
| GPT-4o-mini | Excellent | High | Good | Very Low | Very Fast |
| GPT-4 | Excellent | High | Excellent | High | Medium |
| GPT-4 Turbo | Excellent | High | Excellent | Medium | Fast |
| GPT-3.5 Turbo | Good | Medium | Good | Low | Very Fast |
| Claude 3 Opus | Excellent | High | Excellent | High | Medium |
| Claude 3 Sonnet | Excellent | High | Good | Medium | Fast |
| Claude 3 Haiku | Good | Medium | Good | Low | Very Fast |
| Mistral 7B | Good | Medium | Good | Low | Fast |
| DeepSeek Chat | Good | Medium | Good | Low | Fast | 