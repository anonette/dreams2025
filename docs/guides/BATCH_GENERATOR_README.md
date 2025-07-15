# Batch Dream Generator

This module provides a comprehensive batch dream generation system for cross-linguistic research. It generates 100 dreams per language with detailed API call logging in both JSON and CSV formats.

## Features

- **100 dreams per language**: Generates exactly 100 dreams for each configured language
- **Detailed logging**: Comprehensive logging of every API call with metadata
- **Multiple output formats**: JSON and CSV logs with different levels of detail
- **Error handling**: Robust error handling and filtering of invalid responses
- **Rate limiting**: Built-in rate limiting to avoid API throttling
- **Session tracking**: Unique session IDs for tracking different generation runs

## Usage

### Basic Usage

```bash
# Generate 100 dreams for all languages
python batch_dream_generator.py

# Generate 100 dreams for a specific language
python batch_dream_generator.py --language english

# Generate with custom parameters
python batch_dream_generator.py --language basque --model gpt-4o --temperature 0.8 --dreams-per-language 50
```

### Command Line Arguments

- `--language`: Specific language to generate dreams for (optional, generates for all if not specified)
- `--model`: LLM model to use (default: gpt-4o)
- `--temperature`: Temperature for generation (default: 0.9)
- `--dreams-per-language`: Number of dreams per language (default: 100)
- `--api-keys`: Path to API keys file (optional, uses environment variables if not provided)

### Available Languages

- `english`: English (Latin script)
- `basque`: Basque (Latin script)
- `serbian`: Serbian (Cyrillic script)
- `hebrew`: Hebrew (Hebrew script)
- `slovenian`: Slovenian (Latin script)

## API Keys Setup

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

Or create a JSON file with your API keys:

```json
{
    "openai": "your-openai-key",
    "anthropic": "your-anthropic-key",
    "openrouter": "your-openrouter-key"
}
```

## Output Files

The generator creates several types of log files in the `logs/` directory:

### Per-Language Files
- `{language}_{model}_{timestamp}.json`: Complete data for one language
- `{language}_{model}_{timestamp}_api_calls.csv`: API call details
- `{language}_{model}_{timestamp}_dreams.csv`: Dream content only

### Session Files
- `session_summary_{session_id}.json`: Overall session statistics
- `all_api_calls_{session_id}.csv`: All API calls across languages
- `all_dreams_{session_id}.csv`: All dreams across languages
- `batch_generation_{session_id}.log`: Console log output

## Log Data Structure

### API Call Log (JSON/CSV)
Each API call is logged with the following fields:

```json
{
    "call_id": "unique-uuid",
    "timestamp": "2024-01-01T12:00:00",
    "language": "english",
    "language_code": "en",
    "script": "Latin",
    "model": "gpt-4o",
    "temperature": 0.9,
    "prompt": "Finish: Last night I dreamt of…",
    "dream_number": 1,
    "total_dreams": 100,
    "dream": "the actual dream content...",
    "status": "success",
    "duration_seconds": 2.345,
    "start_time": "2024-01-01T12:00:00",
    "end_time": "2024-01-01T12:00:02",
    "session_id": "20240101_120000",
    "max_tokens": 300,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

### Dream Log (JSON/CSV)
Simplified dream data:

```json
{
    "call_id": "unique-uuid",
    "language": "english",
    "language_code": "en",
    "script": "Latin",
    "model": "gpt-4o",
    "temperature": 0.9,
    "dream": "the actual dream content...",
    "status": "success",
    "timestamp": "2024-01-01T12:00:00",
    "session_id": "20240101_120000"
}
```

## Testing

Run the test script to verify everything works:

```bash
python test_batch_generator.py
```

This will generate 3 test dreams for English to verify the system is working correctly.

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Network issues, rate limits, authentication errors
- **Invalid Responses**: Filters out error messages, apologies, or incomplete responses
- **Rate Limiting**: Built-in delays between API calls
- **Logging**: All errors are logged with full context

## Performance

- **Rate Limiting**: 0.5 second delay between API calls
- **Batch Processing**: Processes one language at a time
- **Memory Efficient**: Streams data to files rather than keeping everything in memory
- **Resumable**: Each language is processed independently

## Integration

The batch generator can be integrated with the existing analysis pipeline:

```python
from batch_dream_generator import BatchDreamGenerator

# Generate dreams
generator = BatchDreamGenerator(api_keys)
results = await generator.generate_all_languages(100)

# Use with existing analysis
from src.pipeline.dream_generator import DreamResearchPipeline
pipeline = DreamResearchPipeline(api_keys)
analysis = pipeline.analyze_results(results) 