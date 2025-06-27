# Structured Logging System

## Overview

The batch dream generator now implements a comprehensive structured logging system that organizes all data and logs in a hierarchical directory structure based on **language**, **model**, and **session**. This provides better organization, easier analysis, and cleaner data management for cross-linguistic research.

## Directory Structure

```
logs/
├── english/
│   ├── gpt-4o/
│   │   ├── session_20240101_120000/
│   │   │   ├── api_calls.csv
│   │   │   ├── dreams.csv
│   │   │   ├── session_data.json
│   │   │   ├── temporal_statistics.json
│   │   │   ├── errors.jsonl
│   │   │   ├── errors.csv
│   │   │   ├── errors.json
│   │   │   ├── rejected_dreams.jsonl
│   │   │   ├── rejected_dreams.csv
│   │   │   └── rejected_dreams.json
│   │   └── session_20240101_140000/
│   └── claude-3-sonnet/
│       └── session_20240101_160000/
├── basque/
│   └── gpt-4o/
│       └── session_20240101_120000/
├── serbian/
│   └── gpt-4o/
│       └── session_20240101_120000/
├── hebrew/
│   └── gpt-4o/
│       └── session_20240101_120000/
├── slovenian/
│   └── gpt-4o/
│       └── session_20240101_120000/
├── session_summary_20240101_120000.json
├── all_api_calls_20240101_120000.csv
├── all_dreams_20240101_120000.csv
├── batch_tracker_20240101_120000.json
├── temporal_statistics_20240101_120000.json
└── batch_generation_20240101_120000.log
```

## File Types and Content

### Language-Specific Files

Each language/model/session directory contains:

#### Core Data Files
- **`api_calls.csv`** - Complete API call data with entropy and temporal metadata
- **`dreams.csv`** - Dream content with linguistic and generation metadata  
- **`session_data.json`** - Comprehensive session data with all metadata

#### Error Tracking Files
- **`errors.jsonl`** - Real-time streaming error log
- **`errors.csv`** - Structured error data for analysis
- **`errors.json`** - Complete error metadata with session information
- **`rejected_dreams.jsonl`** - Real-time streaming rejected dreams log
- **`rejected_dreams.csv`** - Structured rejected dreams data
- **`rejected_dreams.json`** - Complete rejection metadata

#### Analysis Files
- **`temporal_statistics.json`** - Detailed temporal dispersion analysis
- **`entropy_statistics.json`** - Prompt entropy and marker usage analysis

### Global Session Files

Located in the root logs directory:

- **`session_summary_{session_id}.json`** - Cross-language session overview
- **`all_api_calls_{session_id}.csv`** - Flattened data from all languages
- **`all_dreams_{session_id}.csv`** - Combined dreams from all languages
- **`batch_tracker_{session_id}.json`** - Batch processing metadata
- **`temporal_statistics_{session_id}.json`** - Global temporal analysis
- **`batch_generation_{session_id}.log`** - Complete system log

## Data Structure Examples

### session_data.json
```json
{
  "metadata": {
    "language": "english",
    "language_code": "en",
    "script": "Latin",
    "model": "gpt-4o",
    "session_id": "20240101_120000",
    "timestamp": "20240101_120000",
    "successful_calls": 100,
    "failed_calls": 3,
    "total_calls": 103,
    "sampling_config": {
      "temperature": 1.0,
      "top_p": 1.0,
      "presence_penalty": 0.1,
      "frequency_penalty": 0.0,
      "batch_size": 50,
      "use_prompt_variants": true,
      "invisible_marker_probability": 0.3
    }
  },
  "temporal_statistics": {
    "mean_interval": 1847.3,
    "std_interval": 892.1,
    "total_span_hours": 8.2,
    "intervals": [...]
  },
  "entropy_statistics": {
    "total_calls": 103,
    "calls_with_markers": 31,
    "marker_usage_rate": 0.301,
    "marker_type_distribution": {
      "marker_0_prefix": 8,
      "marker_1_suffix": 7,
      "marker_2_middle": 6,
      "none": 72
    },
    "unique_prompt_ids": 103
  },
  "api_calls": [...],
  "dreams": [...]
}
```

### API Calls CSV Structure
```csv
call_id,batch_id,user_id,timestamp,language,language_code,script,model,temperature,top_p,presence_penalty,frequency_penalty,base_prompt,modified_prompt,prompt_id,marker_info,used_invisible_markers,dream_number,batch_size,dream,status,duration_seconds,temporal_delay_seconds,start_time,end_time,session_id,temporal_dispersion,session_independence
```

### Dreams CSV Structure
```csv
call_id,batch_id,user_id,language,language_code,script,model,temperature,dream,status,timestamp,session_id,dream_number,prompt_id,marker_info,temporal_delay
```

## Logging Workflow

### Automatic Setup
1. When `generate_dreams_for_language()` is called, structured logging is automatically set up
2. Directory hierarchy is created: `logs/{language}/{model}/session_{session_id}/`
3. Language-specific data structures are initialized
4. File paths are configured for the current language/model/session

### Data Collection
1. **API Calls**: Each call is logged with complete metadata including entropy and temporal data
2. **Dreams**: Content and generation metadata stored separately
3. **Errors**: Real-time logging to JSONL plus structured CSV/JSON export
4. **Rejections**: Filtered content tracked with rejection reasons

### Export Formats
1. **Real-time Streaming**: JSONL files for immediate monitoring
2. **Structured Analysis**: CSV files for statistical processing
3. **Complete Archives**: JSON files with full metadata
4. **Cross-language Aggregation**: Global files combining all languages

## Benefits for Research

### Organization
- **Language Isolation**: Each language's data is completely separate
- **Model Comparison**: Easy comparison across different models
- **Session Tracking**: Historical analysis of different research sessions
- **Batch Analysis**: Track temporal patterns and batch effects

### Data Integrity
- **No Data Mixing**: Impossible to accidentally combine different languages
- **Complete Metadata**: Every data point includes full generation context
- **Error Tracking**: Comprehensive error and rejection logging
- **Checkpoint Recovery**: Structured checkpoints preserve organization

### Analysis Workflows

#### Single Language Analysis
```bash
# Analyze English data from specific session
cd logs/english/gpt-4o/session_20240101_120000/
python analyze_dreams.py dreams.csv
python analyze_temporal.py temporal_statistics.json
```

#### Cross-Language Comparison
```bash
# Compare marker usage across languages
python compare_entropy.py \
  logs/english/gpt-4o/session_20240101_120000/session_data.json \
  logs/basque/gpt-4o/session_20240101_120000/session_data.json \
  logs/serbian/gpt-4o/session_20240101_120000/session_data.json
```

#### Model Comparison
```bash
# Compare same language across models
python compare_models.py \
  logs/english/gpt-4o/session_20240101_120000/ \
  logs/english/claude-3-sonnet/session_20240101_160000/
```

### Statistical Benefits
- **Clean Sampling**: No cross-contamination between languages or sessions
- **Temporal Tracking**: Complete temporal dispersion analysis per language
- **Entropy Analysis**: Marker effectiveness tracking per language family
- **Error Patterns**: Language-specific error analysis and filtering effectiveness

## Usage Examples

### Basic Usage
```bash
# Generate dreams for English with structured logging
python batch_dream_generator.py --language english --dreams-per-language 100

# This automatically creates:
# logs/english/gpt-4o/session_YYYYMMDD_HHMMSS/
```

### Multi-Language Research
```bash
# Generate for all languages (each gets separate directory)
python batch_dream_generator.py --dreams-per-language 100

# Creates structured hierarchy:
# logs/english/gpt-4o/session_YYYYMMDD_HHMMSS/
# logs/basque/gpt-4o/session_YYYYMMDD_HHMMSS/
# logs/serbian/gpt-4o/session_YYYYMMDD_HHMMSS/
# etc.
```

### Model Comparison Studies
```bash
# Generate with different models
python batch_dream_generator.py --language english --model gpt-4o
python batch_dream_generator.py --language english --model claude-3-sonnet

# Creates parallel structures:
# logs/english/gpt-4o/session_A/
# logs/english/claude-3-sonnet/session_B/
```

## Implementation Details

### Data Structures
- **Language-based dictionaries**: `api_calls_data[language]`, `dreams_data[language]`
- **Error tracking**: `error_data[language]`, `rejected_data[language]`
- **Automatic initialization**: Data structures created when language logging is set up

### File Management
- **Atomic operations**: Each language's data saved independently
- **Multiple formats**: Simultaneous JSONL (streaming), CSV (analysis), JSON (archives)
- **Path management**: Automatic file path configuration per language/model/session

### Checkpoint System
- **Structured recovery**: Checkpoints preserve language-based organization
- **Temporal state**: Temporal dispersion state maintained across resumptions
- **Progress tracking**: Language-specific progress tracking and resumption

## Migration from Flat Structure

The system automatically handles the transition from flat to structured logging:

1. **Backward Compatibility**: Can load old flat-structure checkpoints
2. **Gradual Migration**: New sessions use structured logging automatically
3. **Data Preservation**: No data loss during transition
4. **Format Consistency**: Same data fields, better organization

## Quality Assurance

### Data Validation
- **Language consistency**: All data in language directory matches language field
- **Session isolation**: No data bleeding between sessions
- **Model separation**: Clear model-specific organization
- **Complete metadata**: Every record includes full generation context

### Error Handling
- **Language-specific errors**: Errors tracked per language
- **Structured error data**: Complete error context preserved
- **Real-time monitoring**: JSONL files for immediate error detection
- **Analysis-ready formats**: CSV/JSON for statistical analysis

This structured logging system provides the foundation for rigorous cross-linguistic research with clean data separation, comprehensive metadata tracking, and flexible analysis workflows. 