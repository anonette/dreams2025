# Dreams Project - Logs & Analysis Guide

## 📋 Overview

This document provides a comprehensive guide to the log structure and analysis capabilities of the Dreams project - a cross-linguistic dream generation research platform using GPT-4o.

## 📁 Log Structure

Your logs are organized hierarchically to support cross-linguistic research and analysis:

```
logs/
├── batch_generation_YYYYMMDD_HHMMSS.log     # Real-time generation logs
├── checkpoint_YYYYMMDD_HHMMSS.json          # Progress snapshots
└── [language]/                              # Language-specific folders
    └── gpt-4o/                             # Model-specific folders
        └── session_YYYYMMDD_HHMMSS/        # Individual sessions
            ├── api_calls.csv               # Structured API call data
            ├── dreams.csv                  # Generated dream content
            ├── session_data.json           # Complete session metadata
            ├── temporal_statistics.json    # Time-based analysis
            ├── rejected_dreams.csv         # Filtered content
            ├── rejected_dreams.json        # Rejection details
            └── rejected_dreams.jsonl       # Line-delimited rejected dreams
```

**Supported Languages**: `english`, `serbian`, `hebrew`, `slovenian`, `basque`

## 🔧 Analysis Tools

### 1. Statistical Analysis (`statistical_analysis.py`)

**Purpose**: Cross-linguistic dream research with multilevel modeling and mixed-effects analysis

**Capabilities**:
- Descriptive statistics by language and script
- Success rate analysis across languages
- Duration analysis and temporal patterns
- Batch-level statistics and variation
- Mixed-effects logistic regression
- Temporal pattern analysis
- Visualization creation

**Usage**:
```bash
# Analyze specific session
python statistical_analysis.py --session-id 20250625_155722

# Specify custom logs directory
python statistical_analysis.py --session-id 20250625_155722 --logs-dir custom_logs/
```

**Output Files**:
- `statistical_results/descriptive_stats_[session]_[timestamp].json`
- `statistical_results/multilevel_analysis_[session]_[timestamp].json`
- `statistical_results/analysis_summary_[session]_[timestamp].txt`
- `statistical_results/[session]/success_rate_by_language.png`
- `statistical_results/[session]/duration_by_language.png`
- `statistical_results/[session]/temporal_patterns.png`
- `statistical_results/[session]/batch_success_distribution.png`

### 2. Error Analysis (`analyze_errors.py`)

**Purpose**: Debugging and quality improvement through error pattern analysis

**Capabilities**:
- API error types and frequencies
- Rejection reasons and patterns
- Language-specific error patterns
- Cross-session error aggregation
- CSV export for detailed analysis

**Usage**:
```bash
# Analyze specific session
python analyze_errors.py 20250625_155722

# Analyze all sessions
python analyze_errors.py --all

# Export errors to CSV
python analyze_errors.py --export
python analyze_errors.py --export 20250625_155722  # Specific session
```

**Output**:
- Console analysis reports
- `logs/error_analysis_[session]_errors.csv`
- `logs/error_analysis_[session]_rejections.csv`

### 3. Multilingual Data Analysis (`analyze_multilingual_data.py`)

**Purpose**: Comprehensive multilingual dream data analysis and comparison

**Capabilities**:
- Summary statistics across all languages
- Dream length analysis (character/word counts)
- Prompt entropy analysis and marker usage
- Content pattern analysis (themes, emotions)
- Cross-linguistic comparisons
- Visualization generation
- Report creation

**Usage**:
```bash
python analyze_multilingual_data.py
```

**Output Files**:
- `multilingual_analysis_report.md`
- `analysis_output/multilingual_analysis.png`
- `analysis_output/dream_length_comparison.png`
- `analysis_output/success_rate_comparison.png`
- Console summary tables

### 4. Dream Length Comparison (`compare_dream_lengths.py`)

**Purpose**: Cross-language dream length analysis with cultural insights

**Capabilities**:
- Average word/character counts by language
- Length variability and ranges
- Cross-linguistic length ratios
- Cultural/linguistic explanations
- Statistical comparisons

**Usage**:
```bash
python compare_dream_lengths.py
```

**Output**:
- Console comparison tables
- Key findings and insights
- Cultural explanations for patterns

### 5. Progress Checking (`check_progress.py`)

**Purpose**: Monitor generation session progress and status

**Capabilities**:
- Session completion status
- Success/failure rates by language
- Batch completion tracking
- Real-time progress monitoring
- Cross-session overview

**Usage**:
```bash
# Check all sessions
python check_progress.py

# Check specific session
python check_progress.py 20250625_155722
```

**Output**:
- Console progress reports
- Language-specific statistics
- Batch completion status

## 📊 Log File Types & Analysis Opportunities

### Batch Generation Logs (`.log` files)
**Content**: Real-time generation progress with timestamps, HTTP requests, success/failure status

**Sample Log Entry**:
```
2025-06-25 15:57:28,795 - INFO - Dream 1 for basque (Batch 673aebdf, Prompt ID: 5dff1677): success (6.492s, delay: 0.0s)
```

**Analysis Opportunities**:
- Temporal analysis of generation patterns
- API performance and response times
- Error tracking and debugging
- Batch completion monitoring

### API Calls CSV (`api_calls.csv`)
**Content**: Structured data on each API call with metadata

**Columns**: `timestamp`, `language`, `script`, `status`, `duration_seconds`, `batch_id`, `prompt_id`, `user_id`, `start_time`, `end_time`

**Analysis Opportunities**:
- Success rate analysis by various factors
- Duration analysis and performance optimization
- Temporal pattern identification
- Batch-level statistical analysis

### Dreams CSV (`dreams.csv`)
**Content**: Generated dream content with associated metadata

**Columns**: `timestamp`, `language`, `script`, `dream`, `prompt_id`, `batch_id`, `user_id`

**Analysis Opportunities**:
- Content analysis (themes, length, complexity)
- Cross-linguistic content comparison
- Quality assessment
- Linguistic pattern analysis

### Session Data JSON (`session_data.json`)
**Content**: Comprehensive session information including entropy statistics

**Key Sections**:
- `metadata`: Session configuration and summary
- `entropy_statistics`: Prompt variation and marker usage
- `temporal_statistics`: Time-based performance data

**Analysis Opportunities**:
- Entropy analysis and prompt variation
- Marker usage patterns
- Session-level performance metrics
- Configuration effectiveness analysis

### Temporal Statistics JSON (`temporal_statistics.json`)
**Content**: Time-based analysis data

**Analysis Opportunities**:
- Time series analysis
- Performance trends over time
- Optimal timing identification
- Batch scheduling optimization

### Rejected Dreams Files (`.csv`, `.json`, `.jsonl`)
**Content**: Dreams that were filtered out and reasons for rejection

**Analysis Opportunities**:
- Quality control analysis
- Filter effectiveness evaluation
- Content policy compliance
- Improvement identification

### Checkpoint Files (`checkpoint_*.json`)
**Content**: Progress snapshots for session recovery

**Analysis Opportunities**:
- Progress tracking across sessions
- Recovery point analysis
- Cross-session comparison
- Resource usage optimization

## 🎯 Recommended Analysis Workflows

### Quick Status Check
```bash
# Get overview of all sessions
python check_progress.py

# Quick length comparison across languages
python compare_dream_lengths.py
```

### Comprehensive Analysis
```bash
# Full multilingual analysis with visualizations
python analyze_multilingual_data.py

# Statistical analysis for latest session
python statistical_analysis.py --session-id [latest_session_id]
```

### Error Investigation
```bash
# Analyze errors across all sessions
python analyze_errors.py --all

# Export detailed error data for analysis
python analyze_errors.py --export
```

### Research-Grade Statistical Analysis
```bash
# Run complete statistical modeling
python statistical_analysis.py --session-id [session_id]

# Review generated reports and visualizations in statistical_results/
```

## 📈 Key Metrics Available

### Performance Metrics
- **Success Rate**: Percentage of successful dream generations per language
- **Duration Analysis**: API call timing and performance patterns
- **Batch Efficiency**: Completion rates and timing across batches
- **Error Patterns**: Types and frequencies of failures

### Content Metrics
- **Dream Length**: Character and word count distributions
- **Content Diversity**: Prompt entropy and variation
- **Theme Analysis**: Common dream elements and patterns
- **Language Comparison**: Cross-linguistic content differences

### Quality Metrics
- **Rejection Rates**: Content filtering effectiveness
- **Marker Usage**: Prompt variation success rates
- **Temporal Patterns**: Performance variation over time
- **Cross-Session Consistency**: Reliability across sessions

## 🔍 Advanced Analysis Examples

### Statistical Modeling
The `statistical_analysis.py` tool provides research-grade analysis including:
- **Mixed-effects logistic regression** for language effects
- **Multilevel modeling** with batch as random effects
- **Theme-based binary feature analysis**
- **Temporal pattern modeling**

### Content Analysis
Use `analyze_multilingual_data.py` for:
- **Dream theme detection** (flying, water, family, etc.)
- **Linguistic pattern analysis**
- **Cross-cultural content comparison**
- **Entropy-based diversity measurement**

### Performance Optimization
Combine tools for optimization insights:
```bash
# Identify optimal batch sizes and timing
python statistical_analysis.py --session-id [session]

# Analyze error patterns for improvement
python analyze_errors.py --all

# Check current performance status
python check_progress.py
```

## 📝 Output Directory Structure

After running analyses, you'll find results in:

```
project_root/
├── statistical_results/           # Statistical analysis outputs
│   ├── descriptive_stats_*.json
│   ├── multilevel_analysis_*.json
│   ├── analysis_summary_*.txt
│   └── [session_id]/             # Session-specific visualizations
├── analysis_output/              # Multilingual analysis outputs
│   ├── multilingual_analysis_report.md
│   └── *.png                     # Visualizations
└── logs/                         # Original logs and exported data
    ├── error_analysis_*.csv
    └── [original log structure]
```

## 🚀 Getting Started

1. **Check your current data**:
   ```bash
   python check_progress.py
   ```

2. **Run a quick analysis**:
   ```bash
   python compare_dream_lengths.py
   ```

3. **Generate comprehensive report**:
   ```bash
   python analyze_multilingual_data.py
   ```

4. **For research purposes**:
   ```bash
   python statistical_analysis.py --session-id [your_latest_session]
   ```

This logging and analysis system provides comprehensive insights into cross-linguistic dream generation performance, content patterns, and research opportunities.