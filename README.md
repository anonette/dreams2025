# Dreams Project - Cross-Linguistic Dream Research

A comprehensive research platform for generating and analyzing dreams across multiple languages using various AI models.

## 🚀 Quick Start

### For Dream Generation:
```bash
# Generate 1000 dreams with OpenRouter (single model)
python scripts/generation/generate_1000_dreams_openrouter_single.py

# Generate 1000 dreams with OpenRouter (multiple models)
python scripts/generation/generate_1000_dreams_openrouter_multimodel.py

# Generate with Gemini
python scripts/generation/generate_1000_dreams_gemini.py

# Generate with Mistral
python scripts/generation/generate_1000_dreams_mistral.py
```

### For Analysis:
```bash
# Run comprehensive cultural analysis
python scripts/analysis/comprehensive_cultural_analysis.py

# Run statistical analysis
python scripts/analysis/statistical_analysis.py
```

### For Interactive Apps:
```bash
# Launch Streamlit analyzer
python apps/streamlit_dream_analyzer_optimized.py

# Run dashboard
python apps/run_dashboard.py
```

## 📁 Project Structure

```
Dreams/
├── 📜 README.md                     # This file - project overview
├── 📋 requirements.txt              # Python dependencies
├── 🔧 .env                         # Environment variables
│
├── 🎯 scripts/                     # Main execution scripts
│   ├── 🏭 generation/              # Dream generation scripts
│   │   ├── batch_dream_generator.py           # Main batch generator ⭐
│   │   ├── generate_1000_dreams_openrouter_single.py    # OpenRouter (single) ⭐
│   │   ├── generate_1000_dreams_openrouter_multimodel.py # OpenRouter (multi) ⭐
│   │   ├── generate_1000_dreams_gemini.py     # Google Gemini ⭐
│   │   ├── generate_1000_dreams_mistral.py    # Mistral via OpenRouter ⭐
│   │   └── generate_optimized_dream_batch.py  # Optimized batch generation
│   │
│   ├── 📊 analysis/                # Analysis scripts
│   │   ├── comprehensive_cultural_analysis.py # Cultural analysis ⭐
│   │   ├── statistical_analysis.py            # Statistical analysis ⭐
│   │   ├── dream_thematic_analysis.py         # Thematic analysis
│   │   └── semantic_dream_analyzer.py         # Semantic analysis
│   │
│   └── 🔧 utilities/               # Utility scripts
│       ├── cleanup_empty_sessions.py          # Clean up empty sessions
│       ├── check_data_status.py               # Check data status
│       └── translation_manager.py             # Translation utilities
│
├── 🧪 tests/                       # Test files
│   ├── api_tests/                  # API testing
│   │   ├── test_gemini_api.py                 # Gemini API tests
│   │   ├── test_single_dream.py               # Single dream tests
│   │   └── run_gemini_validation.py           # Gemini validation
│   │
│   ├── validation_tests/           # Validation testing
│   │   ├── test_enhanced_parameters.py        # Parameter validation
│   │   └── test_optimized_config.py           # Config validation
│   │
│   └── translation_tests/          # Translation testing
│       ├── test_hebrew_translation.py         # Hebrew translation tests
│       └── test_translation_stats.py          # Translation statistics
│
├── ⚙️ config/                      # Configuration files
│   ├── languages/                  # Language configurations
│   │   ├── optimized_dream_languages.py      # Optimized language config ⭐
│   │   ├── immediate_dream_languages.py      # Immediate dream config
│   │   └── experimental_languages.py         # Experimental languages
│   │
│   └── models/                     # Model configurations
│
├── 🏗️ src/                         # Core library
│   ├── config/
│   │   └── languages.py                      # Main language config ⭐
│   ├── models/
│   │   └── llm_interface.py                  # LLM interface ⭐
│   ├── analysis/                   # Analysis modules
│   ├── reporting/                  # Reporting modules
│   └── visualization/              # Visualization modules
│
├── 📱 apps/                        # Interactive applications
│   ├── streamlit_dream_analyzer_optimized.py # Main Streamlit app ⭐
│   ├── streamlit_dream_analyzer.py           # Original Streamlit app
│   ├── run_dashboard.py                      # Dashboard app
│   └── semantic_demo.py                      # Semantic analysis demo
│
├── 📚 docs/                        # Documentation
│   ├── guides/                     # Usage guides
│   │   ├── AUTOMATION_README.md              # Automation guide
│   │   ├── GEMINI_GENERATION_GUIDE.md       # Gemini guide
│   │   └── OPTIMIZED_USAGE_GUIDE.md         # Optimization guide
│   │
│   ├── models/                     # Model documentation
│   │   └── MODELS.md                         # Available models
│   │
│   └── reports/                    # Generated reports
│
├── 💾 data/                        # Data directories
│   ├── logs/                       # All log directories
│   │   ├── logs_gemini_1_5_pro/              # Gemini 1.5 Pro logs
│   │   ├── logs_gemini_2_5_flash/            # Gemini 2.5 Flash logs
│   │   ├── logs_mistral/                     # Mistral logs
│   │   └── openrouter/                       # OpenRouter logs
│   │
│   ├── analysis_output/            # Analysis results
│   ├── research_reports/           # Research reports
│   └── translations/               # Translation data
│
└── 🤖 automation/                  # Automation scripts
    ├── automate_dreams.py                    # Main automation script
    ├── run_scaled_generation.py             # Scaled generation
    └── batch_files/                # Batch automation files
        ├── run_900_batches.bat               # Windows batch files
        └── check_api_key.bat                 # API key checker
```

## 🌍 Supported Languages

- **English** (`en`) - Latin script
- **Basque** (`eu`) - Latin script  
- **Serbian** (`sr`) - Cyrillic script
- **Hebrew** (`he`) - Hebrew script
- **Slovenian** (`sl`) - Latin script

## 🤖 Supported Models

### OpenRouter Models
- `anthropic/claude-3.5-sonnet` - Premium reasoning
- `openai/gpt-4o` - OpenAI's flagship model
- `openai/gpt-4o-mini` - Fast and efficient
- `mistralai/mistral-nemo` - Mistral's latest
- `google/gemini-pro-1.5` - Google's advanced model
- `meta-llama/llama-3.1-70b-instruct` - Meta's large model
- `qwen/qwen-2.5-72b-instruct` - Alibaba's model
- `deepseek/deepseek-chat` - DeepSeek's conversational model

### Direct API Models
- **Google Gemini** (via direct API)
- **Mistral** (via OpenRouter)

## 🔧 Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Copy and edit the environment file
   cp .env.example .env
   
   # Add your API keys:
   OPENROUTER_API_KEY=your-openrouter-key
   GEMINI_API_KEY=your-gemini-key
   ANTHROPIC_API_KEY=your-anthropic-key
   ```

3. **Run your first generation:**
   ```bash
   python scripts/generation/generate_1000_dreams_openrouter_single.py
   ```

## 📊 Key Features

### ✅ Dream Generation
- **Batch generation** with configurable parameters
- **Multiple model support** via OpenRouter
- **Resume capability** for interrupted sessions
- **Structured logging** with comprehensive metadata
- **Statistical sampling** with entropy controls

### ✅ Analysis Capabilities
- **Cultural analysis** across languages and scripts
- **Statistical analysis** with multilevel modeling
- **Thematic analysis** of dream content
- **Semantic analysis** using NLP techniques
- **Cross-linguistic comparison**

### ✅ Interactive Tools
- **Streamlit dashboard** for real-time analysis
- **Semantic demo** for exploring dream semantics
- **Data visualization** and reporting tools

### ✅ Research Features
- **Rigorous statistical sampling**
- **Temporal clustering controls**
- **Prompt entropy measures**
- **Session independence**
- **Comprehensive metadata tracking**

## 🚀 Common Workflows

### Generate Dreams with OpenRouter
```bash
# Single model (recommended for beginners)
python scripts/generation/generate_1000_dreams_openrouter_single.py

# Multi-model comparison
python scripts/generation/generate_1000_dreams_openrouter_multimodel.py
```

### Analyze Existing Dreams
```bash
# Cultural analysis
python scripts/analysis/comprehensive_cultural_analysis.py

# Statistical analysis
python scripts/analysis/statistical_analysis.py

# Launch interactive analyzer
python apps/streamlit_dream_analyzer_optimized.py
```

### Check Data Status
```bash
# Check current data status
python scripts/utilities/check_data_status.py

# Clean up empty sessions
python scripts/utilities/cleanup_empty_sessions.py
```

## 📈 Research Applications

This platform supports various research applications:

- **Cross-linguistic dream analysis**
- **Cultural dream pattern studies**
- **AI model comparison for creative tasks**
- **Multilingual NLP research**
- **Psycholinguistic studies**
- **Computational creativity research**

## 🔍 Data Organization

All generated data is organized in the `data/` directory:

- **`logs/`** - Raw generation logs by model
- **`analysis_output/`** - Analysis results and reports
- **`research_reports/`** - Formatted research reports
- **`translations/`** - Translation data and statistics

## 🆘 Troubleshooting

### Common Issues:

1. **Import errors after reorganization:**
   - Update Python path: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
   - Or run from project root: `python -m scripts.generation.script_name`

2. **API key issues:**
   - Check `.env` file exists and has correct keys
   - Verify API key permissions and quotas

3. **Missing dependencies:**
   - Run: `pip install -r requirements.txt`

### Getting Help:

- Check the relevant README in `docs/guides/`
- Review the `PROJECT_REORGANIZATION_PLAN.md` for structure details
- Examine log files in `data/logs/` for detailed error information

## 📝 Contributing

When adding new scripts:

1. **Generation scripts** → `scripts/generation/`
2. **Analysis scripts** → `scripts/analysis/`
3. **Utility scripts** → `scripts/utilities/`
4. **Test scripts** → `tests/` (in appropriate subdirectory)
5. **Interactive apps** → `apps/`
6. **Documentation** → `docs/guides/`

## 📄 License

This project is for research purposes. Please cite appropriately if used in academic work.

---

**🎯 Ready to generate dreams? Start with:**
```bash
python scripts/generation/generate_1000_dreams_openrouter_single.py