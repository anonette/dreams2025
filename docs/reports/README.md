# Dreams Project - Cross-Linguistic Dream Research System

A comprehensive Python system for investigating whether Large Language Models (LLMs) produce dream narratives that reflect language-specific cultural contexts or generate universal themes.

## 🌟 Research Question

**Do LLMs produce dream narratives reflecting language-specific cultural contexts or universal themes?**

This system investigates whether AI-generated dreams vary culturally when prompted in different languages, or whether they converge on universal themes regardless of linguistic context.

## 🚀 Quick Start

### 1. Installation & Setup
```bash
# Clone and setup environment
git clone <repository-url>
cd Dreams
python -m venv dream_env
source dream_env/bin/activate  # On Windows: dream_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Set up your API keys as environment variables:
```bash
# OpenAI (required for GPT models)
export OPENAI_API_KEY="your-openai-key"

# Anthropic (for Claude models)  
export ANTHROPIC_API_KEY="your-anthropic-key"

# OpenRouter (for Mistral, DeepSeek, etc.)
export OPENROUTER_API_KEY="your-openrouter-key"
```

### 3. Run Your First Experiment
```bash
# Quick test with 3 dreams per language
python first_test.py

# Generate larger batch (100 dreams per language)
python batch_dream_generator.py --dreams-per-language 100

# Run cultural analysis on existing data
python cultural_dream_analyst_persona.py
```

## 🛠️ Core Features

### **Multi-Language Dream Generation**
- **Supported Languages**: English, Basque, Serbian, Hebrew, Slovenian
- **Consistent Prompts**: "Finish: Last night I dreamt of..." (translated appropriately)
- **Cultural Diversity**: Covers Western, Basque/Iberian, Slavic/Balkan, Jewish/Middle Eastern, and Central European contexts

### **Multi-Model LLM Support**
- **OpenAI Models**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Anthropic Models**: Claude-3.5-Sonnet, Claude-3-Haiku
- **OpenRouter Models**: Mistral, DeepSeek, and other models
- **Temperature Control**: Experiments with different creativity levels (0.3, 0.7, 0.9, 1.0)

### **Advanced Analysis Capabilities**

#### **1. Advanced Cultural Analysis** (`cultural_dream_analyst_persona.py`) ⭐ **Primary**
- **Hall–Van de Castle System**: Empirical dream content coding with 7 major categories
- **Gottschalk-Gleser Method**: Affective content scoring (anxiety, hostility, alienation)
- **Cultural Scripts Theory**: Cultural meaning-making patterns and worldview indicators
- **Cross-Cultural Comparison**: Statistical analysis across 5 languages
- **Output**: 9 comprehensive CSV/JSON files + interpretive reports

#### **2. Multilingual Analysis** (`Tests/analyze_multilingual_data.py`)
- **Cross-linguistic Comparison**: Dream length, content themes, success rates
- **Statistical Summaries**: Language-specific metrics and patterns  
- **Content Pattern Analysis**: Theme identification across languages
- **Prompt Entropy Analysis**: Marker usage and variation tracking
- **Visualization**: Interactive charts and progress tracking

#### **3. Statistical Analysis** (`statistical_analysis.py`)
- **Cross-linguistic multilevel modeling**
- **Mixed-effects logistic regression** 
- **Success rate and duration analysis**
- **Temporal pattern visualization**
- **Note**: Requires compatible scipy version

#### **4. Interactive Dashboard** (`streamlit_dream_analyzer.py`) ⭐ **New**
- **Web Interface**: Comprehensive analysis dashboard
- **Session Management**: Organized result storage and downloads
- **Real-time Analysis**: Run analyses through web interface
- **Data Visualization**: Interactive charts and tables
- **Download Center**: ZIP files for easy result sharing

#### **5. Progress Tracking** (`Tests/check_progress.py`)
- **Session Monitoring**: Track generation progress
- **Success Rate Analysis**: Quality metrics by language
- **Error Analysis**: Debugging and improvement insights
- **Data Verification**: Ensure complete dataset integrity

## ✅ Research-Ready System

### **🔬 Stateless LLM Configuration**
This system is **properly configured for rigorous research** with complete stateless operation:

**✅ Stateless API Calls:**
- Each GPT-4o query uses single-message arrays: `"messages": [{"role": "user", "content": prompt}]`
- No conversation history maintained between calls
- No system messages carrying session state
- Each API call is completely independent

**✅ Session Independence:**
- `"session_independence": true` in all API calls
- Unique `user_id` generated for each call with entropy suffix
- Fresh `prompt_id` for each request
- No memory between calls ensures unbiased results

**✅ Research Integrity:**
- Cultural analysis performs post-processing only (no additional LLM calls)
- Analyzes existing dream data from CSV files
- No contamination between language/cultural contexts

### **🎯 What This System Can Do**

**Multi-Language Research:**
- ✅ English, Basque, Hebrew, Serbian, Slovenian support
- ✅ Proper language configuration with script system tracking
- ✅ Cross-linguistic comparison capabilities

**Robust Data Collection:**
- ✅ Structured logging by language/model/session
- ✅ Comprehensive metadata tracking
- ✅ Error handling and recovery mechanisms

**Advanced Analysis:**
- ✅ Cultural dream analysis framework (Hall–Van de Castle + Gottschalk-Gleser)
- ✅ Cross-linguistic comparison tools
- ✅ Statistical significance testing
- ✅ Research reporting capabilities

**Research Applications Ready:**
- 🔬 **Cultural bias in AI models** - Compare dream themes across languages
- 🔬 **Universal vs. culture-specific patterns** - Identify cross-cultural commonalities
- 🔬 **Language effects on AI creativity** - Analyze linguistic influence on output
- 🔬 **Cross-cultural psychology** - Study AI-generated cultural patterns

### **🚀 Immediate Research Capability**

You can start research immediately:

```bash
# Generate stateless dreams across all languages
python batch_dream_generator.py --dreams-per-language 100

# Run comprehensive cultural analysis
python cultural_dream_analyst_persona.py

# Full automated research pipeline
python automate_dreams.py --mode once --dreams 100
```

**Research Validation:**
- ✅ Each API call contains only the current prompt
- ✅ No conversation history in messages array
- ✅ Session independence properly maintained
- ✅ Unique identifiers for each request prevent contamination

This ensures valid cultural and linguistic research with unbiased, independent dream generation across all languages and contexts.

## 📁 Project Structure

```
Dreams/
├── src/
│   ├── analysis/           # Analysis engines
│   ├── config/            # Language configurations
│   ├── models/            # LLM interfaces
│   └── visualization/     # Report generation
├── logs/                  # Dream generation logs
│   └── [language]/        # Language-specific folders
│       └── gpt-4o/        # Model-specific folders
│           └── session_*/  # Individual sessions
├── cultural_dream_analysis_output/  # Analysis results
├── research_reports/      # Research documentation
├── batch_dream_generator.py     # Main generation script
├── cultural_dream_analyst_persona.py  # Cultural analysis
├── first_test.py         # Quick test script
└── requirements.txt      # Dependencies
```

## 🔧 How to Run the System

### Generate Dreams

#### **Quick Test (3 dreams per language)**
```bash
python first_test.py
```

#### **Batch Generation (Customizable)**
```bash
# Generate 100 dreams per language
python batch_dream_generator.py --dreams-per-language 100

# Specific language only
python batch_dream_generator.py --language english --dreams-per-language 50

# Multiple languages
python batch_dream_generator.py --language english basque hebrew --dreams-per-language 25

# Different model and temperature
python batch_dream_generator.py --model claude-3.5-sonnet --temperature 0.7 --dreams-per-language 10
```

#### **Fast vs Research Mode**
```bash
# Fast generation (minutes to hours)
python batch_dream_generator.py --dreams-per-language 100

# Research mode with temporal dispersion (hours to days)
python batch_dream_generator.py --dreams-per-language 100 --enable-temporal-dispersion
```

### Analyze Results

#### **Option 1: Interactive Dashboard (Recommended)** ⭐
```bash
# Launch comprehensive web interface
streamlit run streamlit_dream_analyzer.py
# Access at: http://localhost:8501

# Features:
# - Real-time analysis execution
# - Session-based result management
# - Interactive visualizations
# - Easy downloads
```

#### **Option 2: Advanced Cultural Analysis**
```bash
# Run comprehensive cultural analysis (command line)
python cultural_dream_analyst_persona.py

# Output:
# - 9 detailed CSV/JSON analysis files
# - Cross-cultural comparison reports
# - Statistical summaries
# Results saved to: analysis_output/YYYYMMDD/cultural_dream_analysis_HHMMSS/
```

#### **Option 3: Multilingual Analysis**
```bash
# Cross-linguistic comparison
python Tests/analyze_multilingual_data.py

# Features:
# - Dream length analysis
# - Content theme comparison
# - Success rate metrics
# - Multilingual report generation
```

#### **Option 4: Progress & Statistics**
```bash
# Check generation progress
python Tests/check_progress.py

# Statistical analysis (requires compatible scipy)
python statistical_analysis.py --session-id [session_id]
```

#### **Option 5: Research Reports**
```bash
# Generate complete research report
python generate_research_report.py --study-id "DREAM_RESEARCH_2025"
```

### Monitor Progress

```bash
# Check current batch progress
python check_progress.py

# Analyze errors and failures
python analyze_errors.py

# Compare dream lengths across languages
python compare_dream_lengths.py
```

## 📊 Understanding the Output

### Generated Files
- **`logs/[language]/gpt-4o/session_*/dreams.csv`**: Raw dream data
- **`cultural_dream_analysis_output/`**: Cultural analysis results
- **`results/`**: Visualization and analysis outputs

### Analysis Insights
The system provides:
- **Cultural Patterns**: Language-specific thematic elements
- **Universal Themes**: Cross-cultural commonalities
- **Statistical Comparisons**: Significance testing across languages
- **Emotional Profiles**: Affect analysis by culture
- **Visual Reports**: Interactive dashboards and charts

## 🗃️ Comprehensive Logging & Data Management

### **📁 Hierarchical Log Structure**
```
logs/
├── [language]/                    # english, basque, hebrew, serbian, slovenian
│   └── gpt-4o/                   # Model-specific folder
│       └── session_YYYYMMDD_HHMMSS/  # Individual session
│           ├── api_calls.csv          # Complete API call metadata
│           ├── dreams.csv             # Generated dream content  
│           ├── session_data.json      # Complete session data
│           └── temporal_statistics.json  # Timing analysis
```

### **📊 Dream Data Logging**
Every dream generation is comprehensively logged with:

**✅ Complete Metadata:**
- Dream text in target language
- API call details (duration, temperature, model)
- Session independence tracking (`"session_independence": true`)
- Unique identifiers (`call_id`, `batch_id`, `user_id`, `prompt_id`)
- Language metadata (language code, script system)
- Temporal data (timestamps, delays, dispersion)
- Success/failure status with error tracking

**Sample Dream Log Entry:**
```json
{
  "call_id": "54998a2a-8138-4662-8883-b2d9f95ff518",
  "language": "serbian",
  "language_code": "sr", 
  "script": "Cyrillic",
  "dream": "Sinoć sam sanjao da sam se našao u čarobnoj šumi...",
  "status": "success",
  "timestamp": "2025-06-25T15:03:43.392773",
  "session_independence": true,
  "prompt_id": "9368d1f9"
}
```

### **📈 Analysis Output Management**
```
cultural_dream_analysis_output/
├── comprehensive_dream_analysis_YYYYMMDD_HHMMSS.csv      # Complete analysis data
├── cross_cultural_comparison_YYYYMMDD_HHMMSS.json       # Cross-language comparison
├── cultural_markers_analysis_YYYYMMDD_HHMMSS.csv        # Cultural pattern data
├── emotional_analysis_YYYYMMDD_HHMMSS.csv               # Affect analysis
├── settings_narrative_analysis_YYYYMMDD_HHMMSS.csv      # Setting/narrative data
├── detailed_analysis_YYYYMMDD_HHMMSS.json               # Full analysis JSON
└── cultural_interpretation_report_YYYYMMDD_HHMMSS.txt   # Research report
```

### **🔬 Analysis Data Captured**
- **Hall–Van de Castle analysis** (characters, settings, activities)
- **Gottschalk-Gleser affect scores** (anxiety, hostility, social alienation)  
- **Cultural markers** (collectivism, individualism, nature connection)
- **Symbolic elements** (light, water, tree, path, etc.)
- **Narrative structures** (journey, transformation, discovery)
- **Agency levels** (high, balanced, low agency)
- **Cross-linguistic comparisons** with statistical significance

### **🔧 Quality Control & Error Tracking**
- **Rejected dreams** with detailed reasons (`rejected_dreams.csv`, `.json`, `.jsonl`)
- **API errors** with comprehensive metadata
- **Session checkpoints** for recovery and resume capability
- **Temporal analysis** (call timing, dispersion patterns)
- **Success rate monitoring** by language and model

### **📋 Data Export Formats**
**For Statistical Analysis:**
- **CSV files** → R, SPSS, Python pandas
- **JSON files** → Programmatic analysis
- **Structured datasets** → Ready for multilevel modeling

**For Research Documentation:**
- **Markdown reports** → Academic papers
- **Summary statistics** → Quick insights
- **Cross-cultural comparisons** → Publication-ready tables

### **🎯 Research Benefits**
**Complete Reproducibility:**
- Every session has unique ID for exact replication
- Full API call history with parameters
- Cross-reference between raw data and analysis

**Quality Assurance:**
- Comprehensive error tracking
- Success rate monitoring
- Temporal pattern validation

**Multi-Format Exports:**
- Statistical software compatibility
- Research paper integration
- Programmatic analysis support

### **📊 Quick Data Verification**
```bash
# Check your current logs
ls logs/*/gpt-4o/session_*/

# View latest analysis results  
ls cultural_dream_analysis_output/

# Check dream count by language
wc -l logs/*/gpt-4o/session_*/dreams.csv

# Verify session independence
grep "session_independence.*true" logs/*/gpt-4o/session_*/session_data.json
```

This comprehensive logging system ensures full research integrity with complete traceability from raw dream generation through final cultural analysis.

## 📈 Research Applications

This system enables research into:
- **Cultural Bias in AI**: Do LLMs reflect cultural contexts?
- **Universal Themes**: Are certain dream themes universal?
- **Language Effects**: How does language affect AI creativity?
- **Cross-Cultural Psychology**: AI-generated cultural patterns
- **Linguistic Anthropology**: Language-culture relationships

## 🔍 Analysis Examples

### Cultural Analysis Output
```bash
# After running cultural_dream_analyst_persona.py
ENGLISH CULTURAL ANALYSIS:
- Strong nature connection: 58% prevalence
- High agency patterns: 89% of dreams
- Spiritual orientation: 18% prevalence

CROSS-CULTURAL PATTERNS:
- Universal symbols: light, tree, water
- Natural settings preferred across all cultures
- Dreams serve as psychological refuge
```

### Statistical Results
```bash
# Cross-cultural differences in agency patterns
F(4,468) = 23.7, p < 0.001, η² = 0.17 (large effect)

# English: 89% high agency
# Other languages: 62-78% balanced agency
```

## 📚 Dependencies

### Core Requirements
```bash
pip install openai anthropic httpx pandas plotly matplotlib seaborn
pip install scikit-learn nltk deep-translator numpy scipy streamlit
```

### Full Installation
```bash
pip install -r requirements.txt
```

## 🚀 Advanced Usage

### Custom Experiments
```python
# Modify first_test.py for custom configurations
models = ['gpt-4o', 'claude-3.5-sonnet']
temperatures = [0.3, 0.7, 0.9]
languages = ['english', 'basque', 'hebrew']
```

### Batch Processing
```python
from src.pipeline.dream_generator import DreamResearchPipeline

pipeline = DreamResearchPipeline(api_keys)
dreams = await pipeline.generate_dreams(dreams_per_config=5)
analysis = await pipeline.analyze_results(dreams)
```

## 🔗 Additional Resources

### Research Reports in This Study
- **📄 [Cultural Analysis Research Report](cultural_dream_analysis_research_report.md)** - Academic paper
- **📊 [Data Package](data_package/)** - Research datasets

### Project Documentation  
- **[Main README](../../README.md)** - Complete project documentation
- **[MODELS.md](../../MODELS.md)** - Model reference guide
- **[RESEARCH_REPORTING_README.md](../../RESEARCH_REPORTING_README.md)** - Research reporting system
- **[LOGS_ANALYSIS_README.md](../../LOGS_ANALYSIS_README.md)** - Analysis capabilities

---

**Research Question**: Do LLMs produce culturally-specific or universal dream narratives when prompted in different languages?

**How to Find Out**: 
1. `python batch_dream_generator.py --dreams-per-language 100`
2. `python cultural_dream_analyst_persona.py`
3. Check results in `cultural_dream_analysis_output/`
