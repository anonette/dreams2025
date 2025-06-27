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

#### **1. Cultural Dream Analysis** (`cultural_dream_analyst_persona.py`)
- **Hall–Van de Castle System**: Empirical dream content coding
- **Gottschalk-Gleser Method**: Affective content scoring  
- **Cultural Scripts Theory**: Cultural meaning-making patterns
- **Cross-Cultural Comparison**: Statistical analysis across languages

#### **2. Semantic Clustering** (`semantic_dream_clustering.py`)
- **TF-IDF Vectorization**: Converts dreams to numerical vectors
- **K-Means Clustering**: Groups dreams by semantic similarity
- **t-SNE Visualization**: 2D plots showing cluster distributions
- **Language-Agnostic**: Works across languages without translation

#### **3. Statistical Analysis** (`statistical_analysis.py`)
- **Cross-linguistic multilevel modeling**
- **Mixed-effects logistic regression**
- **Success rate and duration analysis**
- **Temporal pattern visualization**

#### **4. Co-occurrence Analysis** (`dream_cooccurrence_analysis.py`)
- **Word Co-occurrence Matrices**: Analyzes relationships between words
- **Statistical Significance**: Identifies meaningful word combinations
- **Cross-Linguistic Patterns**: Compares semantic relationships across languages
- **Network Analysis**: Builds semantic networks from co-occurrence data

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

#### **Cultural Analysis (Recommended)**
```bash
# Run comprehensive cultural analysis
python cultural_dream_analyst_persona.py

# Results saved to: cultural_dream_analysis_output/
```

#### **Semantic Clustering**
```bash
# TF-IDF + K-Means clustering
python semantic_dream_clustering.py

# Interactive demo
python demo_clustering.py
```

#### **Statistical Analysis**
```bash
# Research-grade statistical analysis
python statistical_analysis.py --session-id [session_id]

# Multilingual comparison
python analyze_multilingual_data.py
```

#### **Co-occurrence Analysis**
```bash
# Word relationship analysis
python dream_cooccurrence_analysis.py
```

#### **Interactive Dashboard**
```bash
# Launch web interface
streamlit run streamlit_dream_analyzer.py
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
