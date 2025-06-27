# Cross-Linguistic Dream Research System

A comprehensive Python system for investigating whether Large Language Models (LLMs) produce dream narratives that reflect language-specific cultural contexts or generate universal themes. This research system uses consistent prompts translated into multiple languages to generate dreams, then employs LLM-based analysis to identify cultural patterns and thematic differences.

## 🌟 Research Question

**Do LLMs produce dream narratives reflecting language-specific cultural contexts or universal themes?**

This system investigates whether AI-generated dreams vary culturally when prompted in different languages, or whether they converge on universal themes regardless of linguistic context.

## 🚀 Key Features

### **Multi-Language Dream Generation**
- **Supported Languages**: English, Basque, Serbian, Hebrew, Slovenian
- **Consistent Prompts**: "Finish: Last night I dreamt of..." (translated appropriately)
- **Cultural Diversity**: Covers Western, Basque/Iberian, Slavic/Balkan, Jewish/Middle Eastern, and Central European contexts

### **Multi-Model LLM Support**
- **OpenAI Models**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Anthropic Models**: Claude-3.5-Sonnet, Claude-3-Haiku
- **OpenRouter Models**: Mistral, DeepSeek, and other models
- **Temperature Control**: Experiments with different creativity levels (0.3, 0.7, 0.9, 1.0)

### **Advanced Analysis Techniques**

#### **1. Semantic Clustering (TF-IDF + K-Means)**
- **Algorithm**: TF-IDF vectorization followed by K-Means clustering
- **Process**: 
  1. **Text Preprocessing**: Cleans and tokenizes dream text
  2. **TF-IDF Vectorization**: Converts dreams to numerical vectors based on word importance
  3. **K-Means Clustering**: Groups dreams by semantic similarity
  4. **Cluster Analysis**: Identifies representative themes for each cluster
- **Output**: Discovers thematic groups like "Nature Dreams", "Urban Adventures", "Family Interactions"
- **Cross-Linguistic**: Works across languages without translation, revealing cultural patterns

#### **2. LLM-Based Theme Extraction**
- **Automatic identification** of emotional, symbolic, and cultural themes
- **Structured analysis** with confidence scores and cultural context
- **Thematic clustering** groups dreams into meaningful categories
- **Emotional profiling** rates intensity (fear, joy, wonder, etc.) on 1-10 scale

#### **3. Co-occurrence Analysis**
- **Word co-occurrence matrices** reveal thematic relationships
- **Statistical pattern analysis** identifies frequent word combinations
- **Cross-linguistic comparison** of semantic relationships
- **Cultural linguistic differences** highlighted through collocation patterns

#### **4. Cultural Pattern Analysis**
- **Language-specific theme identification** and frequency analysis
- **Cross-linguistic comparison** of cultural elements and narrative patterns
- **Statistical significance testing** for cultural differences
- **Temporal pattern analysis** of dream generation characteristics

### **Comprehensive Logging & Visualization**
- **Detailed Logs**: Complete dream generation and analysis logs with timestamps
- **Interactive Dashboards**: HTML visualizations for theme comparisons and cultural patterns
- **JSON Results**: Structured data for further analysis
- **Summary Reports**: Human-readable analysis summaries

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dream         │    │   LLM-Based     │    │   Visualization │
│   Generation    │───▶│   Analysis      │───▶│   & Reporting   │
│   Pipeline      │    │   Engine        │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Multi-Model   │    │   Theme         │    │   Interactive   │
│   LLM Interface │    │   Extraction    │    │   Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Dreams/
├── src/
│   ├── analysis/           # LLM-based dream analysis
│   │   ├── llm_dream_analyzer.py      # Main analysis engine
│   │   ├── cooccurrence_analyzer.py   # Word pattern analysis
│   │   └── simple_dream_analyzer.py   # Basic analysis (legacy)
│   ├── config/
│   │   └── languages.py    # Language configurations and prompts
│   ├── models/
│   │   └── llm_interface.py # Multi-model LLM interface
│   ├── pipeline/
│   │   └── dream_generator.py # Main research pipeline
│   └── visualization/
│       └── report_generator.py # Dashboard and chart generation
├── logs/                   # Detailed generation and analysis logs
├── results/               # Analysis results and visualizations
├── first_test.py          # Quick test script
├── requirements.txt       # Python dependencies
└── MODELS.md             # Complete model reference
```

## 🛠️ Installation & Setup

### 1. Clone and Setup Environment
```bash
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

### 3. Verify Setup
```bash
python check_keys.py
python test_setup.py
```

## 🚀 Quick Start

### Run Your First Experiment
```bash
python first_test.py
```

This will:
- Generate 3 dreams per language using GPT-4o at temperature 0.9
- Perform LLM-based thematic analysis
- Create interactive visualizations
- Save results to `results/first_test/`

### Expected Output
```
🌙 Dream Research - First Test Experiment
==================================================
Model: GPT-4o
Temperature: 0.9
Analysis: LLM-based theme identification
Languages: English, Basque, Serbian, Hebrew, Slovenian
==================================================
✅ OpenAI API key found

🚀 Generating 3 dreams per language...
✅ Dream generation completed!

🧠 Analyzing results with LLM...
📈 Creating visualizations...

🎉 First test completed successfully!
📁 Results saved to: results/first_test/
```

## 📊 Understanding the Results

### Generated Files
- **`generated_dreams_*.json`**: Complete dream narratives by language
- **`analysis_results_*.json`**: Full LLM analysis with themes and patterns
- **`dashboard.html`**: Interactive dashboard comparing languages
- **`theme_comparison.html`**: Theme frequency heatmap
- **`summary_report_*.txt`**: Human-readable analysis summary

### Analysis Insights
The system provides:
- **Common Themes**: Universal themes across all languages
- **Cultural Patterns**: Language-specific thematic elements
- **Emotional Profiles**: Emotional intensity patterns by language
- **Thematic Clusters**: Grouped dreams by similarity
- **Cross-Linguistic Differences**: Cultural comparison analysis

## 📝 Comprehensive Logging & Analysis System

The Dreams project includes a sophisticated logging and analysis infrastructure for research-grade data collection and analysis.

### 📁 Log Structure
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

### 🔧 Analysis Tools Available

#### **1. Semantic Clustering Analysis** (`semantic_dream_clustering.py`)
- **TF-IDF Vectorization**: Converts dreams to numerical feature vectors
- **K-Means Clustering**: Groups dreams by semantic similarity (3-8 clusters)
- **Cluster Characterization**: Identifies top keywords and themes per cluster
- **Visualization**: 2D t-SNE plots showing cluster distributions
- **Language-Agnostic**: Works across all languages without translation
- **Output**: Cluster assignments, keyword profiles, visual plots

#### **2. Co-occurrence Analysis** (`dream_cooccurrence_analysis.py`)
- **Word Co-occurrence Matrices**: Analyzes relationships between words
- **Statistical Significance**: Identifies meaningful word combinations
- **Cross-Linguistic Patterns**: Compares semantic relationships across languages
- **Network Analysis**: Builds semantic networks from co-occurrence data
- **Cultural Insights**: Reveals language-specific conceptual associations

#### **3. Comprehensive Cross-Linguistic Analysis** (`analyze_comprehensive_dreams.py`)
- **Word Frequency Analysis**: Top words per language with cultural insights
- **Dream Structure Analysis**: Length patterns and narrative complexity
- **Temporal Pattern Analysis**: Generation timing and efficiency metrics
- **Filtering Analysis**: Content rejection patterns and quality metrics
- **Statistical Comparisons**: Cross-language performance and content differences

#### **4. Complete Cultural Analysis** (`complete_dream_analysis.py`)
- **Deep Cross-Linguistic Comparison**: Comprehensive cultural pattern analysis
- **Thematic Content Analysis**: Identifies cultural themes and symbols
- **Cultural Pattern Detection**: Language-specific cultural elements
- **Narrative Structure Analysis**: Story patterns across cultures
- **Publication-Ready Output**: Research-grade analysis reports

#### **5. Statistical Analysis** (`statistical_analysis.py`)
- **Cross-linguistic multilevel modeling**
- **Mixed-effects logistic regression**
- **Success rate and duration analysis**
- **Temporal pattern visualization**

#### **6. Error Analysis** (`analyze_errors.py`)
- **API error pattern analysis**
- **Rejection reason categorization**
- **Cross-session error aggregation**

#### **7. Multilingual Data Analysis** (`analyze_multilingual_data.py`)
- **Comprehensive cross-language comparison**
- **Dream length and content analysis**
- **Prompt entropy and marker usage**

#### **8. Progress Monitoring** (`check_progress.py`)
- **Real-time session monitoring**
- **Success/failure rate tracking**
- **Batch completion status**

#### **9. Dream Length Comparison** (`compare_dream_lengths.py`)
- **Cross-linguistic length analysis**
- **Cultural pattern insights**

### 📈 Research-Grade Metrics
- **Performance Analytics**: Success rates, API timing, batch efficiency
- **Content Analysis**: Dream themes, length distributions, linguistic patterns
- **Quality Metrics**: Rejection rates, consistency, temporal patterns
- **Statistical Modeling**: Mixed-effects models, multilevel analysis

### 🚀 Quick Analysis Commands
```bash
# Semantic clustering analysis (TF-IDF + K-Means)
python semantic_dream_clustering.py

# Word co-occurrence pattern analysis
python dream_cooccurrence_analysis.py

# Comprehensive cross-linguistic analysis
python analyze_comprehensive_dreams.py

# Complete cultural analysis with themes
python complete_dream_analysis.py

# Demo clustering with interactive workflow
python demo_clustering.py

# Check current progress
python check_progress.py

# Analyze dream lengths across languages
python compare_dream_lengths.py

# Full multilingual analysis with visualizations
python analyze_multilingual_data.py

# Statistical analysis for research
python statistical_analysis.py --session-id [session_id]

# Error pattern analysis
python analyze_errors.py --all

# Interactive Streamlit dashboard (recommended)
streamlit run streamlit_dream_analyzer.py
```

For complete documentation of the logging and analysis system, see [`LOGS_ANALYSIS_README.md`](LOGS_ANALYSIS_README.md).

## ⚡ Fast Generation (Temporal Dispersion SUSPENDED)

**Temporal entropy controls are now SUSPENDED by default** for faster dream generation while keeping research capabilities available.

### **Default Behavior (Fast)**
```bash
# Fast generation (minutes to hours instead of days/weeks)
python batch_dream_generator.py --language english --dreams-per-language 100

# Multiple languages quickly
python batch_dream_generator.py --dreams-per-language 100
```

### **Research Mode (Statistical Robustness)**
```bash
# Enable temporal dispersion for research-grade data
python batch_dream_generator.py --language english --dreams-per-language 100 --enable-temporal-dispersion

# Custom temporal settings
python batch_dream_generator.py --language english \
  --enable-temporal-dispersion \
  --temporal-dispersion 4 \
  --min-temporal-dispersion 45
```

### **Performance Comparison**
- **SUSPENDED (Default)**: 100 dreams in ~10-20 minutes
- **ENABLED (Research)**: 100 dreams in ~8-24 hours

For complete details, see [`TEMPORAL_SUSPENSION_README.md`](TEMPORAL_SUSPENSION_README.md).

## 📊 Research Reporting & Data Sharing

The Dreams project includes a comprehensive research reporting system for generating publication-ready reports and shareable data packages.

### 🎓 **Academic Research Features**
- **Multiple Report Formats**: LaTeX (journal submission), Markdown (web), JSON (structured data)
- **Publication-Ready**: Professional tables, high-resolution figures, proper citations
- **Statistical Analysis**: Chi-square tests, ANOVA, cross-linguistic comparisons
- **Data Packages**: Complete datasets with documentation for sharing

### 📝 **Generate Research Reports**
```bash
# Basic research report
python generate_research_report.py --sessions 20250625_155722

# Custom report with metadata
python generate_research_report.py --sessions 20250625_155722 \
  --title "Cross-Linguistic AI Dream Study" \
  --authors "Dr. Smith" "Dr. Jones" \
  --institution "My University"

# Create sample configuration
python generate_research_report.py --create-sample-config
```

### 📦 **Generated Outputs**
- **LaTeX Report**: Ready for academic journal submission
- **Markdown Report**: GitHub and web-friendly documentation
- **Data Package**: Combined datasets with data dictionary
- **Figures**: Publication-quality visualizations (300 DPI)
- **Tables**: Research-grade statistical summaries
- **ZIP Archive**: Complete sharing package

For complete documentation of the research reporting system, see [`RESEARCH_REPORTING_README.md`](RESEARCH_REPORTING_README.md).

## 🔬 Advanced Usage

### Custom Experiments
Modify `first_test.py` to experiment with:
- Different models (GPT-4, Claude, Mistral)
- Various temperatures (0.3-1.0)
- Multiple dreams per configuration
- Different analysis parameters

### Batch Processing
```python
from src.pipeline.dream_generator import DreamResearchPipeline

# Initialize with your API keys
pipeline = DreamResearchPipeline({
    'openai': 'your-key',
    'anthropic': 'your-key',
    'openrouter': 'your-key'
})

# Generate dreams
dreams = await pipeline.generate_dreams(dreams_per_config=5)

# Analyze results
analysis = await pipeline.analyze_results(dreams)

# Save everything
pipeline.save_results(dreams, analysis, "results/my_experiment")
```

### Model Comparison
```python
# Test different models
models = ['gpt-4o', 'claude-3.5-sonnet', 'mistral-large']
temperatures = [0.3, 0.7, 0.9]

for model in models:
    for temp in temperatures:
        # Run experiment with specific model/temperature
        pass
```

## 📈 Analysis Methodology

### 1. Dream Generation
- **Consistent Prompts**: Same prompt translated into each language
- **Temperature Control**: Varies creativity and randomness
- **Quality Filtering**: Removes error responses and incomplete dreams

### 2. Semantic Clustering (TF-IDF + K-Means)
**Technical Implementation:**
```python
# 1. Text Preprocessing
texts = [preprocess_text(dream) for dream in dreams]

# 2. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(texts)

# 3. K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(X)

# 4. Cluster Analysis
cluster_keywords = extract_top_keywords(vectorizer, kmeans)
```

**Key Features:**
- **Optimal Cluster Detection**: Uses elbow method and silhouette analysis
- **Language-Agnostic**: Works without translation, preserving cultural nuances
- **Dimensionality Reduction**: t-SNE visualization for 2D cluster plots
- **Keyword Extraction**: Top TF-IDF terms per cluster for interpretation

### 3. LLM-Based Analysis
- **Theme Extraction**: GPT-4o analyzes each dream for themes
- **Translation**: Non-English dreams translated for consistent analysis
- **JSON Output**: Structured responses for reliable parsing
- **Cultural Context**: Identifies language-specific cultural elements

### 4. Co-occurrence Analysis
**Statistical Approach:**
- **Word Co-occurrence Matrices**: Sliding window approach for semantic relationships
- **Statistical Significance**: Chi-square tests for meaningful associations
- **Cross-Linguistic Networks**: Compares semantic networks across languages
- **Cultural Pattern Detection**: Identifies language-specific conceptual relationships

### 5. Cultural Pattern Analysis
- **Frequency Counting**: Theme frequencies by language
- **Diversity Metrics**: Unique themes per language
- **Cross-Comparison**: Identifies cultural differences
- **Statistical Testing**: Chi-square and ANOVA for significance

### 6. Visualization
- **Interactive Charts**: Plotly-based dashboards
- **Heatmaps**: Theme frequency across languages
- **Cluster Visualization**: Dream groupings and characteristics
- **t-SNE Plots**: 2D semantic space visualization
- **Network Graphs**: Word co-occurrence relationships

## 🔍 Research Applications

This system enables research into:
- **Cultural Bias in AI**: Do LLMs reflect cultural contexts?
- **Universal Themes**: Are certain dream themes universal?
- **Language Effects**: How does language affect AI creativity?
- **Cross-Cultural Psychology**: AI-generated cultural patterns
- **Linguistic Anthropology**: Language-culture relationships

## 📚 Dependencies

### Core Dependencies
- `openai>=1.0.0`: OpenAI API client
- `anthropic`: Anthropic Claude API
- `httpx`: HTTP client for OpenRouter
- `pandas`: Data manipulation
- `plotly`: Interactive visualizations
- `matplotlib`: Static charts
- `seaborn`: Statistical visualizations

### Analysis Dependencies
- `scikit-learn`: TF-IDF vectorization and K-Means clustering
- `nltk`: Natural language processing and text preprocessing
- `deep-translator`: Translation services
- `collections.Counter`: Theme frequency counting
- `json`: Data serialization
- `asyncio`: Asynchronous processing
- `numpy`: Numerical computing for clustering algorithms
- `scipy`: Statistical analysis and distance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [MODELS.md](MODELS.md): Complete model reference and usage guide
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [OpenRouter API](https://openrouter.ai/docs)

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the logs in `logs/` directory for debugging
- Review `MODELS.md` for model-specific information

---

**Research Question**: Do LLMs produce culturally-specific or universal dream narratives when prompted in different languages?

**Answer**: Run the system and find out! 🌙✨