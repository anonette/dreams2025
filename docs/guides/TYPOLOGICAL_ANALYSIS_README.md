# Typological Linguistic Analysis

This module provides comprehensive typological linguistic analysis capabilities for the Dreams project, exploring relationships between linguistic structural features and narrative patterns in AI-generated dream content.

## 🔬 Overview

The typological analysis system combines:
- **12 WALS (World Atlas of Language Structures) features** per language
- **7 narrative dimensions** scored on a 0.0-1.0 scale
- **Cross-linguistic distance calculations** 
- **Hierarchical clustering** of languages by narrative patterns
- **Rich visualizations** and export capabilities

## 🌍 Supported Languages

Currently supports 5 languages with complete WALS feature profiles:

| Language  | Code | Family | Notable Features |
|-----------|------|--------|------------------|
| English   | `en` | Germanic | SVO, nominative-accusative, analytic |
| Basque    | `eu` | Isolate | SOV, ergative-absolutive, agglutinative |
| Hebrew    | `he` | Semitic | VSO, Semitic root system, evidentials |
| Serbian   | `sr` | Slavic | SVO, rich aspect system, case marking |
| Slovenian | `sl` | Slavic | SVO, dual number, complex case system |

## 📊 WALS Features Analyzed

1. **Tense/Aspect** - Temporal marking systems
2. **Alignment** - Grammatical alignment patterns (nom-acc, erg-abs)
3. **Subject Expression** - Pro-drop parameters
4. **Modality** - Modal expression systems
5. **Evidentiality** - Evidential marking requirements
6. **Word Order** - Basic constituent order (SOV, SVO, VSO)
7. **Case Marking** - Case system complexity
8. **Definiteness** - Article and definiteness systems
9. **Gender** - Gender marking systems
10. **Number** - Number marking (singular/plural/dual)
11. **Negation** - Negation strategies
12. **Voice** - Voice system complexity

## 📝 Narrative Dimensions

Each dream is scored on 7 dimensions (0.0-1.0 scale):

| Dimension | Description | 0.0 = | 1.0 = |
|-----------|-------------|-------|-------|
| **Dreamer Agency** | Control/agency of the dreamer | Passive observer | Active agent |
| **Other Agents** | Presence of other characters | No others | Many active others |
| **Interaction** | Social interaction levels | Solitary | Highly interactive |
| **Emotion** | Emotional intensity | Neutral | Highly emotional |
| **Temporal Coherence** | Timeline coherence | Fragmented | Linear narrative |
| **Cultural Motifs** | Culture-specific elements | Universal | Culture-specific |
| **Narrative Complexity** | Story structure complexity | Simple | Complex plot |

## 🤖 Scoring Methods

### LLM-Based Scoring (Primary)
- Uses GPT-4o with structured JSON prompts
- Temperature: 0.1 for consistency
- Comprehensive validation and error handling
- Language-aware scoring considerations

### Heuristic Scoring (Fallback)
- Text-based analysis using keyword detection
- Word frequency and pattern analysis
- Language-independent baseline approach
- Used when LLM scoring fails or API unavailable

## 🎯 Usage

### Streamlit Interface
```bash
streamlit run streamlit_dream_analyzer.py
```
Navigate to the **🔬 Typological Analysis** tab for interactive analysis.

### Command Line
```bash
# With typological analysis
python generate_research_report.py --sessions 20250706_093349 --title "My Study"

# Skip typological analysis
python generate_research_report.py --sessions 20250706_093349 --no-typological-analysis

# Limit dreams per language
python generate_research_report.py --sessions 20250706_093349 --max-dreams-per-language 25
```

### Programmatic Usage
```python
import asyncio
from src.analysis.typological_analyzer import TypologicalAnalyzer
from src.models.llm_interface import LLMInterface

# Initialize with LLM interface
api_keys = {'openai': 'your-key-here'}
llm_interface = LLMInterface(api_keys)
analyzer = TypologicalAnalyzer(llm_interface)

# Analyze dreams
dreams_by_language = {
    'english': [
        {'dream_id': 'dream_001', 'dream_text': 'I was walking...'},
        # ... more dreams
    ],
    'hebrew': [
        {'dream_id': 'dream_002', 'dream_text': 'הייתי הולך...'},
        # ... more dreams
    ]
}

# Run analysis
results = await analyzer.analyze_dreams(dreams_by_language, max_dreams_per_language=50)

# Generate visualizations
visualizations = analyzer.create_visualizations(results)

# Export results
from pathlib import Path
exported_files = analyzer.export_results(results, Path('output/'))
```

## 📈 Visualizations

### 1. Narrative Heatmap
- **Type**: Interactive heatmap
- **Shows**: Language × narrative dimension correlation matrix
- **Use**: Identify which languages show higher/lower scores on specific dimensions

### 2. Typological Distance Matrix
- **Type**: Distance matrix visualization
- **Shows**: WALS feature-based distances between languages
- **Use**: Understand structural linguistic relationships

### 3. Hierarchical Clustering Dendrogram
- **Type**: Tree diagram
- **Shows**: Language clustering based on narrative patterns
- **Use**: Discover which languages group together by dream characteristics

### 4. Radar Charts
- **Type**: Multi-dimensional profile charts
- **Shows**: Language-specific narrative dimension profiles
- **Use**: Compare overall narrative "signatures" across languages

## 📤 Export Formats

### JSON Export
Complete analysis results with all data structures:
```json
{
  "results": [...],
  "total_analyzed": 100,
  "correlations": {...},
  "language_distances": {...},
  "clusters": {...},
  "summary_stats": {...}
}
```

### CSV Exports
- **`language_narrative_means.csv`** - Mean scores by language/dimension
- **`wals_features.csv`** - WALS feature matrix
- **`typological_distances.csv`** - Pairwise language distances

### Markdown Report
Academic-style comprehensive report with:
- Executive summary
- Methodology description
- Key findings
- Statistical tables
- Interpretation guidelines

## 🔧 Configuration

### API Key Requirements
For LLM-based scoring, provide API keys via environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # optional
export OPENROUTER_API_KEY="your-openrouter-key"  # optional
```

### Analysis Parameters
- **`max_dreams_per_language`**: Limit sample size (default: 50)
- **`temperature`**: LLM temperature for scoring (default: 0.1)
- **`clustering_method`**: Hierarchical clustering method (default: 'ward')

## 🛡️ Error Handling

The system includes robust error handling for:
- **API failures**: Graceful degradation to heuristic scoring
- **Empty/malformed data**: Safe handling with informative warnings
- **JSON parsing errors**: Multiple fallback strategies
- **Visualization failures**: Fallback charts when clustering impossible
- **Memory constraints**: Efficient processing of large datasets

## 📊 Statistical Approach

### Methodology
- **Exploratory**: Pure data-driven discovery without theoretical bias
- **Empirical**: Reports observed patterns, not causal relationships
- **Quantitative**: Numerical scoring with statistical validation
- **Cross-linguistic**: Systematic comparison across language families

### Interpretation Guidelines
1. **Correlation ≠ Causation**: All results show associations, not causes
2. **Sample Dependency**: Results reflect the specific dream sample analyzed
3. **Cultural Context**: Consider cultural factors in interpretation
4. **Statistical Significance**: Large samples provide more reliable patterns

## 🔍 Research Applications

### Computational Linguistics
- Cross-linguistic variation in AI text generation
- Typological influences on narrative structure
- Language model bias assessment

### Digital Humanities
- Cultural patterns in dream narratives
- Cross-cultural psychology via computational methods
- Comparative literature analysis

### AI Ethics & Bias
- Language-specific biases in large language models
- Cultural representation in AI-generated content
- Multilingual AI system evaluation

## 🚀 Performance Considerations

### Optimization Tips
1. **Batch Processing**: Analyze multiple dreams simultaneously
2. **API Rate Limits**: Built-in rate limiting for LLM calls
3. **Memory Management**: Efficient data structures for large datasets
4. **Caching**: Results cached to avoid recomputation

### Scalability
- **Languages**: Easily extensible to additional languages
- **Features**: WALS feature set can be expanded
- **Dimensions**: Narrative dimensions can be customized
- **Models**: Support for multiple LLM providers

## 🔮 Future Enhancements

### Planned Features
- [ ] Additional WALS features (phonological, morphological)
- [ ] Custom narrative dimension definitions
- [ ] Diachronic analysis capabilities
- [ ] Integration with external linguistic databases
- [ ] Real-time analysis streaming

### Research Directions
- [ ] Causal analysis methods
- [ ] Individual variation modeling
- [ ] Cultural context integration
- [ ] Multimodal analysis (text + metadata)

## 📚 References

### Linguistic Typology
- Dryer, Matthew S. & Martin Haspelmath (eds.) 2013. The World Atlas of Language Structures Online. Leipzig: Max Planck Institute for Evolutionary Anthropology.

### Computational Methods
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236-244.

### Dream Research
- Domhoff, G. W. (2003). The scientific study of dreams: Neural networks, cognitive development, and content analysis. American Psychological Association.

---

## 🆘 Support

For issues, questions, or contributions:
1. Check the error logs for detailed diagnostic information
2. Verify API key configuration for LLM-based scoring
3. Ensure sufficient sample sizes for meaningful analysis
4. Review the statistical interpretation guidelines

**Generated by the Dreams Project Typological Analysis Engine** 🌍💭 