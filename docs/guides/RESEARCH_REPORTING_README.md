# Research Reporting & Data Sharing System

## 📋 Overview

The Dreams project includes a sophisticated research reporting system that generates publication-ready reports and shareable data packages from your cross-linguistic dream generation logs. This system is designed for academic research, enabling you to create structured reports suitable for journal submission and comprehensive data packages for collaborative research.

## 🎯 Key Features

### **Research-Grade Reports**
- **Multiple Formats**: LaTeX (for journal submission), Markdown (for web/GitHub), JSON (structured data)
- **Academic Structure**: Abstract, Introduction, Methodology, Results, Discussion, Conclusion
- **Publication-Ready**: Professional tables, high-resolution figures, proper citations
- **Auto-Generated Content**: Intelligent report generation from your log data

### **Statistical Analysis**
- **Descriptive Statistics**: Comprehensive summary statistics by language and session
- **Hypothesis Testing**: Chi-square tests, ANOVA, significance testing
- **Cross-Linguistic Comparison**: Success rates, duration analysis, content patterns
- **Temporal Analysis**: Time-based pattern identification

### **Data Sharing Packages**
- **Complete Datasets**: Combined CSV files ready for analysis
- **Data Dictionary**: Detailed documentation of all variables and formats
- **Analysis Results**: All statistical outputs in structured JSON format
- **Compressed Archives**: ZIP packages for easy sharing and distribution

### **Visualization**
- **Research Figures**: Publication-quality charts and graphs at 300 DPI
- **Cross-Language Comparisons**: Success rates, durations, content analysis
- **Statistical Plots**: Error bars, confidence intervals, significance indicators

## 🚀 Quick Start

### **Generate a Basic Report**
```bash
# Generate report for specific session(s)
python generate_research_report.py --sessions 20250625_155722

# Generate report with custom metadata
python generate_research_report.py --sessions 20250625_155722 \
  --title "My Dream Study" \
  --authors "Dr. Smith" "Dr. Jones" \
  --institution "My University"
```

### **Using Configuration Files**
```bash
# Create sample configuration
python generate_research_report.py --create-sample-config

# Use configuration file
python generate_research_report.py --sessions 20250625_155722 --config sample_metadata.json
```

### **Advanced Options**
```bash
# Multiple sessions with full customization
python generate_research_report.py \
  --sessions 20250625_155722 20250625_154026 \
  --title "Cross-Linguistic AI Dream Generation: A Comparative Study" \
  --authors "Dr. Jane Smith" "Dr. John Doe" "Dr. Maria Garcia" \
  --institution "University of Computational Linguistics" \
  --keywords "cross-linguistic" "AI" "dreams" "cultural bias" \
  --no-data-package \
  --output-dir "my_reports"
```

## 📊 Generated Report Structure

### **Report Files**
```
research_reports/report_YYYYMMDD_HHMMSS/
├── report_YYYYMMDD_HHMMSS_report.md          # Markdown report
├── report_YYYYMMDD_HHMMSS_report.tex         # LaTeX for journals
├── report_YYYYMMDD_HHMMSS_complete_report.json # Structured data
├── report_YYYYMMDD_HHMMSS_metadata.yaml      # Study metadata
├── README.md                                  # Package documentation
├── tables/                                    # Research tables
│   ├── language_comparison.csv
│   └── statistical_tests.csv
├── figures/                                   # Publication figures
│   ├── success_rates_by_language.png
│   └── duration_by_language.png
├── data_package/                             # Data sharing
│   ├── combined_api_calls.csv
│   ├── combined_dreams.csv
│   ├── analysis_results.json
│   └── data_dictionary.json
└── report_YYYYMMDD_HHMMSS_sharing_package.zip # Complete package
```

### **Report Sections**

1. **Abstract**: Study summary with key findings
2. **Introduction**: Research questions and significance
3. **Methodology**: Data collection and analysis methods
4. **Results**: Statistical findings and cross-linguistic patterns
5. **Discussion**: Interpretation and implications
6. **Conclusion**: Summary and future research directions

## 📈 Statistical Analysis Capabilities

### **Descriptive Statistics**
- Overall success rates and confidence intervals
- Language-specific performance metrics
- Duration analysis (mean, median, std, quartiles)
- Dream content characteristics (length, word count)
- Temporal patterns (hourly, daily trends)

### **Inferential Statistics**
- **Chi-square tests**: Language vs success rate associations
- **ANOVA**: Duration and length differences across languages
- **Confidence intervals**: Population parameter estimates
- **Effect sizes**: Practical significance of findings

### **Cross-Linguistic Comparisons**
- Success rate rankings by language
- Duration performance comparisons
- Dream length and content analysis
- Cultural pattern identification

## 🔬 Research Applications

### **Academic Research**
- **Journal Articles**: LaTeX format ready for submission
- **Conference Papers**: Structured findings and visualizations
- **Thesis/Dissertation**: Comprehensive data and analysis
- **Grant Proposals**: Preliminary data and methodology

### **Collaborative Research**
- **Data Sharing**: Standardized datasets with documentation
- **Reproducible Research**: Complete methodology and code
- **Cross-Institution Studies**: Portable data packages
- **Open Science**: GitHub-ready reports and data

### **Quality Assurance**
- **Performance Monitoring**: Success rate tracking
- **Error Analysis**: Failure pattern identification
- **System Optimization**: Duration and efficiency analysis
- **Bias Detection**: Cross-linguistic fairness assessment

## 📝 Configuration Options

### **Study Metadata**
```json
{
  "title": "Cross-Linguistic Patterns in AI-Generated Dream Narratives",
  "authors": ["Dr. Jane Smith", "Dr. John Doe"],
  "institution": "University of Computational Linguistics",
  "keywords": ["cross-linguistic", "AI", "dreams", "cultural bias"],
  "abstract": "This study examines how large language models..."
}
```

### **Command Line Options**
- `--sessions`: Session IDs to include (required)
- `--title`: Study title
- `--authors`: Author names (space-separated)
- `--institution`: Institution name
- `--keywords`: Study keywords
- `--abstract`: Custom abstract
- `--config`: JSON configuration file
- `--logs-dir`: Input directory (default: logs)
- `--output-dir`: Output directory (default: research_reports)
- `--no-data-package`: Skip data package creation
- `--study-id`: Custom study identifier

## 🔍 Quality Control

### **Data Validation**
- Automatic detection of missing or corrupted files
- Validation of statistical assumptions
- Cross-reference checks between data sources
- Comprehensive error reporting

### **Statistical Rigor**
- Appropriate test selection based on data types
- Multiple comparison corrections
- Effect size calculations
- Confidence interval reporting

### **Reproducibility**
- Complete methodology documentation
- Data provenance tracking
- Version control integration
- Replication instructions

## 🌐 Sharing and Collaboration

### **Data Sharing Best Practices**
- **Documentation**: Complete data dictionaries and metadata
- **Privacy**: No personal information in shared datasets
- **Licensing**: Clear usage terms and attribution
- **Versioning**: Unique identifiers for each dataset

### **Citation Format**
The system automatically generates citation information:
```
Smith, J., Doe, J. (2025). Cross-Linguistic Patterns in AI-Generated Dream Narratives. 
University of Computational Linguistics.
```

### **Repository Integration**
Reports are formatted for easy integration with:
- **GitHub**: Markdown reports with embedded figures
- **Academic repositories**: LaTeX files and datasets
- **Research platforms**: Structured JSON data
- **Data archives**: Complete ZIP packages

## 🛠️ Technical Requirements

### **Dependencies**
```bash
pip install pandas numpy scipy matplotlib seaborn plotly pyyaml
```

### **System Requirements**
- Python 3.8+
- 2GB RAM for large datasets
- 500MB disk space for reports
- UTF-8 encoding support for multilingual data

## 📖 Examples

### **Conference Paper Report**
```bash
python generate_research_report.py \
  --sessions 20250625_155722 \
  --title "Measuring Cultural Bias in Multilingual AI Dream Generation" \
  --authors "Dr. Research" \
  --institution "AI Ethics Lab" \
  --keywords "AI bias" "multilingual" "fairness" "NLP"
```

### **Journal Article Report**
```bash
python generate_research_report.py \
  --sessions 20250625_155722 20250625_154026 \
  --config journal_metadata.json \
  --output-dir "journal_submission"
```

### **Data Sharing Package**
```bash
python generate_research_report.py \
  --sessions 20250625_155722 \
  --title "Public Dataset: Cross-Linguistic Dream Generation" \
  --no-data-package false
```

## 🚀 Advanced Usage

### **Batch Report Generation**
```python
from src.reporting.research_reporter import ResearchReporter, ResearchMetadata

# Generate multiple reports
sessions_list = [
    ["20250625_155722"],
    ["20250625_154026", "20250625_152541"],
    ["20250625_150334"]
]

for i, sessions in enumerate(sessions_list):
    metadata = ResearchMetadata(
        study_id=f"DREAM_BATCH_{i+1}",
        title=f"Dream Study Batch {i+1}",
        # ... other metadata
    )
    
    reporter = ResearchReporter()
    result = reporter.generate_research_report(sessions, metadata)
    print(f"Generated: {result['report_id']}")
```

This research reporting system transforms your Dreams project logs into professional, publication-ready reports suitable for academic research, collaboration, and data sharing.

## 🔗 Related Documentation

- [`LOGS_ANALYSIS_README.md`](LOGS_ANALYSIS_README.md): Complete analysis capabilities
- [`README.md`](README.md): Main project documentation
- [`STATISTICAL_APPROACH_README.md`](STATISTICAL_APPROACH_README.md): Statistical methodology

---

**Generate your first research report**: `python generate_research_report.py --sessions [your_session_id]`
