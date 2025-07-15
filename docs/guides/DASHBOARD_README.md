# 🌙 Dream Analysis Dashboard

## Overview

The Dream Analysis Dashboard is a comprehensive Streamlit web interface that provides access to all dream analysis capabilities with session-based result management.

## Features

### 📊 **Data Overview Tab**
- Real-time view of available dream data
- Language-wise breakdown of dreams
- Success rate metrics
- Interactive visualizations

### 🔬 **Cultural Analysis Tab**
- **Basic Cultural Analysis**: Hall-Van de Castle + Gottschalk-Gleser analysis
- **Advanced Cultural Analysis**: LLM-based cultural interpretation
- Character, setting, and emotional analysis
- Cross-cultural comparisons

### 🌍 **Multilingual Analysis Tab**
- Cross-linguistic dream content comparison
- Dream length analysis across languages
- Content theme identification
- Prompt entropy analysis

### 📈 **Progress & Statistics Tab**
- Session progress tracking
- Data quality metrics
- Success rate analysis
- Statistical summaries

### 📁 **Results & Downloads Tab**
- Session-based file organization
- Download individual analysis results
- Download complete session results as ZIP
- File management and organization

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit plotly streamlit-plotly-events
```

### Launch Dashboard

#### Option 1: Direct Launch
```bash
streamlit run streamlit_dream_analyzer.py
```

#### Option 2: Using Launcher Script
```bash
python run_dashboard.py
```

### Access Dashboard

Open your web browser and navigate to:
```
http://localhost:8501
```

## 📋 How to Use

### 1. **Data Overview**
- Review available dream data
- Check language coverage
- Verify data quality

### 2. **Run Analyses**
- Choose analysis type from tabs
- Click "Run Analysis" buttons
- Monitor progress in real-time
- View results immediately

### 3. **Download Results**
- Go to "Results & Downloads" tab
- Download individual analysis results
- Download complete session results
- Results are organized by session ID

## 🗂️ Session Management

### **Session ID Format**
```
YYYYMMDD_HHMMSS
```

### **Session Organization**
```
analysis_output/
├── session_20250705_211527/
│   ├── cultural_analysis/
│   │   ├── cultural_dream_analysis_report.md
│   │   ├── cultural_analysis_data.json
│   │   └── individual_dream_analysis.csv
│   ├── persona_analysis/
│   │   ├── comprehensive_dream_analysis.csv
│   │   ├── cross_language_comparison.csv
│   │   └── cultural_interpretation_report.txt
│   ├── multilingual_analysis/
│   │   └── multilingual_analysis_report.md
│   └── progress_analysis/
│       └── progress_report.json
```

## 📊 Analysis Types

### **Cultural Analysis**
- **Input**: Dream CSV files from logs/
- **Output**: Cultural patterns, character analysis, emotional content
- **Files**: Markdown reports, JSON data, CSV datasets

### **Multilingual Analysis**
- **Input**: Cross-language dream data
- **Output**: Language comparisons, content themes, success rates
- **Files**: Markdown reports, statistical summaries

### **Progress Analysis**
- **Input**: Session metadata
- **Output**: Progress metrics, quality indicators
- **Files**: JSON progress reports

## 🔧 Technical Details

### **Data Sources**
- `logs/[language]/gpt-4o/session_*/dreams.csv`
- `logs/[language]/gpt-4o/session_*/api_calls.csv`
- `logs/[language]/gpt-4o/session_*/session_data.json`

### **Analysis Modules**
- `cultural_dream_analysis.py` - Basic cultural analysis
- `cultural_dream_analyst_persona.py` - Advanced LLM analysis
- `Tests/analyze_multilingual_data.py` - Cross-linguistic analysis
- `Tests/check_progress.py` - Progress tracking

### **Dependencies**
- **Streamlit**: Web interface framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data processing
- **NumPy**: Numerical computations

## 📈 Visualization Features

### **Interactive Charts**
- Dreams by language (bar chart)
- Success rates (bar chart)
- Progress tracking (metrics)
- Real-time data updates

### **Data Tables**
- Sortable language breakdown
- Session statistics
- Analysis results preview

## 💾 File Management

### **Automatic Organization**
- Results organized by session ID
- Timestamped file naming
- Hierarchical folder structure

### **Download Options**
- Individual analysis ZIP files
- Complete session downloads
- Structured file organization

## 🔍 Troubleshooting

### **Common Issues**

#### No Data Available
- Ensure dream generation has been run
- Check logs/ directory structure
- Verify CSV files exist

#### Analysis Fails
- Check Python environment
- Ensure all dependencies installed
- Review error messages in dashboard

#### Downloads Not Working
- Check browser download settings
- Ensure sufficient disk space
- Verify session directory exists

### **Support**
- Check console output for errors
- Review generated log files
- Verify file permissions

## 🎯 Best Practices

### **Session Management**
- Start new session for each research phase
- Download results before ending sessions
- Keep session IDs for reference

### **Analysis Workflow**
1. **Data Overview** - Verify data quality
2. **Cultural Analysis** - Run basic then advanced
3. **Multilingual Analysis** - Cross-linguistic comparison
4. **Progress Check** - Verify completeness
5. **Download Results** - Save all outputs

### **File Organization**
- Use descriptive session names
- Download results regularly
- Maintain backup copies

## 📞 Support

For issues or questions:
- Check DASHBOARD_README.md
- Review console output
- Verify data availability
- Check file permissions

---

**🌙 Dreams Analysis Platform** | Built with Streamlit | Session-based result management 