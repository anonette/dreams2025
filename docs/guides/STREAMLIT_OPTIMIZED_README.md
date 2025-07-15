# Streamlit Dream Analysis - Optimized V2 Data

This is a modified version of the Dream Analysis Streamlit app that works specifically with the `logs_optimized_v2` directory.

## Quick Start

### Option 1: Windows Batch File (Recommended for Windows)
```bash
run_streamlit_optimized.bat
```

### Option 2: Python Script
```bash
python run_streamlit_optimized.py
```

### Option 3: Direct Streamlit Command
```bash
streamlit run streamlit_dream_analyzer_optimized.py
```

## Features

### 📊 Data Overview
- Real-time scanning of `logs_optimized_v2` directory
- Summary statistics for each language
- Success rates and dream counts
- Session information

### 🎨 Thematic Analysis
- Interactive heatmaps showing theme prevalence across languages
- Theme variation analysis
- Cultural pattern visualization

### 📈 Statistical Analysis
- Success rate comparisons
- Dream count analysis
- Statistical tables with detailed metrics

### 🔍 Data Explorer
- Browse individual dreams by language
- Dream content viewer
- Basic text statistics

## Data Structure Expected

The app expects the following directory structure:

```
logs_optimized_v2/
├── english/
│   └── gpt-4o/
│       └── session_OPT_V2_*/
│           ├── dreams.csv
│           ├── api_calls.csv
│           └── session_data.json
├── basque/
│   └── gpt-4o/
│       └── session_OPT_V2_*/
│           └── ...
├── hebrew/
├── serbian/
└── slovenian/
```

## What's Different from the Original

1. **Data Source**: Points to `logs_optimized_v2` instead of `logs`
2. **Session Detection**: Looks for `session_OPT_V2_*` sessions
3. **Enhanced UI**: Updated titles and labels to indicate optimized data
4. **Real-time Data Loading**: Scans actual data files for current statistics

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- NumPy

## Installation

If you don't have the required packages:

```bash
pip install streamlit pandas plotly numpy
```

## Usage Tips

1. **First Run**: The app will automatically scan your `logs_optimized_v2` directory
2. **Data Updates**: Refresh the browser to reload if you add new data
3. **Performance**: Large datasets may take a moment to load
4. **Navigation**: Use the sidebar to see data overview and the main tabs for different analyses

## Troubleshooting

### "No dream data found"
- Ensure `logs_optimized_v2` directory exists
- Check that session directories contain `dreams.csv` files
- Verify the directory structure matches the expected format

### "Directory not found"
- Make sure you're running the script from the correct directory
- The `logs_optimized_v2` folder should be in the same directory as the script

### Import Errors
- Install missing packages: `pip install streamlit pandas plotly numpy`
- Some advanced features may be disabled if optional modules aren't available

## Access

Once running, open your browser to:
**http://localhost:8501**

The app will automatically open in your default browser when started.

## Stopping the App

- Close the browser tab/window
- In the terminal/command prompt, press `Ctrl+C`
- Close the command prompt window (if using batch file)

## Comparison with Original App

| Feature | Original App | Optimized V2 App |
|---------|-------------|------------------|
| Data Source | `logs/` | `logs_optimized_v2/` |
| Session Pattern | `session_*` | `session_OPT_V2_*` |
| UI Labels | Standard | "Optimized V2" branding |
| Data Loading | Static fallback data | Real-time scanning |
| Statistics | Hardcoded | Dynamic from actual files |

This ensures you get accurate, up-to-date analysis of your optimized dream generation results! 