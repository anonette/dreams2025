# Dream Translations Summary

Generated: 2025-07-07 13:58:07
Session: session_OPT_V2_20250707_080149

## Overview

- **Total Dreams**: 500
- **Languages**: 5
- **Translation Files**: 12

## Language Breakdown

| Language | Dreams | Original File | Translation File | CSV File |
|----------|--------|---------------|------------------|----------|
| English | 100 | N/A | N/A | english_translations_session_OPT_V2_20250707_080149.csv |
| Basque | 100 | basque_original_session_OPT_V2_20250707_080149.txt | basque_translated_session_OPT_V2_20250707_080149.txt | basque_translations_session_OPT_V2_20250707_080149.csv |
| Hebrew | 100 | hebrew_original_session_OPT_V2_20250707_080149.txt | hebrew_translated_session_OPT_V2_20250707_080149.txt | hebrew_translations_session_OPT_V2_20250707_080149.csv |
| Serbian | 100 | serbian_original_session_OPT_V2_20250707_080149.txt | serbian_translated_session_OPT_V2_20250707_080149.txt | serbian_translations_session_OPT_V2_20250707_080149.csv |
| Slovenian | 100 | slovenian_original_session_OPT_V2_20250707_080149.txt | slovenian_translated_session_OPT_V2_20250707_080149.txt | slovenian_translations_session_OPT_V2_20250707_080149.csv |

## Files Created

- `basque_translations_session_OPT_V2_20250707_080149.csv`
- `basque_translations_session_OPT_V2_20250707_080149.json`
- `english_translations_session_OPT_V2_20250707_080149.csv`
- `english_translations_session_OPT_V2_20250707_080149.json`
- `hebrew_translations_session_OPT_V2_20250707_080149.csv`
- `hebrew_translations_session_OPT_V2_20250707_080149.json`
- `serbian_translations_session_OPT_V2_20250707_080149.csv`
- `serbian_translations_session_OPT_V2_20250707_080149.json`
- `slovenian_translations_session_OPT_V2_20250707_080149.csv`
- `slovenian_translations_session_OPT_V2_20250707_080149.json`
- `translation_summary_session_OPT_V2_20250707_080149.md`

## Usage Instructions

### JSON Files
Complete data with metadata, original text, and translations. Best for programmatic access.

### CSV Files
Tabular format for spreadsheet analysis. Easy to open in Excel or Google Sheets.

### TXT Files
- `*_original_*.txt`: Original language text only
- `*_translated_*.txt`: English translations with original context

### Loading Translations
```python
import json
import pandas as pd

# Load JSON data
with open('translations/slovenian_translations_session_OPT_V2_20250707_080149.json', 'r') as f:
    data = json.load(f)

# Load CSV data
df = pd.read_csv('translations/slovenian_translations_session_OPT_V2_20250707_080149.csv')
```
