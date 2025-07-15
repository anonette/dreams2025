# Dream Translations Summary

Generated: 2025-07-06 16:49:07
Session: session_20250706_093349

## Overview

- **Total Dreams**: 499
- **Languages**: 5
- **Translation Files**: 5

## Language Breakdown

| Language | Dreams | Original File | Translation File | CSV File |
|----------|--------|---------------|------------------|----------|
| English | 100 | N/A | N/A | english_translations_session_20250706_093349.csv |
| Basque | 100 | basque_original_session_20250706_093349.txt | basque_translated_session_20250706_093349.txt | basque_translations_session_20250706_093349.csv |
| Hebrew | 100 | hebrew_original_session_20250706_093349.txt | hebrew_translated_session_20250706_093349.txt | hebrew_translations_session_20250706_093349.csv |
| Serbian | 100 | serbian_original_session_20250706_093349.txt | serbian_translated_session_20250706_093349.txt | serbian_translations_session_20250706_093349.csv |
| Slovenian | 99 | slovenian_original_session_20250706_093349.txt | slovenian_translated_session_20250706_093349.txt | slovenian_translations_session_20250706_093349.csv |

## Files Created

- `basque_translations_session_20250706_093349.csv`
- `basque_translations_session_20250706_093349.json`
- `english_translations_session_20250706_093349.csv`
- `english_translations_session_20250706_093349.json`
- `hebrew_translations_session_20250706_093349.csv`
- `hebrew_translations_session_20250706_093349.json`
- `serbian_translations_session_20250706_093349.csv`
- `serbian_translations_session_20250706_093349.json`
- `slovenian_translations_session_20250706_093349.csv`
- `slovenian_translations_session_20250706_093349.json`
- `translation_summary_session_20250706_093349.md`
- `translation_summary_session_20250706_093349_20250706_161558.md`

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
with open('translations/slovenian_translations_session_20250706_093349.json', 'r') as f:
    data = json.load(f)

# Load CSV data
df = pd.read_csv('translations/slovenian_translations_session_20250706_093349.csv')
```
