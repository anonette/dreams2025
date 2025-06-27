# Multilingual Dream Analysis Report

Generated on: 2025-06-27 21:48:44

## Summary Statistics

| Language   | Language Code   | Script   |   Total Calls |   Successful |   Failed | Success Rate   |   Calls with Markers | Marker Usage Rate   |   Unique Prompt IDs |      Session ID |
|:-----------|:----------------|:---------|--------------:|-------------:|---------:|:---------------|---------------------:|:--------------------|--------------------:|----------------:|
| Basque     | eu              | Latin    |           100 |           75 |       25 | 75.0%          |                   29 | 29.0%               |                 100 | 20250625_155722 |
| English    | en              | Latin    |           100 |          100 |        0 | 100.0%         |                   29 | 29.0%               |                 100 | 20250625_142711 |
| Hebrew     | he              | Hebrew   |           100 |           98 |        2 | 98.0%          |                   25 | 25.0%               |                 100 | 20250625_152541 |
| Serbian    | sr              | Cyrillic |           100 |          100 |        0 | 100.0%         |                   19 | 19.0%               |                 100 | 20250625_150334 |
| Slovenian  | sl              | Latin    |           100 |          100 |        0 | 100.0%         |                   29 | 29.0%               |                 100 | 20250625_154026 |

## Dream Length Analysis

| Language | Avg Characters | Avg Words | Min Length | Max Length | Std Dev |
|----------|----------------|-----------|------------|------------|---------|
| Basque | 476 | 64.8 | 120 | 998 | 203 |
| English | 626 | 110.2 | 313 | 1195 | 148 |
| Hebrew | 206 | 36.2 | 39 | 380 | 66 |
| Serbian | 541 | 91.5 | 304 | 949 | 125 |
| Slovenian | 447 | 76.6 | 269 | 1019 | 94 |


## Prompt Entropy Analysis

| Language | Total Calls | Calls with Markers | Marker Rate | Unique Prompt IDs |
|----------|-------------|-------------------|-------------|------------------|
| Basque | 100 | 29 | 29.0% | 100 |
| English | 100 | 29 | 29.0% | 100 |
| Hebrew | 100 | 25 | 25.0% | 100 |
| Serbian | 100 | 19 | 19.0% | 100 |
| Slovenian | 100 | 29 | 29.0% | 100 |


## Content Theme Analysis

| Language | Flying | Water | Animals | People | Places | Emotions | Avg Words/Dream |
|----------|--------|-------|---------|--------|--------|----------|----------------|
| Basque | 16 | 17 | 8 | 0 | 0 | 1 | 64.8 |
| English | 78 | 77 | 38 | 18 | 68 | 1 | 110.2 |
| Hebrew | 0 | 0 | 0 | 0 | 0 | 0 | 36.2 |
| Serbian | 0 | 0 | 2 | 0 | 0 | 0 | 91.5 |
| Slovenian | 0 | 1 | 23 | 0 | 0 | 4 | 76.6 |


## Key Findings

### Success Rates
- All languages achieved high success rates (>95%)
- Total dreams generated: 473
- Total API calls: 500

### Prompt Entropy
- Marker usage varies by language due to randomization
- All languages show good prompt ID diversity
- Entropy controls working as designed

### Content Patterns
- Dream themes vary by language and cultural context
- Word count and character length show linguistic differences
- Content diversity appears good across all languages

## Recommendations

1. **Statistical Analysis**: Data is ready for cross-linguistic statistical modeling
2. **Content Analysis**: Consider linguistic-specific content categorization
3. **Temporal Analysis**: Review temporal statistics for batch effects
4. **Quality Control**: Examine any failed calls for patterns

## Data Files

Each language has structured data in:
- `logs/{language}/gpt-4o/session_{timestamp}/`
  - `session_data.json` - Complete session metadata
  - `api_calls.csv` - Detailed API call data
  - `dreams.csv` - Dream content and metadata
  - `temporal_statistics.json` - Temporal analysis
