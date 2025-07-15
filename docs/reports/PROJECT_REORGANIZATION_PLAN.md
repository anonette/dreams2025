# Dreams Project Reorganization Plan

## Current Problem
The project has grown organically with 50+ files scattered in the root directory, making it difficult to navigate and maintain.

## Proposed New Structure

```
Dreams/
├── README.md                          # Main project overview & navigation
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment template
│
├── scripts/                          # Main execution scripts
│   ├── generation/                   # Dream generation scripts
│   │   ├── __init__.py
│   │   ├── batch_dream_generator.py
│   │   ├── generate_1000_dreams_gemini.py
│   │   ├── generate_1000_dreams_mistral.py
│   │   ├── generate_1000_dreams_openrouter_single.py
│   │   ├── generate_1000_dreams_openrouter_multimodel.py
│   │   ├── generate_optimized_dream_batch.py
│   │   ├── generate_900_batch_system.py
│   │   ├── generate_900_dreams.py
│   │   ├── generate_100_more_dreams.py
│   │   ├── generate_hebrew_slovenian_dreams.py
│   │   ├── optimized_batch_v2.py
│   │   ├── run_scaled_generation.py
│   │   └── scale_dream_generation.py
│   │
│   ├── analysis/                     # Analysis scripts
│   │   ├── __init__.py
│   │   ├── analyze_optimized_dreams.py
│   │   ├── analyze_optimized_translations.py
│   │   ├── analyze_optimized_v2.py
│   │   ├── comprehensive_cultural_analysis.py
│   │   ├── detailed_statistical_analysis.py
│   │   ├── dream_thematic_analysis.py
│   │   ├── statistical_analysis.py
│   │   ├── translation_based_statistical_analysis.py
│   │   ├── semantic_dream_analyzer.py
│   │   └── generate_research_report.py
│   │
│   └── utilities/                    # Utility scripts
│       ├── __init__.py
│       ├── cleanup_empty_sessions.py
│       ├── check_data_status.py
│       ├── json_date_fix.py
│       ├── show_basque_fix.py
│       ├── verify_900_dream_system.py
│       └── translation_manager.py
│
├── tests/                            # All test files
│   ├── __init__.py
│   ├── api_tests/
│   │   ├── test_gemini_api.py
│   │   ├── test_single_dream.py
│   │   ├── test_model_names.py
│   │   └── run_gemini_validation.py
│   │
│   ├── validation_tests/
│   │   ├── test_gemini_2_5_flash_validation.py
│   │   ├── test_enhanced_parameters.py
│   │   ├── test_experimental_dreams.py
│   │   ├── test_optimized_config.py
│   │   ├── test_pure_immediate_dreams.py
│   │   └── test_three_prompt_strategies.py
│   │
│   └── translation_tests/
│       ├── test_hebrew_translation.py
│       ├── test_refined_translations.py
│       ├── test_translation_sample.py
│       ├── test_translation_stats.py
│       └── test_translations.py
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── languages/
│   │   ├── __init__.py
│   │   ├── optimized_dream_languages.py
│   │   ├── immediate_dream_languages.py
│   │   ├── pure_immediate_dream_languages.py
│   │   └── experimental_languages.py
│   │
│   └── models/
│       ├── __init__.py
│       └── model_configs.py
│
├── src/                              # Core library (keep existing structure)
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── languages.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm_interface.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dream_analysis.py
│   │   ├── llm_discourse_analyzer.py
│   │   ├── nlp_analyzer.py
│   │   └── typological_analyzer.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── research_reporter.py
│   └── visualization/
│       ├── __init__.py
│       └── report_generator.py
│
├── apps/                             # Interactive applications
│   ├── __init__.py
│   ├── streamlit_dream_analyzer.py
│   ├── streamlit_dream_analyzer_optimized.py
│   ├── run_dashboard.py
│   └── semantic_demo.py
│
├── docs/                             # Documentation
│   ├── README.md                     # Documentation index
│   ├── guides/
│   │   ├── AUTOMATION_README.md
│   │   ├── BATCH_GENERATOR_README.md
│   │   ├── DASHBOARD_README.md
│   │   ├── GEMINI_GENERATION_GUIDE.md
│   │   ├── GEMINI_VALIDATION_README.md
│   │   ├── LOGS_ANALYSIS_README.md
│   │   ├── OPTIMIZED_CONFIGURATION_GUIDE.md
│   │   ├── OPTIMIZED_USAGE_GUIDE.md
│   │   ├── RESEARCH_REPORTING_README.md
│   │   ├── STATISTICAL_APPROACH_README.md
│   │   ├── STREAMLIT_OPTIMIZED_README.md
│   │   ├── STRUCTURED_LOGGING_README.md
│   │   └── TYPOLOGICAL_ANALYSIS_README.md
│   │
│   ├── models/
│   │   └── MODELS.md
│   │
│   └── reports/
│       ├── basque_statistical_fix_20250706_160120.md
│       └── batch900_completion_report_session.md
│
├── data/                             # Data directories
│   ├── logs/                         # Consolidated log directories
│   │   ├── gpt4o/                    # Renamed from "logs-with system prompts"
│   │   ├── gemini_1_5_pro/
│   │   ├── gemini_2_5_flash/
│   │   ├── gemini_2_5_pro/
│   │   ├── mistral/
│   │   ├── openrouter/               # New for OpenRouter logs
│   │   └── optimized_v2/
│   │
│   ├── analysis_output/
│   ├── cultural_analysis_output/
│   ├── enhanced_parameters_output/
│   ├── experimental_dreams_output/
│   ├── July5reports/
│   ├── optimized_validation_output/
│   ├── pure_immediate_dreams_output/
│   ├── refined_translations_output/
│   ├── research_reports/
│   ├── three_strategies_output/
│   ├── translations/
│   ├── translations-old/
│   └── test_jsons/
│
└── automation/                       # Automation scripts
    ├── __init__.py
    ├── automate_dreams.py
    └── batch_files/
        ├── run_900_batches.bat
        ├── run_scaled_generation.bat
        ├── run_streamlit_optimized.bat
        └── check_api_key.bat
```

## File Categorization

### Generation Scripts (→ `scripts/generation/`)
- `batch_dream_generator.py` ⭐ (main batch generator)
- `generate_1000_dreams_gemini.py` ⭐
- `generate_1000_dreams_mistral.py` ⭐
- `generate_1000_dreams_openrouter_single.py` ⭐ (new)
- `generate_1000_dreams_openrouter_multimodel.py` ⭐ (new)
- `generate_optimized_dream_batch.py`
- `generate_900_batch_system.py`
- `generate_900_dreams.py`
- `generate_100_more_dreams.py`
- `generate_hebrew_slovenian_dreams.py`
- `optimized_batch_v2.py`
- `run_scaled_generation.py`
- `scale_dream_generation.py`

### Analysis Scripts (→ `scripts/analysis/`)
- `analyze_optimized_dreams.py`
- `analyze_optimized_translations.py`
- `analyze_optimized_v2.py`
- `comprehensive_cultural_analysis.py`
- `detailed_statistical_analysis.py`
- `dream_thematic_analysis.py`
- `statistical_analysis.py`
- `translation_based_statistical_analysis.py`
- `semantic_dream_analyzer.py`
- `generate_research_report.py`

### Test Scripts (→ `tests/`)
- `test_gemini_api.py`
- `test_single_dream.py`
- `test_model_names.py`
- `run_gemini_validation.py`
- `test_gemini_2_5_flash_validation.py`
- `test_enhanced_parameters.py`
- `test_experimental_dreams.py`
- `test_optimized_config.py`
- `test_pure_immediate_dreams.py`
- `test_three_prompt_strategies.py`
- `test_hebrew_translation.py`
- `test_refined_translations.py`
- `test_translation_sample.py`
- `test_translation_stats.py`
- `test_translations.py`

### Configuration Files (→ `config/`)
- `optimized_dream_languages.py`
- `immediate_dream_languages.py`
- `pure_immediate_dream_languages.py`
- `experimental_languages.py`

### Interactive Apps (→ `apps/`)
- `streamlit_dream_analyzer.py`
- `streamlit_dream_analyzer_optimized.py`
- `run_dashboard.py`
- `semantic_demo.py`

### Utility Scripts (→ `scripts/utilities/`)
- `cleanup_empty_sessions.py`
- `check_data_status.py`
- `json_date_fix.py`
- `show_basque_fix.py`
- `verify_900_dream_system.py`
- `translation_manager.py`

### Documentation (→ `docs/`)
- All `*_README.md` files
- `MODELS.md`
- Report markdown files

### Automation (→ `automation/`)
- `automate_dreams.py`
- All `.bat` files

## Implementation Steps

1. **Create new directory structure**
2. **Move files to appropriate directories**
3. **Update import statements in moved files**
4. **Create `__init__.py` files for Python packages**
5. **Update main README.md with navigation**
6. **Test that everything still works**

## Benefits of New Structure

✅ **Clear separation of concerns**
- Generation scripts in one place
- Analysis scripts in another
- Tests organized by category

✅ **Better navigation**
- Easy to find what you need
- Logical grouping of related files

✅ **Scalability**
- Easy to add new scripts in appropriate categories
- Clear place for new functionality

✅ **Maintenance**
- Easier to update related files
- Clear dependencies between components

✅ **Documentation**
- All guides in one place
- Clear project overview

## Next Steps

1. Switch to Code mode to implement the file moves
2. Update import paths
3. Test the reorganized structure
4. Update main README with navigation

This reorganization will make the Dreams project much more professional and maintainable!