# Gemini 2.5 Pro Dream Generation Guide

## Overview

This script generates **1000 dreams per language** (5000 total) using Google's **Gemini 2.5 Pro** model via Google's direct API. It generates dreams in batches of 100 for optimal performance and resumability.

### Key Features
- **Model**: Google Gemini 2.5 Pro (`google/gemini-2.5-pro`)
- **Languages**: English, Basque, Serbian, Hebrew, Slovenian
- **Target**: 1000 dreams per language (5000 total)
- **Batch Size**: 100 dreams per batch (10 batches per language)
- **Resumable**: Automatically resumes from interruption
- **Same Parameters**: Uses proven optimized configuration (temp=1.1, top_p=0.98)

## 🚀 Quick Start

### 1. Get Google Gemini API Key
1. Visit https://ai.google.dev/
2. Sign up and get your API key
3. Set the environment variable:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-gemini-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### 2. Install Dependencies
```bash
pip install openai pandas numpy
```

### 3. Run the Generator
```bash
python generate_1000_dreams_gemini.py
```

## 📊 What Will Happen

### Batch Structure
- **50 total batches** (10 per language)
- **100 dreams per batch**
- **Automatic progress saving** after each batch
- **Resumable** if interrupted

### Expected Timeline
- **~2-3 hours total** (estimated)
- **~25-30 minutes per language**
- **~2-3 minutes per batch**

### Progress Display
```
🌍 Starting ENGLISH - Batch 1/10
📝 Prompt: You woke up and immediately wrote down your long dream. What did you write?
🎯 Generating dreams 1-100 for english
  ✅ Dream 1: success (2.1s) - 450 chars, 85 words
  ✅ Dream 2: success (1.8s) - 523 chars, 97 words
  ...
📊 Batch 1 complete:
  ✅ Success: 98/100 (98.0% success)
  📈 Avg length: 487 chars, 91 words
  ⏱️  Duration: 180.5s (avg 1.8s per dream)
```

## 📁 Output Structure

Dreams will be saved to `logs_gemini_2_5_pro/`:

```
logs_gemini_2_5_pro/
├── english/
│   └── gemini-2-5-pro/
│       └── session_GEMINI_20250109_123456/
│           ├── dreams.csv
│           ├── api_calls.csv
│           └── session_data.json
├── basque/
├── hebrew/
├── serbian/
├── slovenian/
├── all_dreams_GEMINI_20250109_123456.csv
├── all_api_calls_GEMINI_20250109_123456.csv
└── session_summary_GEMINI_20250109_123456.json
```

## 🔄 Resumption

If the script is interrupted:
1. **Simply run it again** - it will automatically detect existing progress
2. **Resume from where it left off** - no duplicate generation
3. **Progress is saved** after each batch completion

## 📈 Monitoring Progress

### During Generation
The script shows real-time progress:
- Current language and batch
- Individual dream generation status
- Batch completion statistics
- Overall progress tracking

### Language Status Example
```
📋 ENGLISH STATUS:
  Current: 347/1000 dreams (34.7%)
  Batches: 3/10 completed
  🎯 Need 653 more dreams (7 batches)
```

### Final Summary
```
🎊 SUCCESS! Gemini 2.5 Pro batch generation completed!
✅ Generated 4,987/5,000 dreams
📊 Overall success rate: 99.7%
📁 All data saved to: logs_gemini_2_5_pro/
🆔 Session ID: GEMINI_20250109_123456

📊 Language Breakdown:
   English: 1000 dreams (100.0% success)
    Basque:  998 dreams (99.8% success)
    Hebrew:  999 dreams (99.9% success)
   Serbian:  995 dreams (99.5% success)
 Slovenian:  995 dreams (99.5% success)
```

## ⚙️ Configuration Details

### Model Settings
- **Model**: `google/gemini-2.5-pro`
- **Temperature**: 1.1 (enhanced creativity)
- **Top-p**: 0.98 (wider vocabulary)
- **Max Tokens**: 1000
- **No System Prompt**: Pure immediate dream scenario

### Language Prompts
Each language uses the optimized prompt from your existing configuration:

- **English**: "You woke up and immediately wrote down your long dream. What did you write?"
- **Basque**: "Esnatu eta berehala zure amets luzea idatzi duzu. Zer idatzi duzu?"
- **Serbian**: "Пробудио си се и одмах записао свој дугачак сан. Шта си написао?"
- **Hebrew**: "התעוררת ומיד כתבת את החלום הארוך שלך. מה כתבת?"
- **Slovenian**: "Zbudil si se in takoj zapisal svoje dolge sanje. Kaj si zapisal?"

## 💰 Cost Estimation

Google AI pricing for Gemini 2.5 Pro (check current rates at https://ai.google.dev/pricing):
- **Input**: ~$0.00125 per 1K tokens
- **Output**: ~$0.005 per 1K tokens
- **Estimated total**: ~$30-50 for 5000 dreams

## 🛠️ Troubleshooting

### Common Issues

**"GEMINI_API_KEY environment variable not set"**
- Make sure you set the environment variable correctly
- Restart your terminal/command prompt after setting it

**"Google Gemini API key is required"**
- Check that your API key is valid
- Ensure your Google AI account has sufficient quota/credits

**"openai library not available"**
```bash
pip install openai
```

**Generation seems slow**
- This is normal - high-quality generation takes time
- Each dream takes 1-3 seconds
- The script includes small delays to be respectful to the API

### Interruption Recovery
If the script stops:
1. **Check the last output** - it shows how many dreams were completed
2. **Simply restart** - the script will resume automatically
3. **Progress is preserved** - no work is lost

## 📊 Integration with Existing Analysis

The generated dreams will work with your existing analysis tools:

```bash
# Analyze the new Gemini dreams
python analyze_optimized_v2.py GEMINI_20250109_123456

# Run thematic analysis
python dream_thematic_analysis.py --data-dir logs_gemini_2_5_pro

# Statistical comparison with other models
python compare_model_performance.py
```

## 🎯 Success Criteria

**Target Achievement:**
- ✅ 1000 dreams per language (5000 total)
- ✅ Using Gemini 2.5 Pro model
- ✅ Same proven prompt and parameters
- ✅ Organized in batches of 100
- ✅ Full resumability and progress tracking

**Quality Expectations:**
- High success rate (>95%)
- Rich, culturally authentic dreams
- Consistent with previous optimized results
- No AI disclaimers or artifacts

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Ensure your Google Gemini API key is valid and has sufficient quota
3. Verify all dependencies are installed
4. Check the `logs_gemini_2_5_pro/` directory for error logs
5. The script includes detailed error messages to help diagnose problems

---

**Ready to generate 5000 dreams with Gemini 2.5 Pro!** 🚀 