# Statistical Approach for Cross-Linguistic Dream Research

This document outlines the rigorous statistical methodology implemented for robust cross-linguistic dream research with proper controls for LLM stochasticity and temporal drift.

## 🎯 **Research Design**

### **Multilevel Modeling Structure**
- **Level 1**: Individual dream generation (API calls)
- **Level 2**: Batches (temporal clusters)
- **Level 3**: Languages (fixed effects)

### **Statistical Controls**
- **Session Independence**: Unique UUID for each API call
- **Temporal Dispersion**: Distributed sampling across time
- **Batch Random Effects**: Accounting for batch-level variation
- **Language Fixed Effects**: Cross-linguistic comparisons

## 📊 **Sampling Protocol**

### **GPT-4o Sampling Parameters**
```python
# Rigorous sampling configuration
SamplingConfig(
    batch_size=50,                    # Optimal batch size for temporal dispersion
    temporal_dispersion_hours=2,      # Hours between batches
    session_independence=True,        # Unique user IDs per call
    min_samples_per_language=500,     # Minimum for statistical power
    user_id_regeneration=True,        # UUID per call
    presence_penalty=0.1,             # Reduces repetition
    frequency_penalty=0.0,            # Allows natural variation
    temperature=1.0,                  # Maximum creativity
    top_p=1.0                         # Full vocabulary access
    # No max_tokens - model decides natural dream length
)
```

### **API Call Hygiene**
- **Stateless calls**: No context reuse
- **Random delays**: 1.5-3 seconds between calls
- **Temporal dispersion**: 2+ hours between batches
- **UUID generation**: Unique user ID per call

## 🔬 **Statistical Models**

### **1. Basic Logistic Regression**
```python
# Language effect on success rate
model = smf.glm(
    formula="success_binary ~ language_factor",
    data=df,
    family=sm.families.Binomial()
).fit()
```

### **2. Mixed-Effects Model**
```python
# Batch as random effect, language as fixed effect
model = mixedlm(
    "success_binary ~ language_factor", 
    df, 
    groups=df["batch_id"]
).fit()
```

### **3. Theme Analysis**
```python
# Binary features for dream content
for theme in ['flying', 'falling', 'chase', 'water', 'family', 'work', 'anxiety', 'travel']:
    model = smf.glm(
        formula=f"contains_{theme} ~ language_factor",
        data=success_df,
        family=sm.families.Binomial()
    ).fit()
```

## 📈 **Data Structure**

### **API Call Log Fields**
```json
{
    "call_id": "unique-uuid",
    "batch_id": "batch-uuid",
    "user_id": "unique-user-uuid",
    "timestamp": "2024-01-01T12:00:00",
    "language": "english",
    "language_code": "en",
    "script": "Latin",
    "model": "gpt-4o",
    "temperature": 1.0,
    "top_p": 1.0,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.0,
    "prompt": "Finish: Last night I dreamt of…",
    "dream_number": 1,
    "batch_size": 50,
    "dream": "the actual dream content...",
    "status": "success",
    "duration_seconds": 2.345,
    "start_time": "2024-01-01T12:00:00",
    "end_time": "2024-01-01T12:00:02",
    "session_id": "20240101_120000",
    "temporal_dispersion": 2,
    "session_independence": true
}
```

### **Statistical Features**
- **Binary outcomes**: success_binary, error_binary, filtered_binary
- **Temporal features**: hour_of_day, day_of_week, batch_sequence
- **Batch features**: batch_success_rate, batch_duration_mean
- **Content features**: contains_flying, contains_falling, etc.
- **Language features**: is_english, is_basque, etc.

## 🚀 **Usage**

### **1. Generate Dreams with Statistical Protocol**
```bash
# Generate 500 dreams per language with rigorous sampling
python batch_dream_generator.py --dreams-per-language 500 --batch-size 50 --temporal-dispersion 2

# Generate for specific language
python batch_dream_generator.py --language english --dreams-per-language 500
```

### **2. Run Statistical Analysis**
```bash
# Analyze session data
python statistical_analysis.py --session-id 20240101_120000
```

### **3. Complete Research Pipeline**
```python
from batch_dream_generator import BatchDreamGenerator, SamplingConfig
from statistical_analysis import DreamStatisticalAnalyzer

# Configure sampling
config = SamplingConfig(
    batch_size=50,
    temporal_dispersion_hours=2,
    min_samples_per_language=500
)

# Generate dreams
generator = BatchDreamGenerator(api_keys, sampling_config=config)
results = await generator.generate_all_languages(500)

# Analyze results
analyzer = DreamStatisticalAnalyzer()
analysis = analyzer.run_complete_analysis(session_id)
```

## 📊 **Output Files**

### **Generation Logs**
- `{language}_{model}_{timestamp}.json`: Complete language data
- `{language}_{model}_{timestamp}_api_calls.csv`: API call details
- `{language}_{model}_{timestamp}_dreams.csv`: Dream content
- `session_summary_{session_id}.json`: Session statistics
- `all_api_calls_{session_id}.csv`: All calls for analysis
- `all_dreams_{session_id}.csv`: All dreams for analysis
- `batch_tracker_{session_id}.json`: Batch-level statistics

### **Statistical Results**
- `descriptive_stats_{session_id}_{timestamp}.json`: Descriptive statistics
- `multilevel_analysis_{session_id}_{timestamp}.json`: Model results
- `analysis_summary_{session_id}_{timestamp}.txt`: Human-readable summary
- `{session_id}/success_rate_by_language.png`: Visualizations
- `{session_id}/duration_by_language.png`: Performance analysis
- `{session_id}/temporal_patterns.png`: Time series analysis
- `{session_id}/batch_success_distribution.png`: Batch variation

## 🔍 **Statistical Power**

### **Sample Size Justification**
- **500 dreams per language**: Sufficient for detecting medium effect sizes
- **5 languages**: Enables cross-linguistic comparisons
- **2,500 total dreams**: Robust statistical power
- **50 dreams per batch**: Optimal for temporal dispersion

### **Effect Size Detection**
- **Small effects**: Cohen's d = 0.2 (80% power with 500 samples)
- **Medium effects**: Cohen's d = 0.5 (99% power with 500 samples)
- **Large effects**: Cohen's d = 0.8 (99.9% power with 500 samples)

## 🛡️ **Robustness Measures**

### **Temporal Dispersion**
- **Distributed sampling**: Spread across multiple days
- **Time zone variation**: Natural temporal diversity
- **Batch intervals**: 2+ hours between batches
- **Session independence**: No cross-batch memory

### **Stochasticity Controls**
- **UUID generation**: Unique user ID per call
- **Temperature=1.0**: Maximum randomness
- **Top_p=1.0**: Full vocabulary access
- **Presence penalty**: Reduces repetition
- **No frequency penalty**: Allows natural variation

### **Error Handling**
- **API error logging**: Full error context
- **Response filtering**: Remove invalid responses
- **Retry logic**: Handle transient failures
- **Rate limiting**: Respect API limits

## 📋 **Research Questions**

### **Primary Hypotheses**
1. **Language Effect**: Do different languages produce different dream themes?
2. **Script Effect**: Does writing system affect dream content?
3. **Temporal Effect**: Does time of day affect dream generation?
4. **Batch Effect**: Is there batch-level variation in dream quality?

### **Secondary Analyses**
1. **Theme Prevalence**: Which themes are most common across languages?
2. **Content Length**: Do dream lengths vary by language?
3. **Success Rates**: Are some languages more challenging for the model?
4. **Temporal Patterns**: Are there time-based patterns in dream content?

## 🎯 **Expected Outcomes**

### **Statistical Significance**
- **Language differences**: Expected p < 0.001
- **Script effects**: Expected p < 0.01
- **Temporal effects**: Expected p < 0.05
- **Batch variation**: Expected significant random effects

### **Effect Sizes**
- **Language effects**: Expected medium to large effects
- **Script effects**: Expected small to medium effects
- **Temporal effects**: Expected small effects
- **Batch effects**: Expected moderate random variation

## 🔬 **Future Extensions**

### **Advanced Models**
- **Hierarchical models**: Language → Script → Batch → Dream
- **Time series analysis**: Temporal patterns in dream content
- **Content analysis**: NLP-based theme extraction
- **Cross-validation**: Robust model validation

### **Additional Controls**
- **Model versioning**: Track GPT-4o updates
- **Prompt variation**: Test different prompt formulations
- **Cultural context**: Include cultural background variables
- **Linguistic features**: Add linguistic typology variables

This statistical approach ensures robust, replicable research with proper controls for the inherent stochasticity of LLM-based dream generation. 