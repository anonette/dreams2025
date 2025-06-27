# Temporal Clustering and Prompt Entropy Controls

## Overview

This document describes the enhanced statistical controls implemented to address temporal clustering biases and session randomness scope limitations in LLM-based cross-linguistic dream research.

## Temporal Clustering Concerns

### Problem Statement
LLMs like GPT-4o may exhibit temporal coherence within small time windows:
- Calls made in bursts may hit the same underlying server or micro-version
- This risks capturing localized quirks in the model's sampling distribution
- Results may not be representative of the broader population
- Analogous to sampling only one time of day when studying human behavior

### Solution: Enhanced Temporal Dispersion

#### Multi-Level Temporal Controls
1. **Minimum Dispersion**: 30 minutes between individual calls (configurable)
2. **Randomized Delays**: 1.5x to 3x minimum delay with random variation
3. **Extended Windows**: 10% chance of much longer delays (2-24 hours)
4. **Batch Separation**: Original 2+ hour delays between batches maintained

#### Implementation
```python
class TemporalDispersionManager:
    def get_next_call_delay(self) -> float:
        min_delay = self.config.min_temporal_dispersion_minutes * 60
        base_delay = random.uniform(min_delay, min_delay * 3)
        
        # 10% chance of extended delay for temporal diversity
        if random.random() < 0.1:
            extended_delay = random.uniform(
                self.config.temporal_dispersion_hours * 3600,
                self.config.max_temporal_dispersion_hours * 3600
            )
            base_delay = max(base_delay, extended_delay)
        
        return base_delay
```

## Prompt Entropy Measures

### Problem Statement
Even with unique user IDs, identical prompts may cluster in the model's sampling space, potentially introducing systematic biases.

### Solution: Invisible Marker System

#### Prompt ID Tagging
- Each API call receives a unique 8-character prompt ID
- Enables clustering analysis and pattern detection
- Logged for post-hoc statistical analysis

#### Template Suffixes with Invisible Markers
Five types of invisible markers with semantic invariance:
1. **HTML Comments**: `<!-- -->`
2. **Zero-Width Space**: `\u200B`
3. **Word Joiner**: `\u2060`
4. **Zero-Width No-Break Space**: `\uFEFF`
5. **Zero-Width Non-Joiner**: `‌`

#### Marker Placement Strategy
- **Prefix**: Marker before prompt text
- **Suffix**: Marker after prompt text  
- **Middle**: Marker after first word (if possible)

#### Statistical Configuration
- **Default Usage**: 30% of prompts receive markers
- **Random Selection**: Marker type and position randomly chosen
- **Logging**: All variants and markers logged for analysis

## Enhanced Logging and Analysis

### New Data Fields
```json
{
  "prompt_id": "a1b2c3d4",
  "base_prompt": "Finish: Last night I dreamt of…",
  "modified_prompt": "Finish:<!-- --> Last night I dreamt of…",
  "marker_info": "marker_0_prefix",
  "used_invisible_markers": true,
  "temporal_delay_seconds": 1847.3
}
```

### Temporal Statistics Tracking
- Call time recording for all API requests
- Inter-call interval analysis (mean, std, min, max)
- Total temporal span measurement
- Statistical distribution analysis

### Entropy Analysis Export
```json
{
  "prompt_entropy_statistics": {
    "total_calls": 500,
    "calls_with_markers": 148,
    "marker_usage_rate": 0.296,
    "marker_type_distribution": {
      "marker_0_prefix": 32,
      "marker_1_suffix": 29,
      "none": 352
    },
    "unique_prompt_ids": 500,
    "language_specific_entropy": {
      "english": {
        "total_calls": 100,
        "calls_with_markers": 31,
        "marker_usage_rate": 0.31,
        "unique_prompt_ids": 100
      }
    }
  }
}
```

## Command Line Interface

### New Arguments
```bash
# Temporal Dispersion Controls
--min-temporal-dispersion 30      # Minutes between calls (default: 30)
--max-temporal-dispersion 24      # Max hours for diversity (default: 24)

# Prompt Entropy Controls  
--use-prompt-variants              # Enable invisible markers (default: true)
--no-prompt-variants              # Disable all variants
--marker-probability 0.3          # Marker usage rate (default: 0.3)
--prompt-variant-types 5          # Number of marker types (default: 5)
```

### Usage Examples
```bash
# Standard enhanced protocol
python batch_dream_generator.py --language english --dreams-per-language 100

# Conservative temporal dispersion (longer delays)
python batch_dream_generator.py --language english \
  --min-temporal-dispersion 60 --max-temporal-dispersion 48

# High entropy with 50% marker usage
python batch_dream_generator.py --language english \
  --marker-probability 0.5 --prompt-variant-types 10

# Disable entropy controls (pure temporal only)
python batch_dream_generator.py --language english --no-prompt-variants
```

## Statistical Rigor Benefits

### Session Independence
- Enhanced UUID generation with entropy suffixes
- Multiple sources of randomization per call
- Broader temporal coverage reduces server clustering

### Bias Mitigation
- Invisible markers provide tokenization variation
- Random temporal spacing prevents burst effects
- Prompt ID tracking enables post-hoc pattern detection

### Research Validity
- Samples span broader population of model states
- Reduces risk of micro-version or server-specific quirks
- Maintains semantic equivalence across variants

## File Outputs

### Enhanced Logs
1. **`temporal_statistics_{session_id}.json`** - Detailed temporal analysis
2. **`all_api_calls_{session_id}.csv`** - Complete call metadata with entropy fields
3. **`session_summary_{session_id}.json`** - Enhanced session statistics

### Checkpoint Recovery
- Temporal manager state preservation
- Call time history restoration
- Seamless resume with maintained temporal patterns

## Research Applications

### Cross-Linguistic Analysis
- Compare entropy effectiveness across language families
- Detect script-specific clustering patterns
- Analyze temporal coherence in different writing systems

### Model Behavior Studies
- Identify server-level consistency patterns
- Measure impact of temporal dispersion on output diversity
- Quantify effectiveness of invisible marker entropy

### Statistical Controls
- Use prompt IDs for multilevel modeling covariates
- Include temporal delays as random effects
- Control for marker types in regression analyses

## Implementation Notes

### Performance Considerations
- Average delay: 45-90 minutes between calls
- Extended delays: 2-24 hour windows for 10% of calls
- Memory overhead: Minimal (tracking arrays only)

### Robustness Features
- Checkpoint system preserves temporal state
- Error logging includes all entropy metadata
- Graceful degradation if entropy generation fails

### Quality Assurance
- Semantic invariance testing for all marker types
- Temporal pattern verification in logs
- Statistical validation of entropy distribution 