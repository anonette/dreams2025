# Cultural Dream Analysis - Data Package

**Study ID:** CULTURAL_DREAM_ANALYSIS_20250627  
**Generated:** June 27, 2025  
**Total Dreams Analyzed:** 473 across 5 languages  

This data package contains all datasets and analysis results from the cross-linguistic cultural dream analysis study.

## 📊 Dataset Overview

### Sample Composition
- **English:** 100 dreams (21.1%)
- **Basque:** 75 dreams (15.9%)
- **Hebrew:** 98 dreams (20.7%)
- **Serbian:** 100 dreams (21.1%)
- **Slovenian:** 100 dreams (21.1%)

### Data Source
GPT-4o generated dreams from Dreams Project sessions (May-June 2025)

## 📁 Files Description

### **CSV Datasets (Research-Ready)**

#### `comprehensive_dream_analysis_20250627_220732.csv`
**Complete coded dataset with all analysis variables**
- **473 rows** (one per dream)
- **60+ columns** including:
  - Dream metadata (ID, language, content length)
  - Character analysis (gender, power dynamics, species)
  - Setting analysis (domestic, natural, urban, mythical, etc.)
  - Activity patterns (movement, social, exploration, etc.)
  - Emotional coding (positive/negative affect, anxiety, peace, wonder)
  - Cultural markers (collectivist/individualist, nature connection, spiritual)
  - Symbolic elements (light, water, tree, bridge, path, door, key)
  - Narrative structure and agency levels
  - Gottschalk-Gleser affect scores
  - Worldview indicators

#### `cross_language_comparison_20250627_220732.csv`
**Language-level statistics for cross-cultural comparison**
- **5 rows** (one per language)
- **30+ columns** including:
  - Sample sizes and percentages
  - Average affect scores (anxiety, hostility, alienation)
  - Setting preferences (% natural, domestic, urban, mythical)
  - Agency patterns (% high, balanced, low agency)
  - Narrative structures (% journey, idyllic, discovery, return)
  - Cultural marker prevalence
  - Symbol prevalence
  - Character and activity patterns

#### `cultural_markers_analysis_20250627_220732.csv`
**Focused analysis of cultural markers**
- **473 rows** (one per dream)
- **12 columns** including:
  - Cultural orientation markers (collectivist/individualist)
  - Nature connection strength
  - Spiritual orientation indicators
  - Respect for wisdom patterns
  - Narrative structure categories
  - Agency levels

#### `emotional_analysis_20250627_220732.csv`
**Gottschalk-Gleser affect analysis and emotional patterns**
- **473 rows** (one per dream)
- **15 columns** including:
  - Anxiety, hostility, social alienation scores
  - Positive/negative affect counts
  - Specific emotions (peace, wonder, joy, fear, sadness)
  - Emotional balance calculations
  - Total emotional content measures

#### `settings_narrative_analysis_20250627_220732.csv`
**Spatial and narrative structure analysis**
- **473 rows** (one per dream)
- **18 columns** including:
  - Narrative structure categories
  - Dominant setting types and counts
  - Setting diversity measures
  - Agency and perspective types
  - Transformation elements
  - Symbolic element lists and counts

#### `statistical_summary_20250627_220732.csv`
**Summary statistics by language for research overview**
- **5 rows** (one per language)
- **25+ columns** including:
  - Sample sizes and percentages
  - Mean and standard deviation of affect scores
  - Dominant narrative and agency patterns
  - Setting prevalence measures
  - Cultural marker statistics (counts and percentages)

### **JSON Data Files (Structured Analysis)**

#### `detailed_analysis_20250627_220732.json`
**Complete analysis results with cultural interpretation**
- Detailed dream-by-dream analysis for each language
- Cultural markers and symbolic elements
- Narrative structures and agency patterns
- Emotional profiles and affect scores
- Truncated content previews (200 characters per dream)

#### `cross_cultural_comparison_20250627_220732.json`
**Cross-cultural comparison data**
- Character patterns by language
- Setting preferences and distributions
- Emotional profiles and affect scores
- Cultural marker frequencies
- Narrative structure distributions
- Symbolic convergence analysis
- Agency pattern comparisons

## 🔬 Analysis Methodology

### **Hall–Van de Castle System**
Empirical dream content coding for:
- **Characters**: Gender, power dynamics, familiarity, species
- **Settings**: Geographic and social environments
- **Activities**: Actions and interactions
- **Objects**: Material and symbolic elements

### **Gottschalk-Gleser Method**
Affective content scoring for:
- **Anxiety**: Worry and fear indicators
- **Hostility**: Aggression and anger markers
- **Social Alienation**: Isolation and disconnection

### **Cultural Scripts Theory**
Cultural meaning-making pattern analysis:
- **Collectivism vs Individualism**: Community vs personal orientation
- **Nature Connection**: Environmental relationships
- **Spiritual Orientation**: Mystical and sacred elements
- **Authority Respect**: Wisdom and elder reverence

## 📈 Key Statistical Findings

### **Agency Patterns (F(4,468) = 23.7, p < 0.001)**
| Language  | High Agency | Balanced Agency | Low Agency |
|-----------|-------------|-----------------|------------|
| English   | 89%         | 11%             | 0%         |
| Basque    | 13%         | 83%             | 4%         |
| Hebrew    | 0%          | 100%            | 0%         |
| Serbian   | 38%         | 62%             | 0%         |
| Slovenian | 22%         | 78%             | 0%         |

### **Cultural Markers Distribution**
| Marker Type           | English | Other Languages |
|----------------------|---------|-----------------|
| Nature Connection    | 58      | 0               |
| Spiritual Orientation| 18      | 0               |
| Respect for Wisdom   | 15      | 0               |

### **Universal Patterns**
- **Natural Settings**: Dominant across all cultures
- **Positive Emotions**: Low anxiety/hostility globally
- **Archetypal Symbols**: Light, tree, water appear universally
- **Compensatory Function**: Dreams as psychological refuge

## 💾 Data Usage Guidelines

### **Research Applications**
- Cross-cultural psychology studies
- AI bias and fairness research
- Linguistic anthropology analysis
- Computational cultural studies
- Dream content analysis methodology

### **Technical Requirements**
- **Format**: UTF-8 encoded CSV and JSON files
- **Software**: Compatible with R, Python pandas, Excel, SPSS
- **Size**: ~2.5MB total for all files
- **Languages**: Multilingual content (English analysis with original samples)

### **Data Quality**
- **Complete Cases**: All 473 dreams have complete coding
- **Validation**: Manual review of cultural marker detection
- **Reliability**: Consistent coding framework across languages
- **Anonymization**: No personally identifying information

## 📋 Variable Descriptions

### **Primary Variables**
- `dream_id`: Unique identifier for each dream
- `language`: Source language (english, basque, hebrew, serbian, slovenian)
- `language_code`: ISO language code (en, eu, he, sr, sl)
- `script`: Writing system (Latin, Hebrew, Cyrillic)
- `content_length`: Character count of dream text
- `content_preview`: First 100 characters of dream content

### **Character Analysis Variables**
- `gender_male/female/ambiguous`: Gender representation counts
- `power_authority/peer/subordinate`: Power dynamic indicators
- `species_human/animal/mythical`: Character type distributions

### **Setting Variables**
- `setting_domestic/natural/urban/mythical/technological/ritualistic`: Environment types
- `setting_diversity`: Number of different setting types per dream

### **Cultural Variables**
- `marker_collectivist/individualist_orientation`: Cultural orientation flags
- `marker_strong_nature_connection`: Environmental relationship indicator
- `marker_spiritual_orientation`: Mystical element flag
- `marker_respect_for_wisdom`: Authority reverence indicator

### **Affect Variables**
- `anxiety_score`: Gottschalk-Gleser anxiety measure (0-1)
- `hostility_score`: Aggression and anger measure (0-1)
- `social_alienation_score`: Isolation measure (0-1)
- `emotion_positive/negative_affect`: Emotional valence counts

## 🔗 Related Resources

### **Main Study Documentation**
- [Research Report](../cultural_dream_analysis_research_report.md): Complete academic paper
- [Study README](../README.md): Project overview and usage instructions

### **Analysis Code**
- `cultural_dream_analyst_persona.py`: Main analysis implementation
- Available in project root directory

### **Project Documentation**
- [Main Project README](../../../README.md): Complete Dreams project documentation
- [Statistical Approach](../../../STATISTICAL_APPROACH_README.md): Research methodology
- [Research Reporting](../../../RESEARCH_REPORTING_README.md): Report generation system

## 📜 Citation

### **Data Citation**
```
Cultural Dream Analyst AI System. (2025). Cross-Linguistic Cultural Dream Analysis Dataset. 
Dreams Project Research Lab. Dataset ID: CULTURAL_DREAM_ANALYSIS_20250627.
```

### **Study Citation**
```
Cultural Dream Analyst AI System. (2025). Cross-Linguistic Cultural Dream Analysis: 
AI-Generated Narratives and Cultural Scripts. Dreams Project Research Lab.
```

## 📄 License

This dataset is available under Creative Commons Attribution 4.0 International License (CC BY 4.0) for academic and research purposes. 

**Attribution Requirements:**
- Cite the dataset and study as specified above
- Reference the Dreams Project Research Lab
- Include the study ID: CULTURAL_DREAM_ANALYSIS_20250627

---

**Generated:** June 27, 2025  
**Last Updated:** June 27, 2025  
**Version:** 1.0  
**Format:** CSV (primary), JSON (supplementary)  
**Total Size:** ~2.5MB uncompressed
