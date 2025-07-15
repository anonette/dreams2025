# Fixing Basque Statistical Anomalies with Semantic Similarity

Generated: 2025-07-06 16:01:20

## Problem Statement

Basque language shows statistical anomalies in dream thematic analysis:
- Many themes show 0% or very low percentages
- Creates 'zero-inflation' in statistical tests
- Median values much lower than other languages
- Skews cross-linguistic comparisons

## Root Cause

**Translation → Keyword Mismatch**:
1. Basque dreams contain rich thematic content
2. Google Translate produces valid English translations
3. English translations use different words than expected keywords
4. Keyword matching fails → false negatives

## Semantic Similarity Solution

**Method**: TF-IDF vectorization + Cosine similarity
**Benefits**:
- Detects themes regardless of exact word choice
- Captures cultural concepts expressed differently
- Reduces false negatives from translation variations
- More balanced cross-linguistic comparisons

## Results Comparison

| Theme | Current | Semantic | Improvement |
|-------|---------|----------|-------------|
| Transportation | 34.0% | 65.0% | +31.0% |
| Animals | 39.0% | 58.0% | +19.0% |
| Money | 19.0% | 31.0% | +12.0% |
| Violence | 22.0% | 35.0% | +13.0% |

## Statistical Impact

- **Median improvement**: 28.0% → 46.5% (+18.5%)
- **Zero-inflation reduced**: More balanced distribution
- **Cross-linguistic validity**: Better comparisons with other languages

## Implementation Recommendation

1. **Immediate**: Apply semantic similarity to Basque analysis
2. **Validation**: Compare with native Basque speakers
3. **Extension**: Apply to all translated languages
4. **Research**: Publish methodology improvements

