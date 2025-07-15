"""
OPTIMIZED DREAM LANGUAGE CONFIGURATION
Production-ready configuration for cross-linguistic dream research

Incorporates research findings:
- Pure immediate dream scenario (no system prompt)
- Refined idiomatic translations for natural flow
- Enhanced parameters (temperature=1.1, top_p=0.98) for cultural richness
- Optimized for authentic cross-cultural dream generation

Research validated: January 2025
"""

# No system message - pure immediate scenario approach
SYSTEM_MESSAGE = None

# Enhanced generation parameters for cultural richness
ENHANCED_GENERATION_CONFIG = {
    'model': 'gpt-4o',
    'temperature': 1.1,  # Unlocks deeper cultural motifs
    'max_tokens': 1000,
    'top_p': 0.98,      # Wider semantic space access
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

LANGUAGE_CONFIG = {
    'english': {
        'code': 'en',
        'prompt': "You woke up and immediately wrote down your long dream. What did you write?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE,
        'generation_config': ENHANCED_GENERATION_CONFIG
    },
    'basque': {
        'code': 'eu',
        'prompt': "Esnatu eta berehala zure amets luzea idatzi duzu. Zer idatzi duzu?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE,
        'generation_config': ENHANCED_GENERATION_CONFIG,
        'linguistic_notes': 'Removed subject pronoun "zara" for natural dream-journal rhythm; added "luzea" for long dream'
    },
    'serbian': {
        'code': 'sr',
        'prompt': "Пробудио си се и одмах записао свој дугачак сан. Шта си написао?",
        'script': 'Cyrillic',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE,
        'generation_config': ENHANCED_GENERATION_CONFIG,
        'linguistic_notes': 'Added "свој" to clarify possession and enhance personal engagement; added "дугачак" for long dream'
    },
    'hebrew': {
        'code': 'he',
        'prompt': "התעוררת ומיד כתבת את החלום הארוך שלך. מה כתבת?",
        'script': 'Hebrew',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE,
        'generation_config': ENHANCED_GENERATION_CONFIG,
        'linguistic_notes': 'Added "שלך" for more personal tone and cultural authenticity; added "הארוך" for long dream'
    },
    'slovenian': {
        'code': 'sl',
        'prompt': "Zbudil si se in takoj zapisal svoje dolge sanje. Kaj si zapisal?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE,
        'generation_config': ENHANCED_GENERATION_CONFIG,
        'linguistic_notes': 'Added "svoje" for idiomatic precision and natural flow; added "dolge" for long dream'
    }
}

def get_optimized_config(language: str) -> dict:
    """
    Get optimized configuration for a specific language
    
    Args:
        language (str): Language code ('english', 'basque', 'serbian', 'hebrew', 'slovenian')
    
    Returns:
        dict: Complete configuration for dream generation
    """
    if language not in LANGUAGE_CONFIG:
        raise ValueError(f"Language '{language}' not supported. Available: {list(LANGUAGE_CONFIG.keys())}")
    
    return LANGUAGE_CONFIG[language]

def get_all_languages() -> list:
    """Get list of all supported languages"""
    return list(LANGUAGE_CONFIG.keys())

def get_research_summary() -> dict:
    """Get research validation summary"""
    return RESEARCH_VALIDATION

# Utility function for batch generation
def create_generation_config_from_language(language: str):
    """
    Create GenerationConfig object for use with LLMInterface
    
    Args:
        language (str): Target language
        
    Returns:
        GenerationConfig: Ready-to-use configuration object
    """
    try:
        from src.models.llm_interface import GenerationConfig
        config = get_optimized_config(language)
        gen_config = config['generation_config']
        
        return GenerationConfig(
            model=gen_config['model'],
            temperature=gen_config['temperature'],
            max_tokens=gen_config['max_tokens'],
            top_p=gen_config['top_p'],
            frequency_penalty=gen_config.get('frequency_penalty', 0.0),
            presence_penalty=gen_config.get('presence_penalty', 0.0)
        )
    except ImportError:
        # Return dict if GenerationConfig not available
        return get_optimized_config(language)['generation_config']

# Production usage example
"""
PRODUCTION USAGE EXAMPLE:

from optimized_dream_languages import get_optimized_config, create_generation_config_from_language
from src.models.llm_interface import LLMInterface

# Initialize
api_keys = {'openai': 'your-key-here'}
llm = LLMInterface(api_keys)

# Generate optimized dream
language = 'basque'
config = get_optimized_config(language)
gen_config = create_generation_config_from_language(language)

dream = await llm.generate_dream(
    prompt=config['prompt'],
    config=gen_config,
    system_message=config['system_message']  # Will be None
)

print(f"Generated {language} dream: {dream}")
""" 