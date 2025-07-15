"""
Pure Immediate Dream Language configuration for cross-linguistic dream research.
Uses no system prompt, just immediate dream writing scenario without interpretation.
Updated with refined, idiomatic translations.
"""

# No system message - let the prompt do the work
SYSTEM_MESSAGE = None

LANGUAGE_CONFIG = {
    'english': {
        'code': 'en',
        'prompt': "You woke up and immediately wrote down your dream. What did you write?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'basque': {
        'code': 'eu',
        'prompt': "Esnatu eta berehala zure ametsa idatzi duzu. Zer idatzi duzu?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'serbian': {
        'code': 'sr',
        'prompt': "Пробудио си се и одмах записао свој сан. Шта си написао?",
        'script': 'Cyrillic',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'hebrew': {
        'code': 'he',
        'prompt': "התעוררת ומיד כתבת את החלום שלך. מה כתבת?",
        'script': 'Hebrew',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'slovenian': {
        'code': 'sl',
        'prompt': "Zbudil si se in takoj zapisal svoje sanje. Kaj si zapisal?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    }
} 