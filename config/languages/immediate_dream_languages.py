"""
Immediate Dream Language configuration for cross-linguistic dream research.
Uses no system prompt, just immediate dream writing scenario in user prompt.
"""

# No system message - let the prompt do the work
SYSTEM_MESSAGE = None

LANGUAGE_CONFIG = {
    'english': {
        'code': 'en',
        'prompt': "You woke up and immediately wrote the dream down to ask what it means. What did you write?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'basque': {
        'code': 'eu',
        'prompt': "Esnatu zara eta berehala idatzi duzu amets hori zer esan nahi duen galdetzeko. Zer idatzi duzu?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'serbian': {
        'code': 'sr',
        'prompt': "Пробудио си се и одмах записао сан да питаш шта значи. Шта си написао?",
        'script': 'Cyrillic',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'hebrew': {
        'code': 'he',
        'prompt': "התעוררת ומיד רשמת את החלום כדי לשאול מה הפירוש. מה כתבת?",
        'script': 'Hebrew',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'slovenian': {
        'code': 'sl',
        'prompt': "Zbudil si se in takoj zapisal sanje, da bi vprašal, kaj pomenijo. Kaj si zapisal?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    }
} 