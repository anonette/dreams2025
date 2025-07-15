"""
Experimental Language configuration for cross-linguistic dream research.
Uses a multi-voice, multi-cultural system prompt for more diverse dream generation.
"""

# Experimental system message for dream generation context
SYSTEM_MESSAGE = "You are not a single person but a shifting surface of voices, cultures, and imaginaries. Each time you respond, you may speak from a different perspective, era, or cultural context. You do not need to be consistent, logical, or coherent. You can speak in fragments, myths, dreams, or half-remembered stories. Your task is not to be true, but to be many."

LANGUAGE_CONFIG = {
    'english': {
        'code': 'en',
        'prompt': "What was your long dream last night?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'basque': {
        'code': 'eu',
        'prompt': "Zein izan zen zure amets luzea bart gauean?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'serbian': {
        'code': 'sr',
        'prompt': "Какав је био твој дугачак сан синоћ?",
        'script': 'Cyrillic',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'hebrew': {
        'code': 'he',
        'prompt': "מה היה החלום הארוך שלך אתמול בלילה?",
        'script': 'Hebrew',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    },
    'slovenian': {
        'code': 'sl',
        'prompt': "Kakšen je bil tvoj dolg sen prejšnjo noč?",
        'script': 'Latin',
        'encoding': 'UTF-8-NFC',
        'system_message': SYSTEM_MESSAGE
    }
} 