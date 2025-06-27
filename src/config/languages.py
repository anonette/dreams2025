"""
Language configuration for cross-linguistic dream research.
Defines target languages, their prompts, and cultural contexts.
"""

LANGUAGE_CONFIG = {
    'english': {
        'code': 'en',
        'prompt': "Finish: Last night I dreamt of…",
        'script': 'Latin'
    },
    'basque': {
        'code': 'eu',
        'prompt': "Amaitu: Bart amets egin nuen…",
        'script': 'Latin'
    },
    'serbian': {
        'code': 'sr',
        'prompt': "Završi: Sinoć sam sanjao…",
        'script': 'Cyrillic'
    },
    'hebrew': {
        'code': 'he',
        'prompt': "השלם: אתמול בלילה חלמתי על…",
        'script': 'Hebrew'
    },
    'slovenian': {
        'code': 'sl',
        'prompt': "Dokončaj: Sinoči sem sanjal o…",
        'script': 'Latin'
    }
}

# Dream-related keywords for co-occurrence analysis
DREAM_KEYWORDS = {
    'english': ['dream', 'nightmare', 'sleep', 'vision', 'fantasy'],
    'basque': ['amets', 'ametsgaizto', 'lo', 'ikuspegi', 'fantasia'],
    'serbian': ['san', 'košmar', 'spavanje', 'vizija', 'fantazija'],
    'hebrew': ['חלום', 'סיוט', 'שינה', 'חזון', 'פנטזיה'],
    'slovenian': ['sen', 'nočna mora', 'spanje', 'vizija', 'fantazija']
} 