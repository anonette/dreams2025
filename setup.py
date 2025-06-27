"""
Setup script for the dream research system.
"""

from setuptools import setup, find_packages

setup(
    name="dream-research-system",
    version="1.0.0",
    description="Cross-linguistic LLM dream narrative research system",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "spacy>=3.6.0",
        "googletrans==4.0.0rc1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dream-research=main:main",
        ],
    },
) 