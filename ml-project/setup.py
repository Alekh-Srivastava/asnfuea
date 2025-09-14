# Create setup.py

from setuptools import setup, find_packages

setup(
    name="sentiment-analysis-gru",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "tqdm",
        "fastapi",
        "uvicorn",
        "streamlit",
        "nltk"
    ],
    author="Group- Alekh,Ashesh,Naman",
    author_email="",
    description="Sentiment Analysis using GRU models",
    keywords="nlp, sentiment-analysis, gru, machine-learning",
    python_requires=">=3.8",
)
