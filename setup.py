from setuptools import setup, find_packages

setup(
    name="ai_text_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "pyyaml",
        "kaggle",
    ],
    entry_points={
        "console_scripts": [
            "ai-detector=ai_text_detector.cli:main",
        ],
    },
    author="Your Name",
    description="A learning project for detecting AI-generated text with CLI + YAML + GPU auto-detect.",
    license="MIT",
    python_requires=">=3.8",
)
