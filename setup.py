"""
Setup script for AI Assistant Comparison Project
"""

from setuptools import setup, find_packages

setup(
    name="ollive-ai-comparison",
    version="1.0.0",
    description="AI Assistant Comparison Framework - OSS vs Frontier Models",
    author="Senior Backend and AI Developer",
    author_email="dev@ollive.ai",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",

        # API clients
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "google-generativeai>=0.3.0",
        "requests>=2.31.0",

        # Utilities
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",

        # Visualization and reporting
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",

        # Optional: For faster inference
        "accelerate>=0.24.0",

        # Development tools
        "pytest>=7.4.0",
        "black>=23.0.0",
    ],
    extras_require={
        "gpu": ["torch[cuda]"],
        "quantization": ["bitsandbytes>=0.41.0"],
        "security": ["guardrails-ai>=0.2.0"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ollive-eval=run_evaluation:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)