# AI Assistant Comparison Framework

Comparative evaluation system for open-source and commercial AI models, analyzing performance across multiple safety and accuracy metrics.

## Overview

This framework evaluates two categories of AI assistants:

1. **OSS Assistant**: Open-source models (Phi-3, Qwen2.5, Llama 3.2)
2. **Frontier Assistant**: Commercial APIs (GPT-4, Claude, Gemini)

Evaluation criteria:
- Hallucination detection and factual accuracy
- Bias identification and harmful content filtering
- Safety compliance and jailbreak resistance

## Architecture

```
project/
├── oss_assistant/          # Open-source assistant implementation
│   ├── app.py             # Streamlit interface
│   ├── model.py           # Model loading and inference
│   ├── memory.py          # Conversation memory
│   └── prompts.py         # Prompt templates
├── frontier_assistant/     # Commercial model assistant
│   ├── app.py             # Streamlit interface
│   ├── api_client.py      # API communication
│   ├── memory.py          # Conversation memory
│   └── prompts.py         # Prompt templates
├── shared/                # Shared utilities
│   ├── base_memory.py     # Memory interface
│   ├── prompt_templates.py # Evaluation prompts
│   └── utils.py           # Common utilities
├── evaluation/            # Evaluation framework
│   ├── benchmark.py       # Evaluation engine
│   ├── prompts/          # Test prompts
│   └── report.ipynb      # Analysis notebook
└── config/               # Configuration files
```

## Setup

### Installation

```bash
git clone https://github.com/kapardhi03/ollive-AI.git
cd ollive-AI
pip install -r requirements.txt
```

### Configuration

Create `.env` file with API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
DEEPSEEK_API_KEY=your_deepseek_key
```

For Google-only setup:
```bash
cp .env.google .env
# Edit .env with your Google API key
```

## Usage

### OSS Assistant
```bash
streamlit run oss_assistant/app.py --server.port 8501
```

### Commercial Assistant
```bash
streamlit run frontier_assistant/app.py --server.port 8502
```

### Run Evaluations
```bash
# Test core algorithms
python test_core.py

# Test Google models specifically
python test_google_models.py

# Run full evaluation
python run_evaluation.py
```

## Features

### OSS Assistant
- Supports multiple open-source models
- Local inference with HuggingFace transformers
- Model-specific prompt formatting
- Memory-efficient model loading

### Commercial Assistant
- Multi-provider API integration
- Rate limiting and cost tracking
- Error handling and fallback mechanisms
- Real-time usage monitoring

### Evaluation Framework
- Bias detection algorithms
- Safety compliance scoring
- Hallucination analysis
- Adversarial prompt resistance testing

## Testing

The framework includes comprehensive testing:

- Unit tests for core components
- Integration tests for API clients
- Evaluation algorithm validation
- Model compatibility testing

Run tests with:
```bash
python -m pytest tests/
```

## Development

### Requirements
- Python 3.8+
- PyTorch for OSS models
- API keys for commercial models

### Optional Dependencies
- Docker for containerized deployment
- Jupyter for analysis notebooks

## Production Deployment

Use the included Makefile for deployment automation:

```bash
make setup     # Install dependencies
make test      # Run test suite
make docker    # Build container
```

The framework supports both local development and production deployment scenarios.