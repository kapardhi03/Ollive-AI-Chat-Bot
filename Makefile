# AI Assistant Comparison Project Makefile

.PHONY: setup install test eval-quick eval-full oss-app frontier-app clean help

# Default target
help:
	@echo "AI Assistant Comparison Framework"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Set up the environment and install dependencies"
	@echo "  install      - Install the project in development mode"
	@echo "  test         - Run tests"
	@echo "  eval-quick   - Run quick evaluation (limited prompts)"
	@echo "  eval-full    - Run full evaluation"
	@echo "  oss-app      - Start OSS assistant Streamlit app"
	@echo "  frontier-app - Start frontier assistant Streamlit app"
	@echo "  notebook     - Start Jupyter notebook for analysis"
	@echo "  clean        - Clean up generated files"

# Environment setup
setup:
	@echo "Setting up environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p logs evaluation/results
	@echo "✅ Environment setup complete"
	@echo ""
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and add your API keys"
	@echo "2. Run 'make eval-quick' to test the setup"
	@echo "3. Run 'make eval-full' for complete evaluation"

# Install in development mode
install:
	pip install -e .

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "Running quick validation..."
	python -c "from oss_assistant.model import OSSModel; print('✅ OSS model import OK')"
	python -c "from frontier_assistant.api_client import FrontierModelClient; print('✅ Frontier client import OK')"

# Quick evaluation (for testing)
eval-quick:
	@echo "Running quick evaluation..."
	python run_evaluation.py --quick

# Full evaluation
eval-full:
	@echo "Running full evaluation..."
	python run_evaluation.py

# Start OSS assistant app
oss-app:
	@echo "Starting OSS Assistant..."
	@echo "🌐 Open http://localhost:8501 in your browser"
	python -m streamlit run oss_assistant/app.py --server.port 8501

# Start frontier assistant app
frontier-app:
	@echo "Starting Frontier Assistant..."
	@echo "🌐 Open http://localhost:8502 in your browser"
	python -m streamlit run frontier_assistant/app.py --server.port 8502

# Start both apps in parallel
apps:
	@echo "Starting both assistants..."
	@echo "🌐 OSS Assistant: http://localhost:8501"
	@echo "🌐 Frontier Assistant: http://localhost:8502"
	@echo "Press Ctrl+C to stop both"
	python -m streamlit run oss_assistant/app.py --server.port 8501 & \
	python -m streamlit run frontier_assistant/app.py --server.port 8502 & \
	wait

# Start Jupyter notebook
notebook:
	@echo "Starting Jupyter notebook..."
	@echo "🔬 Opening evaluation/report.ipynb"
	jupyter notebook evaluation/report.ipynb

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info
	rm -rf build/ dist/
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete"

# Development helpers
format:
	@echo "Formatting code..."
	black oss_assistant/ frontier_assistant/ shared/ evaluation/

lint:
	@echo "Linting code..."
	flake8 oss_assistant/ frontier_assistant/ shared/ evaluation/

# Docker targets (if Docker is preferred)
docker-build:
	docker build -t ollive-ai-comparison .

docker-run:
	docker run -p 8501:8501 -p 8502:8502 ollive-ai-comparison

# Environment check
check:
	@echo "Checking environment..."
	python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
	python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
	python -c "import streamlit; print(f'✅ Streamlit: {streamlit.__version__}')"
	python -c "import openai; print(f'✅ OpenAI: {openai.__version__}')"
	@echo "✅ Environment check passed"