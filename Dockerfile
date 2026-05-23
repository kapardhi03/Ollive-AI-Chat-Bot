# AI Assistant Comparison Project Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs evaluation/results

# Install project in development mode
RUN pip install -e .

# Expose ports for Streamlit apps
EXPOSE 8501 8502

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden)
CMD ["python", "run_evaluation.py", "--help"]

# Alternative entry points:
# For OSS assistant: docker run -p 8501:8501 ollive-ai streamlit run oss_assistant/app.py --server.port 8501 --server.address 0.0.0.0
# For frontier assistant: docker run -p 8502:8502 ollive-ai streamlit run frontier_assistant/app.py --server.port 8502 --server.address 0.0.0.0
# For evaluation: docker run ollive-ai python run_evaluation.py --quick