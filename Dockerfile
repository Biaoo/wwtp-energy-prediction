# Use Python 3.10 slim image as base
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY data/ ./data/

# Install UV package manager
RUN pip install --no-cache-dir uv

# Install project dependencies using UV
RUN uv sync --frozen

# Expose port for Gradio
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the Gradio app
CMD ["uv", "run", "python", "src/gradio_app.py"]