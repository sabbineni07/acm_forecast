# Multi-stage Dockerfile for ACM Forecast Framework
# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables for PySpark (use default-jdk path)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Stage 2: Dependencies stage
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-minimal.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application stage
FROM dependencies as app

# Copy application code
COPY . /app

# Install the package in editable mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Default command
CMD ["python", "-c", "from acm_forecast import __version__; print(f'ACM Forecast Framework v{__version__}')"]

# Stage 4: Development stage (extends app)
FROM app as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Stage 5: Jupyter/Notebook stage (for interactive development)
FROM dependencies as jupyter

# Install Jupyter and additional tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook

# Copy application code
COPY . /app

# Install the package
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

