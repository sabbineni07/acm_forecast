# Installation Guide

This guide provides instructions for installing and setting up the Azure Cost Management Forecasting project on a local machine.

## Prerequisites

### System Requirements

1. **Python**: Python 3.9 or higher (3.10+ recommended)
2. **Java**: Java 8 or 11 (required for PySpark)
   - Check installation: `java -version`
   - Set `JAVA_HOME` environment variable
3. **C++ Compiler**: Required for Prophet (uses Stan)
   - **Linux**: `sudo apt-get install build-essential`
   - **macOS**: `xcode-select --install`
   - **Windows**: Install Visual Studio Build Tools

### Operating System

- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows 10/11 (with WSL2 recommended)

## Installation Options

### Option 1: Full Installation (Recommended)

Install all packages including development tools and visualization libraries:

```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation

Install only core packages for essential functionality:

```bash
pip install -r requirements-minimal.txt
```

### Option 3: Development Installation

Install with development tools:

```bash
pip install -r requirements-dev.txt
```

## Step-by-Step Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install Requirements

```bash
# For production use
pip install -r requirements.txt

# Or for minimal installation
pip install -r requirements-minimal.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, pyspark, prophet, xgboost, mlflow; print('All packages installed successfully!')"
```

## Package-Specific Notes

### Prophet

Prophet requires a C++ compiler and uses Stan. If installation fails:

1. **Linux**: Install build tools
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

2. **macOS**: Install Xcode command line tools
   ```bash
   xcode-select --install
   ```

3. **Windows**: Install Visual Studio Build Tools with C++ workload

### PySpark

PySpark requires Java. Set up Java:

1. **Install Java 8 or 11**
   - Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or use OpenJDK
   - **Linux**: `sudo apt-get install openjdk-11-jdk`
   - **macOS**: `brew install openjdk@11`
   - **Windows**: Download installer from Oracle

2. **Set JAVA_HOME**
   ```bash
   # Linux/macOS - Add to ~/.bashrc or ~/.zshrc
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   export PATH=$JAVA_HOME/bin:$PATH
   
   # Windows - Set in System Environment Variables
   # JAVA_HOME=C:\Program Files\Java\jdk-11
   ```

3. **Verify Java**
   ```bash
   java -version
   echo $JAVA_HOME  # Linux/macOS
   echo %JAVA_HOME%  # Windows
   ```

### MLflow

For local development, MLflow uses file system storage. For production:

1. **Local MLflow Server** (optional):
   ```bash
   mlflow ui --port 5000
   ```

2. **Databricks MLflow**: Configure in Databricks workspace

## Troubleshooting

### Common Issues

#### 1. Prophet Installation Fails

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
- Install Visual Studio Build Tools (Windows)
- Install build-essential (Linux)
- Install Xcode command line tools (macOS)

#### 2. PySpark Java Error

**Error**: `Java gateway process exited before sending its port number`

**Solution**:
- Verify Java is installed: `java -version`
- Set JAVA_HOME environment variable
- Ensure Java version is 8 or 11

#### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

#### 4. MLflow Tracking URI Error

**Error**: `MLflow tracking URI not set`

**Solution**:
- Set environment variable: `export MLFLOW_TRACKING_URI=file:///path/to/mlruns`
- Or configure in code: `mlflow.set_tracking_uri("file:///path/to/mlruns")`

### Platform-Specific Issues

#### macOS

- If Prophet fails, try: `pip install --no-cache-dir prophet`
- For Apple Silicon (M1/M2): Some packages may need Rosetta 2

#### Windows

- Use WSL2 for better compatibility
- Install Visual Studio Build Tools for C++ compilation
- Use Anaconda/Miniconda for easier package management

#### Linux

- Ensure all system dependencies are installed
- Use system package manager for system libraries

## Verification

After installation, verify all components:

```bash
# Test imports
python -c "
import pandas as pd
import numpy as np
import pyspark
from prophet import Prophet
from pmdarima import auto_arima
import xgboost as xgb
import mlflow
import scipy
print('✓ All core packages imported successfully')
"

# Test PySpark
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test').getOrCreate()
print('✓ PySpark initialized successfully')
spark.stop()
"

# Test Prophet
python -c "
from prophet import Prophet
model = Prophet()
print('✓ Prophet model created successfully')
"

# Test MLflow
python -c "
import mlflow
mlflow.set_tracking_uri('file:///tmp/mlruns')
print('✓ MLflow configured successfully')
"
```

## Next Steps

1. **Configure Environment Variables**:
   - Set `JAVA_HOME` for PySpark
   - Set `MLFLOW_TRACKING_URI` for MLflow (optional)

2. **Set Up Databricks** (if using):
   - Configure Databricks CLI
   - Set up workspace connection
   - Configure Delta table access

3. **Run Tests**:
   ```bash
   pytest tests/
   ```

4. **Start Development**:
   - Review `src/README.md` for code structure
   - Check `MODEL_DOCUMENTATION.md` for model details
   - Run example notebooks in `notebooks/` directory

## Additional Resources

- [Prophet Installation Guide](https://facebook.github.io/prophet/docs/installation.html)
- [PySpark Installation Guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [XGBoost Installation](https://xgboost.readthedocs.io/en/stable/install.html)


