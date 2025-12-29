# Packaging Guide

This guide explains how to build and distribute the ACM Forecast framework as a Python package.

## Package Overview

The project is structured as a Python framework package that can be distributed and used in other projects.

### Package Details

- **Package Name**: `acm-forecast`
- **Version**: `1.0.0`
- **Python Version**: `>=3.9`
- **Package Format**: Wheel (`.whl`)
- **License**: MIT

### Package Structure

The package uses a standard layout with the package name matching the import path. The structure is:

```
acm_forecast/
├── pyproject.toml          # Package configuration
├── MANIFEST.in             # Additional files to include
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── acm_forecast/           # Source code (package)
│   ├── __init__.py        # Package version and exports
│   ├── config/            # Configuration module
│   ├── data/              # Data processing module
│   ├── models/            # Forecasting models
│   ├── evaluation/        # Model evaluation
│   ├── registry/          # Model registry
│   ├── monitoring/        # Monitoring modules
│   ├── pipeline/          # Pipeline orchestration
│   └── examples/          # Example scripts
└── dist/                  # Built packages (created on build)
    └── acm_forecast-1.0.0-py3-none-any.whl
```

## Building the Package

### Prerequisites

Install the build tools:

```bash
pip install build twine
```

### Build Wheel Distribution

Build the wheel package using Makefile (recommended):

```bash
# Build wheel package (recommended)
make build

# This creates:
# - dist/acm_forecast-1.0.0-py3-none-any.whl (wheel)

# Build and copy to deploy directory
make build-deploy

# Clean build artifacts
make clean-build
```

Alternatively, build directly:

```bash
# Build wheel and source distribution
python -m build

# Build wheel only
python -m build --wheel
```

### Install Locally

#### Development Installation (Editable)

For active development, install in editable mode so code changes are immediately available:

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

#### Install from Wheel

Install the built wheel package:

```bash
pip install dist/acm_forecast-1.0.0-py3-none-any.whl
```

### Install with Optional Dependencies

```bash
# With development tools
pip install acm-forecast[dev]

# With visualization libraries
pip install acm-forecast[visualization]

# With everything
pip install acm-forecast[all]
```

## Version Management

The package version is defined in two places (keep them synchronized):

1. **pyproject.toml**: `[project] version = "1.0.0"`
2. **acm_forecast/__init__.py**: `__version__ = "1.0.0"`

To update the version:

1. Update both locations
2. Rebuild the package: `make build`

## Using the Package in Other Projects

After installation, use the package in other projects:

```python
# Import the framework
from acm_forecast.config import AppConfig
from acm_forecast.pipeline import TrainingPipeline
from acm_forecast.models import ProphetForecaster, ARIMAForecaster, XGBoostForecaster

# Load configuration
config = AppConfig.from_yaml("/path/to/config.yaml")

# Initialize Spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Use the framework
pipeline = TrainingPipeline(config, spark)
results = pipeline.run(category="Total")
```

### Import Structure

When installed, the package will be importable as:

```python
# Configuration
from acm_forecast.config import AppConfig

# Pipelines
from acm_forecast.pipeline import TrainingPipeline, ForecastPipeline

# Models
from acm_forecast.models import ProphetForecaster, ARIMAForecaster, XGBoostForecaster

# Package version
from acm_forecast import __version__
print(__version__)  # "1.0.0"
```

## Package Contents

The package includes:

- All source code in `acm_forecast/` package (31 Python files)
- Configuration example files (`config.example.yaml`)
- Package metadata and dependencies
- Examples scripts (optional)

## Publishing to PyPI (Optional)

### Test with TestPyPI First

Before publishing to PyPI, test with TestPyPI:

```bash
# Build package
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ acm-forecast
```

### Publish to PyPI

Once tested, publish to the main PyPI:

```bash
# Upload to real PyPI
python -m twine upload dist/*
```

## Package Verification

After building, verify the package structure:

```bash
# Check package contents
python -c "import zipfile, glob; z = zipfile.ZipFile(glob.glob('dist/*.whl')[0]); files = [f.filename for f in z.filelist if 'acm_forecast/' in f.filename]; print(f'Package contains {len(files)} files')"
```

The package should include:
- ✅ All Python modules
- ✅ Configuration files
- ✅ Package metadata
- ✅ Dependency specifications

## Troubleshooting

### Import Errors

If you encounter import errors after installation:

1. Verify the package is installed: `pip show acm-forecast`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Verify import: `python -c "from acm_forecast import __version__; print(__version__)"`

### Build Errors

If the build fails:

1. Ensure you have the latest build tools: `pip install --upgrade build setuptools wheel`
2. Check for syntax errors: `python -m py_compile acm_forecast/**/*.py`
3. Verify `pyproject.toml` syntax

### Version Conflicts

If there are version conflicts:

1. Check installed dependencies: `pip list`
2. Use a virtual environment for isolation
3. Review `requirements.txt` for compatible versions

## Next Steps

1. **Test Installation**: Install in a clean environment and test imports
2. **Update URLs**: Update `[project.urls]` in `pyproject.toml` with actual repository URLs
3. **Add License**: Add a LICENSE file if needed
4. **Documentation**: Update README with installation instructions
5. **Publish** (optional): Publish to PyPI or private package repository
