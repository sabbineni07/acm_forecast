# Test Suite Documentation

This directory contains the comprehensive test suite for the ACM Forecast framework.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests for individual components
│   ├── test_config.py      # Configuration module tests
│   ├── test_performance_metrics.py  # Performance metrics tests
│   ├── test_plugin_factory.py  # Plugin factory tests
│   └── test_plugins_*.py   # Plugin-specific tests
├── integration/             # Integration tests for component interactions
│   ├── test_config_pipeline.py  # Config and pipeline integration tests
│   ├── test_data_source_delta.py  # Data source integration tests
│   ├── test_data_quality_delta.py  # Data quality integration tests
│   └── test_data_pipeline_delta.py  # Data pipeline integration tests
└── e2e/                     # End-to-end tests for full workflows
    ├── test_app_runner.py  # AppRunner E2E tests
    └── test_full_pipeline.py  # Full pipeline E2E tests
```

## Running Tests

### Using Makefile (Recommended)

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only

# Run with coverage
make test-cov          # Terminal coverage report
make test-cov-html     # HTML coverage report
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=acm_forecast --cov-report=html
```

### Run Tests by Type

```bash
# Unit tests only
pytest tests/unit -m unit
# Or use Makefile:
make test-unit

# Integration tests only
pytest tests/integration -m integration
# Or use Makefile:
make test-integration

# End-to-end tests only
pytest tests/e2e -m e2e
# Or use Makefile:
make test-e2e
```

### Run Specific Test Files

```bash
# Run config tests
pytest tests/unit/test_config.py

# Run plugin factory tests
pytest tests/unit/test_plugin_factory.py

# Run AppRunner tests
pytest tests/e2e/test_app_runner.py
```

### Run Tests Matching Patterns

```bash
# Run tests matching a pattern
pytest -k "config"

# Run tests excluding slow tests
pytest -m "not slow"
```

## Test Markers

Tests are marked with categories for easy filtering:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.requires_spark` - Tests requiring PySpark
- `@pytest.mark.requires_mlflow` - Tests requiring MLflow

## Coverage Goals

The project aims for:
- Minimum coverage: 70% (configured in `pyproject.toml`)
- Target coverage: 80%+
- Critical modules: 90%+

Current coverage can be viewed by running:
```bash
make test-cov-html
open htmlcov/index.html
```

## Writing Tests

### Unit Tests

Unit tests should:
- Test individual functions/classes in isolation
- Use mocks for external dependencies
- Be fast (< 1 second per test)
- Have clear, descriptive names

Example:
```python
@pytest.mark.unit
def test_data_config_required_fields():
    """Test that required fields are enforced"""
    config = DataConfig(
        delta_table_path="test.db.table",
        database_name="test_db",
        table_name="test_table",
    )
    assert config.delta_table_path == "test.db.table"
```

### Integration Tests

Integration tests should:
- Test interactions between components
- Use real (but minimal) data where possible
- Test realistic scenarios
- May take longer than unit tests

### End-to-End Tests

E2E tests should:
- Test complete workflows
- Use real configurations
- Verify end-to-end behavior
- May be marked as `@pytest.mark.slow`

## Fixtures

Shared fixtures are defined in `conftest.py`:
- `temp_config_file` - Temporary YAML config file
- `sample_app_config` - Sample AppConfig instance
- `sample_data_config` - Sample DataConfig instance
- `sample_model_config` - Sample ModelConfig instance
- And more...

## Test Summary

### Current Test Coverage

- **Unit Tests**: Configuration, performance metrics, plugin factory, plugins
- **Integration Tests**: Data source, data quality, data pipeline, config-pipeline integration
- **End-to-End Tests**: AppRunner, full pipeline workflows

### Test Markers

Tests are organized with markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_spark` - Tests needing PySpark
- `@pytest.mark.requires_mlflow` - Tests needing MLflow

## Continuous Integration

Tests should pass in CI/CD pipelines:
- All unit tests should pass
- Integration tests should pass (may require test databases/services)
- E2E tests may be run conditionally

### Docker Testing

Tests can also be run in Docker:

```bash
# Run all tests in Docker
make docker-test

# Run specific test types in Docker
make docker-test-unit
make docker-test-integration
make docker-test-e2e
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Install package in editable mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies

Install test dependencies:
```bash
pip install -e ".[dev]"
```

### Coverage Not Working

Ensure pytest-cov is installed:
```bash
pip install pytest-cov
```

