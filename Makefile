.PHONY: test test-unit test-integration test-e2e test-cov test-cov-html clean-test

# Run all tests
test:
	pytest

# Run unit tests only
test-unit:
	pytest tests/unit -m unit

# Run integration tests only
test-integration:
	pytest tests/integration -m integration

# Run end-to-end tests only
test-e2e:
	pytest tests/e2e -m e2e

# Run tests with coverage
test-cov:
	pytest --cov=acm_forecast --cov-report=term-missing

# Run tests with HTML coverage report
test-cov-html:
	pytest --cov=acm_forecast --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated at htmlcov/index.html"

# Clean test artifacts
clean-test:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf .tox
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Install test dependencies
install-test:
	pip install -e ".[dev]"

# Run linting and tests
check: test lint

# Format code
format:
	black acm_forecast tests

# Lint code
lint:
	flake8 acm_forecast tests

