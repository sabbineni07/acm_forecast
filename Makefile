.PHONY: help test test-unit test-integration test-e2e test-cov test-cov-html clean-test
.PHONY: format lint check install-test
.PHONY: docker-build docker-build-no-cache docker-build-app docker-build-dev docker-build-jupyter
.PHONY: docker-up docker-down docker-restart docker-ps docker-logs docker-logs-follow
.PHONY: docker-exec-app docker-exec-dev docker-shell-app docker-shell-dev
.PHONY: docker-run-training docker-run-forecast docker-run-pipeline
.PHONY: docker-test docker-test-unit docker-test-integration docker-test-e2e docker-test-cov
.PHONY: docker-lint docker-format docker-typecheck
.PHONY: docker-mlflow docker-jupyter docker-azurite
.PHONY: docker-clean docker-clean-all docker-prune
.PHONY: docker-azure-up docker-azure-down

# Default target
help:
	@echo "ACM Forecast Framework - Makefile Commands"
	@echo "=========================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test                  - Run all tests"
	@echo "  make test-unit             - Run unit tests only"
	@echo "  make test-integration      - Run integration tests only"
	@echo "  make test-e2e              - Run end-to-end tests only"
	@echo "  make test-cov              - Run tests with coverage (terminal)"
	@echo "  make test-cov-html         - Run tests with coverage (HTML report)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format                - Format code with black"
	@echo "  make lint                  - Lint code with flake8"
	@echo "  make check                 - Run linting and tests"
	@echo ""
	@echo "Docker - Build:"
	@echo "  make docker-build          - Build all Docker images (dev, default)"
	@echo "  make docker-build-dev      - Build development images"
	@echo "  make docker-build-prod     - Build production images"
	@echo "  make docker-build-no-cache - Build all images without cache"
	@echo "  make docker-build-app      - Build application image (prod)"
	@echo "  make docker-build-dev-image - Build development image"
	@echo "  make docker-build-jupyter  - Build Jupyter image"
	@echo ""
	@echo "Docker - Services:"
	@echo "  make docker-up             - Start all services (dev, default)"
	@echo "  make docker-up-dev         - Start development services"
	@echo "  make docker-up-prod        - Start production services"
	@echo "  make docker-down           - Stop all services (dev)"
	@echo "  make docker-down-dev       - Stop development services"
	@echo "  make docker-down-prod      - Stop production services"
	@echo "  make docker-restart        - Restart all services (dev)"
	@echo "  make docker-restart-dev    - Restart development services"
	@echo "  make docker-restart-prod   - Restart production services"
	@echo "  make docker-ps             - Show running containers"
	@echo "  make docker-logs           - Show logs from all services"
	@echo "  make docker-logs-follow    - Follow logs from all services"
	@echo ""
	@echo "Docker - Execution:"
	@echo "  make docker-exec-app       - Execute command in app container (use CMD='command')"
	@echo "  make docker-shell-app      - Open bash shell in app container"
	@echo "  make docker-shell-dev      - Open bash shell in dev container"
	@echo "  make docker-run-training   - Run training pipeline"
	@echo "  make docker-run-forecast   - Run forecast pipeline"
	@echo "  make docker-run-pipeline   - Run complete pipeline"
	@echo ""
	@echo "Docker - Testing:"
	@echo "  make docker-test           - Run all tests in Docker (includes Java)"
	@echo "  make docker-test-unit      - Run unit tests in Docker"
	@echo "  make docker-test-integration - Run integration tests in Docker (Delta tables)"
	@echo "  make docker-test-e2e       - Run e2e tests in Docker"
	@echo "  make docker-test-cov       - Run tests with coverage in Docker (terminal)"
	@echo "  make docker-test-cov-html  - Run tests with coverage in Docker (HTML report)"
	@echo ""
	@echo "Docker - Code Quality:"
	@echo "  make docker-format         - Format code in Docker"
	@echo "  make docker-lint           - Lint code in Docker"
	@echo "  make docker-typecheck      - Type check with mypy in Docker"
	@echo ""
	@echo "Docker - Individual Services:"
	@echo "  make docker-mlflow         - Start MLflow service"
	@echo "  make docker-jupyter        - Start Jupyter Lab"
	@echo "  make docker-azurite        - Start Azurite (Azure Storage Emulator)"
	@echo ""
	@echo "Docker - Azure ADLS:"
	@echo "  make docker-azure-up       - Start services with Azure ADLS config"
	@echo "  make docker-azure-down     - Stop services with Azure ADLS config"
	@echo ""
	@echo "Docker - Cleanup:"
	@echo "  make docker-clean          - Stop and remove containers (dev)"
	@echo "  make docker-clean-dev      - Stop and remove dev containers"
	@echo "  make docker-clean-prod     - Stop and remove prod containers"
	@echo "  make docker-clean-all      - Remove containers, volumes, images (dev)"
	@echo "  make docker-clean-all-dev  - Remove dev containers, volumes, images"
	@echo "  make docker-clean-all-prod - Remove prod containers, volumes, images"
	@echo "  make docker-prune          - Prune Docker system (clean everything)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-test            - Clean test artifacts"

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
	pytest -x -vv --disable-warnings -p no:warnings

# Run unit tests only
test-unit:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
	pytest -x -vv --disable-warnings -p no:warnings tests/unit -m unit

# Run integration tests only
test-integration:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
	if [ -z "$$JAVA_HOME" ]; then \
		echo "Error: JAVA_HOME could not be detected. Java is required for integration tests."; \
		echo "Please set JAVA_HOME or install Java (Java 8 or 11 required for PySpark)."; \
		exit 1; \
	fi; \
	echo "Using JAVA_HOME: $$JAVA_HOME"; \
	pytest -x -vv --disable-warnings tests/integration -m integration

# Run end-to-end tests only
test-e2e:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
	pytest tests/e2e -m e2e

# Run tests with coverage
test-cov:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
	pytest --cov=acm_forecast --cov-report=term-missing

# Run tests with HTML coverage report
test-cov-html:
	@export JAVA_HOME=$${JAVA_HOME:-$$(/usr/libexec/java_home 2>/dev/null || echo "")}; \
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

# ============================================================================
# Docker - Build
# ============================================================================

# Build all Docker images (development by default)
docker-build:
	docker compose build

# Build all images without cache
docker-build-no-cache:
	docker compose build --no-cache

# Build production images
docker-build-prod:
	docker compose -f docker-compose.prod.yml build

# Build development images
docker-build-dev:
	docker compose -f docker-compose.dev.yml build

# Build application image (production)
docker-build-app:
	docker compose -f docker-compose.prod.yml build acm-forecast-app

# Build development image
docker-build-dev-image:
	docker compose -f docker-compose.dev.yml build acm-forecast-dev

# Build Jupyter image
docker-build-jupyter:
	docker compose -f docker-compose.dev.yml build jupyter

# ============================================================================
# Docker - Services Management
# ============================================================================

# Start all services (development by default)
docker-up:
	docker compose up -d

# Start development services
docker-up-dev:
	docker compose -f docker-compose.dev.yml up -d

# Start production services
docker-up-prod:
	docker compose -f docker-compose.prod.yml up -d

# Stop all services (development)
docker-down:
	docker compose down

# Stop development services
docker-down-dev:
	docker compose -f docker-compose.dev.yml down

# Stop production services
docker-down-prod:
	docker compose -f docker-compose.prod.yml down

# Restart all services (development)
docker-restart:
	docker compose restart

# Restart development services
docker-restart-dev:
	docker compose -f docker-compose.dev.yml restart

# Restart production services
docker-restart-prod:
	docker compose -f docker-compose.prod.yml restart

# Show running containers
docker-ps:
	docker compose ps

# Show logs from all services
docker-logs:
	docker compose logs

# Follow logs from all services
docker-logs-follow:
	docker compose logs -f

# ============================================================================
# Docker - Execution
# ============================================================================

# Execute command in app container (usage: make docker-exec-app CMD="python script.py")
docker-exec-app:
	@if [ -z "$(CMD)" ]; then \
		echo "Usage: make docker-exec-app CMD='your command here'"; \
		exit 1; \
	fi
	docker compose -f docker-compose.prod.yml exec acm-forecast-app $(CMD)

# Open bash shell in app container (production)
docker-shell-app:
	docker compose -f docker-compose.prod.yml exec acm-forecast-app bash

# Open bash shell in dev container
docker-shell-dev:
	docker compose exec acm-forecast-dev bash

# Run training pipeline (production)
docker-run-training:
	docker compose -f docker-compose.prod.yml run --rm acm-forecast-app python acm_forecast/examples/run_training.py

# Run forecast pipeline (production)
docker-run-forecast:
	docker compose -f docker-compose.prod.yml run --rm acm-forecast-app python acm_forecast/examples/run_forecast.py

# Run complete pipeline (production)
docker-run-pipeline:
	docker compose -f docker-compose.prod.yml run --rm acm-forecast-app python acm_forecast/examples/run_complete_pipeline.py

# ============================================================================
# Docker - Testing
# ============================================================================

# Run all tests in Docker (uses persistent dev container)
docker-test:
	@./scripts/run_tests_in_docker.sh tests/ -v

# Run unit tests in Docker
docker-test-unit:
	@./scripts/run_tests_in_docker.sh tests/unit/ -v

# Run integration tests in Docker (includes Delta table tests with Java)
docker-test-integration:
	@./scripts/run_tests_in_docker.sh tests/integration/ -v -m "integration and requires_spark"

# Run e2e tests in Docker
docker-test-e2e:
	@./scripts/run_tests_in_docker.sh tests/e2e/ -v

# Run tests with coverage in Docker (terminal report)
docker-test-cov:
	@./scripts/run_tests_in_docker.sh tests/ --cov=acm_forecast --cov-report=term-missing

# Run tests with coverage in Docker (HTML report)
docker-test-cov-html:
	@./scripts/run_tests_in_docker.sh tests/ --cov=acm_forecast --cov-report=html --cov-report=term-missing
	@echo "Coverage report available at htmlcov/index.html"

# ============================================================================
# Docker - Code Quality
# ============================================================================

# Format code in Docker
docker-format:
	docker compose run --rm acm-forecast-dev black acm_forecast tests

# Lint code in Docker
docker-lint:
	docker compose run --rm acm-forecast-dev flake8 acm_forecast tests

# Type check with mypy in Docker
docker-typecheck:
	docker compose run --rm acm-forecast-dev mypy acm_forecast

# ============================================================================
# Docker - Individual Services
# ============================================================================

# Start MLflow service (common service)
docker-mlflow:
	docker compose -f docker-compose.common.yml up -d mlflow
	@echo "MLflow UI available at http://localhost:5001"

# Start Jupyter Lab (dev only)
docker-jupyter:
	docker compose -f docker-compose.dev.yml up -d jupyter
	@echo "Jupyter Lab available at http://localhost:8888"

# Start Azurite (Azure Storage Emulator, common service)
docker-azurite:
	docker compose -f docker-compose.common.yml --profile with-azurite up -d azurite
	@echo "Azurite services available at:"
	@echo "  - Blob: http://localhost:10000"
	@echo "  - Queue: http://localhost:10001"
	@echo "  - Table: http://localhost:10002"

# ============================================================================
# Docker - Azure ADLS Configuration
# ============================================================================

# Start services with Azure ADLS config (requires docker compose.override.yml)
docker-azure-up:
	@if [ ! -f docker compose.override.yml ]; then \
		echo "Error: docker compose.override.yml not found"; \
		echo "Copy docker compose.azure.yml.example to docker compose.override.yml and configure it"; \
		exit 1; \
	fi
	docker compose -f docker-compose.yml -f docker compose.override.yml up -d

# Stop services with Azure ADLS config
docker-azure-down:
	@if [ ! -f docker compose.override.yml ]; then \
		echo "Error: docker compose.override.yml not found"; \
		exit 1; \
	fi
	docker compose -f docker-compose.yml -f docker compose.override.yml down

# ============================================================================
# Docker - Cleanup
# ============================================================================

# Stop and remove containers (development)
docker-clean:
	docker compose down

# Stop and remove production containers
docker-clean-prod:
	docker compose -f docker-compose.prod.yml down

# Stop and remove development containers
docker-clean-dev:
	docker compose -f docker-compose.dev.yml down

# Remove containers, volumes, and images (development)
docker-clean-all:
	docker compose down -v --rmi local

# Remove production containers, volumes, and images
docker-clean-all-prod:
	docker compose -f docker-compose.prod.yml down -v --rmi local

# Remove development containers, volumes, and images
docker-clean-all-dev:
	docker compose -f docker-compose.dev.yml down -v --rmi local

# Prune Docker system (clean everything)
docker-prune:
	@echo "This will remove all unused Docker resources. Continue? (y/N)"
	@read confirm && [ "$$confirm" = "y" ] || exit 1
	docker system prune -a --volumes

