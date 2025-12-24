#!/bin/bash
# Run pytest tests in Docker container
# This ensures Java is available for PySpark tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running tests in Docker container...${NC}"

# Get the container name
CONTAINER_NAME="acm-forecast-dev"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Container ${CONTAINER_NAME} is not running. Starting it...${NC}"
    docker compose -f docker-compose.dev.yml up -d acm-forecast-dev
    echo "Waiting for container to be ready..."
    sleep 5
fi

# Parse command line arguments
TEST_ARGS="$@"

# Default test command if no arguments provided
if [ -z "$TEST_ARGS" ]; then
    TEST_ARGS="tests/ -v"
fi

echo -e "${GREEN}Running: pytest ${TEST_ARGS}${NC}"
echo ""

# Run tests in container (use venv python with Java environment)
docker exec -e PYTHONPATH=/app -e PATH="/opt/venv/bin:${PATH}" -e JAVA_HOME=/usr/lib/jvm/default-java ${CONTAINER_NAME} /opt/venv/bin/pytest ${TEST_ARGS}

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}❌ Some tests failed (exit code: ${EXIT_CODE})${NC}"
fi

exit $EXIT_CODE

