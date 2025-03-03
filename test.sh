#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Red Ribbon MK3 Test Suite ===${NC}"
echo "Running from directory: $(pwd)"

# Create test directory structure if it doesn't exist
mkdir -p tests-unit/custom_nodes/red_ribbon/utils

# Copy the test file if it's not already there
if [ ! -f tests-unit/custom_nodes/red_ribbon/utils/test_instantiate.py ]; then
  echo -e "${YELLOW}Creating test_instantiate.py...${NC}"
  cp test_instantiate.py tests-unit/custom_nodes/red_ribbon/utils/test_instantiate.py
fi

# Run diagnostic checks
echo -e "${YELLOW}Running directory structure diagnostics...${NC}"
echo "Project root: $(pwd)"
if [ -d "./custom_nodes" ]; then
  echo -e "${GREEN}✓${NC} custom_nodes directory found"
  ls -la ./custom_nodes
else
  echo -e "${RED}✗${NC} custom_nodes directory not found"
fi

if [ -d "./custom_nodes/red_ribbon" ]; then
  echo -e "${GREEN}✓${NC} red_ribbon module directory found"
  ls -la ./custom_nodes/red_ribbon
else
  echo -e "${RED}✗${NC} red_ribbon module directory not found"
fi

if [ -d "./custom_nodes/red_ribbon/utils" ]; then
  echo -e "${GREEN}✓${NC} utils directory found"
  ls -la ./custom_nodes/red_ribbon/utils
else
  echo -e "${RED}✗${NC} utils directory not found"
fi

# Run the Python test
echo -e "\n${YELLOW}Running Python unit tests...${NC}"
python -m tests-unit.custom_nodes.red_ribbon.utils.test_instantiate

# Check if the test succeeded
if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}All tests passed!${NC}"
  exit 0
else
  echo -e "\n${RED}Tests failed!${NC}"
  exit 1
fi