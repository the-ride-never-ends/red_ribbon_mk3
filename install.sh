#!/bin/bash

echo "Setting up the environment..."

# Check if Python 3.12 or later is installed
if ! python3 -c 'import sys; assert sys.version_info >= (3, 12)' &> /dev/null
then
    echo "Python 3.12 or later is not installed. Please install Python 3.12 or later and add it to your PATH."
    exit 1
fi

# Check if the virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    # Create a virtual environment with Python 3.12 or later if it doesn't exist
    echo "Creating a virtual environment with Python 3.12 or later..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating the virtual environment '.venv'..."
source .venv/bin/activate

# Install required packages from requirements.txt
if [[ -f "requirements.txt" ]]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

if [[ -f "requirements_custom_nodes.txt" ]]; then
    echo "Installing packages for custom nodes..."
    pip install -r requirements_custom_nodes.txt
else
    echo "requirements_custom_nodes.txt not found. Skipping package installation for custom nodes."
fi

echo "Setup complete!"
