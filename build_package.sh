#!/bin/bash

# Run the setup.py command
python setup.py sdist bdist_wheel

# Use pip to install the package
pip install .

# Optional: Print a message indicating successful execution
echo "Build completed successfully."

