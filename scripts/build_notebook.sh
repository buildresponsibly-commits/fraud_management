#!/usr/bin/env bash
set -euo pipefail

# Ensure dependencies
python -m pip install --quiet --upgrade jupytext papermill ipykernel

# Ensure python kernel is available
python -m ipykernel install --user --name python3 --display-name "Python 3" >/dev/null 2>&1 || true

# Convert script to notebook
jupytext --to ipynb Fraud_notebook.py

# Execute notebook and save outputs
papermill --kernel python3 Fraud_notebook.ipynb Fraud_notebook.executed.ipynb

echo "Built Fraud_notebook.ipynb and Fraud_notebook.executed.ipynb"