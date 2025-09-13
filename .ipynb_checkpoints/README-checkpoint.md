# Fraud Management

This repository contains a Python script for credit card fraud detection with risk-based business rules.

## View outputs in GitHub as a notebook

This project is set up to keep a rendered Jupyter Notebook in the repo so you can view outputs directly on GitHub.

Two files are produced from the source script `Fraud_notebook.py`:

- `Fraud_notebook.ipynb`: the notebook version of the script
- `Fraud_notebook.executed.ipynb`: the executed notebook with captured outputs

We use Jupytext to convert the Python script to a notebook and Papermill to execute it and persist outputs.

### One-command local build

Run the build script to generate both notebooks:

```
bash scripts/build_notebook.sh
```

This will:
- install tools (jupytext, papermill, ipykernel) if needed
- convert `Fraud_notebook.py` -> `Fraud_notebook.ipynb`
- execute it to create `Fraud_notebook.executed.ipynb`

If you're using VS Code, a task is configured:
- Open the Command Palette → "Run Task" → "Build Fraud Notebook"

### Keeping the script and notebook in sync

`Fraud_notebook.py` includes Jupytext cell markers, so it round-trips cleanly with `.ipynb`. Treat the `.py` file as the source of truth; rebuild notebooks after changes.

### Continuous Integration (optional, already added)

On pushes to `main`, GitHub Actions will:
1. Convert `Fraud_notebook.py` to `Fraud_notebook.ipynb`
2. Execute it and save `Fraud_notebook.executed.ipynb`
3. Commit and push any changes back to the repo

Workflow file: `.github/workflows/build-notebook.yml`

### Dataset note

If `creditcard.csv` is not present, the script automatically generates a synthetic dataset so the notebook always runs end-to-end.

### Requirements

See `requirements.txt` for core Python packages. The build script installs additional tooling automatically.
