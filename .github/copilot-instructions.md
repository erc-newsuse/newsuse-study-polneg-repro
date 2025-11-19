# GitHub Copilot Instructions

## Project Architecture & Overview
- **Goal**: Reproducible research pipeline for political negativity analysis in news.
- **Core Stack**: Python (>=3.11), R, DVC (Data Version Control).
- **Structure**:
  - `stages/`: Pipeline scripts (Python & R). Each script typically corresponds to a DVC stage.
  - `project/`: Local Python package containing shared utilities, configuration (`config`), and paths (`paths`).
  - `analyses/`: Quarto documents for reporting and visualization.
  - `data/`: DVC-managed data storage (`raw`, `proc`, `ml`).
  - `dvc.yaml`: Defines the dependency graph and execution stages.

## Critical Workflows
- **Pipeline Execution**: Use `dvc repro` to run the pipeline. Do not run scripts in `stages/` manually unless debugging.
- **Data Management**:
  - `dvc.yaml` defines inputs (`deps`) and outputs (`outs`).
  - `dvc-dehydrated.yaml` is a simplified pipeline for reproduction without full text data.
- **Environment**:
  - Python dependencies in `pyproject.toml`.
  - R dependencies managed via `Makefile` (and `renv` implicitly or system libs).
  - Use `make init` to setup the environment.

## Code Conventions

### Python
- **Style**:
  - **Strings**: ALWAYS use double quotes (`"string"`).
  - **Naming**: `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
  - **Docstrings**: MUST use **NumpyDoc** convention for all functions, classes, and methods.
  - **Error Messages**: Concise, single quasi-sentence (lowercase start, no period) OR full sentences (uppercase start, period end) for long messages.
- **Patterns**:
  - **Pandas**: Prefer method chaining (fluent interface) for data transformations.
  - **Type Hinting**: Use standard `typing` and `pydantic` for data models (e.g., in `project/gpt.py`).
  - **Scripts**: Use `# %%` markers to denote code cells (Jupytext/VS Code compatible).
  - **Imports**: Import internal modules from `project` (e.g., `from project import config, paths`).

### R
- **Integration**: R scripts (`stages/*.R`) often use `reticulate` to import the Python `project` package for consistent paths and config.
  ```r
  library(reticulate)
  project <- import("project")
  config  <- project$config
  ```
- **Style**: Tidyverse style (`dplyr`, pipes `%>%`).

## Key Files & Paths
- `dvc.yaml`: The source of truth for data dependencies.
- `project/__init__.py`: Exposes key package components.
- `stages/make_*.py`: Data processing scripts.
- `analyses/*.qmd`: Quarto analysis files.

## Common Tasks
- **Adding a Stage**:
  1. Create script in `stages/`.
  2. Define stage in `dvc.yaml` with `cmd`, `deps`, and `outs`.
  3. Run `dvc repro`.
- **Loading Data**:
  - Use `project.paths` to locate files.
  - In Python: `pd.read_parquet(paths.proc / "dataset.parquet")`.
  - In R: `read_parquet(as.character(paths$proc / "dataset.parquet"))`.
