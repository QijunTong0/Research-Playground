# Research Repository Template

This is a template for a research repository that uses `uv` for package management and `pre-commit` for code quality.

## Setup

1.  **Install uv**:
    If you don't have `uv` installed, follow the official installation instructions:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Create and activate a virtual environment**:
    This project uses `pyproject.toml` to manage dependencies. Install them with:
    ```bash
    uv sync
    ```

## Pre-commit Hooks

This repository uses `pre-commit` to enforce code style and quality.

1.  **Install pre-commit**:
    ```bash
    uv pip install pre-commit
    ```

2.  **Install the git hooks**:
    ```bash
    pre-commit install
    ```

Now, `pre-commit` will run automatically on every commit.

## Managing Dependencies

To add a new package, add it to the `dependencies` list in `pyproject.toml` and then run:

```bash
uv sync
```

Or you can directly add by
```bash
uv add some-new-package
```

## manim commands
Rendering manim scene defined in main.py
```bash
manim -pql main.py SceneClass
manim -pqh main.py SceneClass
```