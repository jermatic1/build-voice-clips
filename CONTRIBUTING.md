# Contributing

Install the project with the development tools:

```bash
uv sync --group dev
```

Run the tests:

```bash
uv run pytest
```

Lint and check formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

Tests do not load the model, so they run without a GPU or the checkpoint.
