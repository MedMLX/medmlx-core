.PHONY: typecheck

typecheck:
	uv run --locked --extra dev pyright --project . --warnings
	bash -o pipefail -c 'rg --files --hidden --no-ignore --glob "*.py" --glob "*.pyi" --null scripts src tests | xargs -0 uv run --locked --extra dev ruff check --config pyproject.toml --select TID251,ANN401 --ignore-noqa --no-force-exclude --'
