.PHONY: install run test lint format clean-runtime init-db demo-data didi-example-data

install:
	uv sync

run:
	uv run streamlit run app.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

init-db:
	uv run python scripts/init_db.py

demo-data:
	uv run python scripts/generate_demo_data.py

didi-example-data:
	uv run python -m scripts.generate_didi_example_data

clean-runtime:
	uv run python scripts/clean_runtime.py
