.PHONY: install lint format test test-cov clean build

install:
	uv pip install -e ".[dev]"

lint:
	ruff check pytorch_scheduler/ tests/
	pyright pytorch_scheduler/

format:
	ruff format pytorch_scheduler/ tests/
	ruff check --fix pytorch_scheduler/ tests/

test:
	pytest

test-cov:
	pytest --cov=pytorch_scheduler --cov-report=html

clean:
	rm -rf dist/ build/ *.egg-info htmlcov/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	uv build
