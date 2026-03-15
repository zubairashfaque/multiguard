.PHONY: install lint format test test-all test-smoke ingest train-baseline train-fusion train-distill evaluate benchmark export-onnx serve serve-docker clean

install:
	poetry install
	poetry run pre-commit install

lint:
	poetry run ruff check src/ tests/ scripts/
	poetry run black --check src/ tests/ scripts/
	poetry run mypy src/

format:
	poetry run black src/ tests/ scripts/
	poetry run isort src/ tests/ scripts/
	poetry run ruff check --fix src/ tests/ scripts/

test:
	poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-all:
	poetry run pytest tests/unit/ tests/integration/ -v --cov=src --cov-report=term-missing

test-smoke:
	poetry run pytest tests/smoke/ -v

ingest:
	poetry run python scripts/ingest_data.py

train-baseline:
	poetry run python scripts/train.py --config configs/train/baseline.yaml

train-fusion:
	poetry run python scripts/train.py --config configs/train/fusion_ablation.yaml

train-distill:
	poetry run python scripts/train.py --config configs/train/distillation.yaml

evaluate:
	poetry run python scripts/evaluate.py --config configs/train/baseline.yaml --checkpoint models/checkpoints/baseline/best_model.pt

benchmark:
	poetry run python scripts/run_benchmark.py --all-checkpoints

export-onnx:
	poetry run python scripts/export_model.py --format onnx --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml

PORT ?= 8000
serve:
	poetry run uvicorn src.serving.app:app --host 0.0.0.0 --port $(PORT) --reload

serve-docker:
	docker-compose -f infrastructure/docker-compose.yml up --build

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache dist build
