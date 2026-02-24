.PHONY: help install clean test lint format docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Clean generated files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make ingest        - Run data ingestion"
	@echo "  make preprocess    - Run data preprocessing"
	@echo "  make train         - Train model"
	@echo "  make serve         - Serve API locally"

install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

docker-build:
	docker build -t financial-sentiment-mlops:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

ingest:
	python scripts/ingest_data.py

preprocess:
	python scripts/preprocess_data.py

train:
	python scripts/train.py

serve:
	uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000
