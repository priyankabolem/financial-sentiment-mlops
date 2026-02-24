# Setup Guide

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional)
- Git

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/priyankabolem/financial-sentiment-mlops.git
cd financial-sentiment-mlops
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Running the Project

### Option 1: Using Docker Compose

```bash
docker-compose up -d
```

Services will be available at:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Option 2: Local Development

#### Run Data Pipeline

```bash
# Data ingestion
python scripts/ingest_data.py

# Data preprocessing
python scripts/preprocess_data.py

# Model training
python scripts/train.py
```

#### Run API Server

```bash
uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000
```

#### Run Demo App

```bash
streamlit run app.py
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
financial-sentiment-mlops/
├── configs/          # Configuration files
├── src/              # Source code
│   ├── data_ingestion/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── models/
│   ├── training/
│   ├── deployment/
│   └── monitoring/
├── tests/            # Test suite
├── scripts/          # Executable scripts
├── infrastructure/   # Docker, Kubernetes configs
└── app.py           # Interactive demo
```

## Common Issues

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Model download issues:**
Check internet connection or manually download FinBERT model.

**Missing API keys:**
Update .env file with valid API credentials.

## Documentation

- API docs: http://localhost:8000/docs (when running)
- Main README: [README.md](README.md)

## Contact

For issues or questions, please open an issue on GitHub.
