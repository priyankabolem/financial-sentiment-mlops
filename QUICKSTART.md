# Quick Start Guide

This guide will help you get the Financial Sentiment MLOps pipeline up and running quickly.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional, for containerized deployment)
- Git

## Setup Steps

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/priyankabolem/financial-sentiment-mlops.git
cd financial-sentiment-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# OR
pip install -r requirements.txt && pip install -e .
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys (optional for demo)
# For demo purposes, you can skip API keys and use synthetic data
```

### 3. Quick Demo (Without Real Data)

The easiest way to test the system is using Docker Compose:

```bash
# Start all services
docker-compose up -d

# Wait for services to start (about 30 seconds)
docker-compose logs -f api
```

Access the services:
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### 4. Test the API

**Using the Interactive Docs:**

1. Go to http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Enter a sample text: `"Tesla stock surges on strong earnings report"`
5. Click "Execute"

**Using cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock market rallying with record profits"}'
```

**Using Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Stock market rallying with record profits"}
)

print(response.json())
```

## Running the Full Pipeline (With Real Data)

### Step 1: Data Ingestion

```bash
# Make sure you've configured API keys in .env
python scripts/ingest_data.py

# This will fetch data from:
# - Alpha Vantage (requires API key)
# - Reddit (requires API credentials)
# - News API (requires API key)
```

### Step 2: Preprocess Data

```bash
python scripts/preprocess_data.py

# This will:
# - Clean text data
# - Engineer features
# - Save processed data
```

### Step 3: Train Model

```bash
python scripts/train.py

# This will:
# - Load processed data
# - Train FinBERT model
# - Track experiments in MLflow
# - Save best model
```

### Step 4: Deploy API

```bash
# Option 1: Local development
uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000

# Option 2: Docker
docker-compose up -d
```

## Using the Makefile

The project includes a Makefile for common tasks:

```bash
# Install dependencies
make install

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Data pipeline
make ingest       # Ingest data
make preprocess   # Preprocess data
make train        # Train model

# Docker
make docker-build # Build Docker image
make docker-up    # Start services
make docker-down  # Stop services

# Serve API
make serve        # Run API locally

# MLflow UI
make mlflow       # Start MLflow UI
```

## Monitoring & Observability

### View Metrics in Prometheus

1. Go to http://localhost:9090
2. Try these queries:
   - `sentiment_request_total` - Total requests
   - `sentiment_request_latency_seconds` - Request latency
   - `rate(sentiment_request_total[5m])` - Request rate

### View Dashboards in Grafana

1. Go to http://localhost:3000
2. Login: admin / admin
3. Add Prometheus as data source:
   - URL: http://prometheus:9090
4. Import dashboards or create custom ones

### Track Experiments in MLflow

1. Go to http://localhost:5000
2. View experiments, runs, and metrics
3. Compare different model versions

## Testing

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Port Already in Use

If you see "port already in use" errors:

```bash
# Find and kill processes
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
lsof -ti:5000 | xargs kill -9  # Kill process on port 5000

# OR change ports in docker-compose.yml
```

### Model Download Issues

If the FinBERT model fails to download:

1. Check internet connection
2. Try downloading manually:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")
```

### API Keys Not Working

- Verify API keys are correct in `.env`
- Check if API has rate limits
- For demo, you can skip data ingestion and use synthetic data

## Next Steps

1. **Customize Model**: Modify `configs/model/finbert.yaml` for different architectures
2. **Add Data Sources**: Extend `src/data_ingestion/` with new sources
3. **Improve Features**: Add new features in `src/feature_engineering/`
4. **Deploy to Cloud**: Use Kubernetes manifests in `infrastructure/kubernetes/`
5. **Monitor Production**: Set up Grafana dashboards for production monitoring

## Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs or questions
- Review code comments for implementation details

## Clean Up

```bash
# Stop Docker services
docker-compose down

# Remove volumes (careful - deletes data!)
docker-compose down -v

# Clean Python artifacts
make clean
```

---

**Happy MLOps! 🚀**
