# Financial Sentiment MLOps - Enterprise-Grade End-to-End ML Lifecycle System

[![🤗 Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis)
[![CI/CD Pipeline](https://github.com/priyankabolem/financial-sentiment-mlops/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/priyankabolem/financial-sentiment-mlops/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, production-grade Machine Learning system that covers the complete ML lifecycle from business problem definition to continuous monitoring and automated retraining. This project demonstrates advanced ML Engineering and MLOps capabilities suitable for industry-level deployment.

## 🎮 Try the Live Demo

**[👉 Click here to try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/priyankabolem/financial-sentiment-analysis)**

Experience real-time financial sentiment analysis with:
- ✨ Interactive web interface
- 📊 Live predictions with confidence scores
- 📈 Visual sentiment distributions
- 💡 Pre-loaded examples (news, social media, earnings)
- 🚀 Instant results powered by FinBERT

> **Note:** The demo showcases the ML model in action. For the full production MLOps infrastructure (Docker, Kubernetes, monitoring, CI/CD), see the sections below.

## 🎯 Project Objective

Build a financial sentiment intelligence platform that:
- Analyzes financial news, earnings call transcripts, and social media data
- Performs multi-class sentiment analysis (positive, negative, neutral)
- Provides explainable insights for decision-making
- Supports traders, analysts, and fintech platforms
- Continuously monitors and improves model performance in production

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Ingestion Layer                        │
│  News API │ Alpha Vantage │ Twitter API │ Reddit API │ Custom   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                          │
│  Text Cleaning │ Feature Engineering │ Data Validation │ DVC    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Layer                                │
│  FinBERT │ Hyperparameter Tuning │ MLflow Tracking │ Registry   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Deployment Layer                               │
│  FastAPI │ Docker │ Kubernetes │ Model Versioning │ A/B Testing │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Monitoring Layer                                │
│  Prometheus │ Grafana │ Data Drift │ Performance │ Alerting     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 1. **Problem Framing & Metrics**
- ✅ Business KPIs: Sentiment accuracy vs market movements
- ✅ ML Metrics: F1-score, ROC-AUC, precision, recall
- ✅ Financial Impact: Backtesting against market data
- ✅ Risk Evaluation: Confidence thresholds and uncertainty quantification

### 2. **Data Pipeline**
- ✅ Multi-source ingestion (News API, Alpha Vantage, Reddit)
- ✅ Real-time and batch processing
- ✅ Data validation with Great Expectations
- ✅ Version control with DVC
- ✅ Feature store for reusability

### 3. **Model Development**
- ✅ FinBERT transformer for financial sentiment
- ✅ Transfer learning and fine-tuning
- ✅ Hyperparameter optimization with Optuna
- ✅ Experiment tracking with MLflow
- ✅ Model registry and versioning

### 4. **Deployment**
- ✅ REST API with FastAPI
- ✅ Docker containerization
- ✅ Kubernetes orchestration
- ✅ Horizontal scaling
- ✅ API rate limiting and caching

### 5. **Monitoring & Observability**
- ✅ Real-time metrics with Prometheus
- ✅ Visualization dashboards with Grafana
- ✅ Data drift detection with Evidently
- ✅ Model performance monitoring
- ✅ Automated alerting

### 6. **CI/CD Pipeline**
- ✅ Automated testing (unit, integration, performance)
- ✅ Code quality checks (Black, Flake8, isort)
- ✅ Security scanning with Trivy
- ✅ Automated deployment
- ✅ Rollback capabilities

## 📋 Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- (Optional) Kubernetes cluster
- (Optional) Cloud account (AWS/GCP/Azure)

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/priyankabolem/financial-sentiment-mlops.git
cd financial-sentiment-mlops
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# NEWS_API_KEY=your_key_here
# ALPHA_VANTAGE_API_KEY=your_key_here
# etc.
```

### 4. Create Required Directories

```bash
mkdir -p data/{raw,processed,features} models logs mlruns
```

## 🎮 Usage

### Quick Start with Docker Compose

```bash
# Start all services (MLflow, API, Prometheus, Grafana)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

Access the services:
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Running the ML Pipeline

#### Step 1: Data Ingestion

```bash
python scripts/ingest_data.py
```

This will:
- Fetch data from configured sources (News API, Alpha Vantage, Reddit)
- Validate and clean the data
- Save raw data to `data/raw/`

#### Step 2: Data Preprocessing

```bash
python scripts/preprocess_data.py
```

This will:
- Clean and normalize text
- Engineer features
- Save processed data to `data/processed/`

#### Step 3: Model Training

```bash
python scripts/train.py
```

This will:
- Load processed data
- Train FinBERT model
- Track experiments in MLflow
- Save best model to `models/best_model/`

#### Step 4: Model Deployment

```bash
# Local development
uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000

# Production with Docker
docker-compose up -d
```

### Making Predictions

#### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Stock market rallying with strong earnings"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Markets are crashing due to recession fears",
            "Neutral outlook for the economy",
            "Tech stocks surge on AI optimism"
        ]
    }
)
print(response.json())
```

#### Using cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock prices are soaring with record profits"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Bullish market trends", "Bearish sentiment prevails"]}'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# View coverage report
open htmlcov/index.html
```

## 📊 Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Key metrics:
- `sentiment_request_total` - Total API requests
- `sentiment_request_latency_seconds` - Request latency
- `sentiment_prediction_total` - Predictions by sentiment class

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

Pre-configured dashboards:
- API Performance
- Model Predictions Distribution
- System Resources
- Error Rates

### Data Drift Detection

```python
from src.monitoring.data_drift import DataDriftDetector
import pandas as pd

# Load reference and current data
reference_df = pd.read_parquet("data/processed/reference_data.parquet")
current_df = pd.read_parquet("data/processed/current_data.parquet")

# Detect drift
detector = DataDriftDetector(reference_df)
drift_report = detector.detect_drift(current_df, save_report=True)

print(f"Dataset drift detected: {drift_report['dataset_drift']}")
print(f"Drifted columns: {drift_report['number_of_drifted_columns']}")
```

## 🔧 Configuration

All configurations are managed with Hydra in the `configs/` directory:

- `configs/config.yaml` - Main configuration
- `configs/data/` - Data source configurations
- `configs/model/` - Model architectures
- `configs/training/` - Training parameters
- `configs/deployment/` - Deployment settings

Example: Override configuration

```bash
# Use different model
python scripts/train.py model=distilbert

# Change batch size
python scripts/train.py training.batch_size=32

# Multiple overrides
python scripts/train.py model=finbert training.num_epochs=10 training.batch_size=16
```

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t financial-sentiment-mlops:latest .

# Run container
docker run -p 8000:8000 financial-sentiment-mlops:latest
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f infrastructure/kubernetes/

# Check deployment
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment sentiment-api --replicas=3
```

## 📈 Performance

- **Latency**: < 100ms (p99)
- **Throughput**: > 100 requests/second
- **Model Accuracy**: ~85% F1-score on test set
- **Availability**: 99.9% uptime

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Priyanka Bolem**

- GitHub: [@priyankabolem](https://github.com/priyankabolem)
- LinkedIn: [Priyanka Bolem](https://linkedin.com/in/priyankabolem)

## 🙏 Acknowledgments

- FinBERT model by ProsusAI
- MLflow for experiment tracking
- FastAPI for API framework
- Evidently AI for drift detection

## 📚 Project Structure

```
financial-sentiment-mlops/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── data/                  # Data configs
│   ├── model/                 # Model configs
│   ├── training/              # Training configs
│   └── deployment/            # Deployment configs
├── src/                       # Source code
│   ├── data_ingestion/        # Data collection
│   ├── data_preprocessing/    # Text cleaning
│   ├── feature_engineering/   # Feature creation
│   ├── models/                # Model definitions
│   ├── training/              # Training logic
│   ├── evaluation/            # Model evaluation
│   ├── deployment/            # API and serving
│   ├── monitoring/            # Monitoring tools
│   └── utils/                 # Utilities
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Performance tests
├── scripts/                   # Executable scripts
│   ├── ingest_data.py
│   ├── preprocess_data.py
│   └── train.py
├── infrastructure/            # Infrastructure as Code
│   ├── docker/                # Dockerfiles
│   ├── kubernetes/            # K8s manifests
│   ├── terraform/             # Terraform configs
│   ├── prometheus/            # Prometheus config
│   └── grafana/               # Grafana dashboards
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
├── .github/                   # GitHub Actions
│   └── workflows/             # CI/CD pipelines
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── features/              # Feature store
├── models/                    # Saved models
├── mlruns/                    # MLflow artifacts
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Multi-container setup
└── README.md                  # This file
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **MLOps Best Practices**: End-to-end ML lifecycle management
2. **System Design**: Scalable, maintainable architecture
3. **Production Deployment**: Real-world deployment strategies
4. **Monitoring**: Data drift, model performance, system health
5. **DevOps**: CI/CD, containerization, orchestration
6. **Software Engineering**: Testing, documentation, code quality

---

**⭐ If you find this project helpful, please consider giving it a star!**
