# Phish Detector 🎣

**ML-powered phishing email detection system with explainable predictions.**

A production-ready machine learning pipeline for detecting phishing emails using the CEAS dataset. Features include TF-IDF text analysis, heuristic feature engineering, model explainability, and a FastAPI REST service.

## Features

## Igor Radulovic, Jovana Mitric, Mija Ljuka, Katarina Kovijanic

- ✅ **Full ML Pipeline**: Data preprocessing → Feature engineering → Training → Evaluation
- ✅ **Explainable AI**: Global feature importance + local explanations per email
- ✅ **Comprehensive Metrics**: Precision, Recall, F1, ROC-AUC, **False Positive Rate (FPR)**
- ✅ **Production Ready**: FastAPI service, Docker support, structured logging
- ✅ **Reproducible**: Fixed random seeds, versioned dependencies, clear configuration

## Dataset

This project uses the **CEAS phishing email dataset** with the following columns:

| Column | Description |
|--------|-------------|
| `sender` | Email sender address |
| `receiver` | Email recipient address |
| `date` | Email date (note: may contain outliers like year 2100) |
| `subject` | Email subject line |
| `body` | Email body content |
| `label` | 0 = legitimate, 1 = phishing |
| `urls` | URLs found in the email |

**Note**: The dataset may contain outlier dates (e.g., year 2100). Dates are parsed for analytics only and not used in model features.

## Project Structure

```
phish-detector/
├── data/
│   └── raw.csv              # Place your dataset here
├── models/
│   └── model.joblib         # Trained pipeline (generated)
├── reports/
│   ├── metrics.json         # Evaluation metrics
│   ├── confusion_matrix.csv # Confusion matrix
│   ├── threshold_metrics.csv# Metrics at different thresholds
│   ├── top_features_phishing.csv
│   └── top_features_legit.csv
├── src/
│   ├── __init__.py
│   ├── config.py            # All configuration
│   ├── utils.py             # Helper functions
│   ├── preprocessing.py     # Data cleaning & splitting
│   ├── features.py          # Feature engineering
│   ├── train.py             # Model training
│   ├── evaluate.py          # Evaluation & metrics
│   ├── explain.py           # Model explainability
│   ├── inference.py         # CLI prediction
│   ├── api.py               # FastAPI service
│   └── visualize.py         # Generate charts & graphs
├── frontend/
│   └── index.html           # Web UI for phishing detection
├── Metrike/                   # Generated visualization images
│   ├── 01_metrics_summary.png
│   ├── 02_metrics_table.png
│   ├── 03_confusion_matrix.png
│   ├── 04_class_distribution.png
│   ├── 05_roc_curve.png
│   ├── 06_threshold_analysis.png
│   └── 07_feature_importance.png
├── tests/
│   ├── test_features.py
│   └── test_inference.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CEAS dataset CSV file at `data/raw.csv` with the required columns.

### 3. Train Model

```bash
python -m src.train
```

This will:
- Load and preprocess the data
- Train the phishing detection model
- Save the model to `models/model.joblib`

### 4. Evaluate Model

```bash
python -m src.evaluate
```

Output includes:
- Precision, Recall, F1 Score
- **False Positive Rate (FPR)** - critical for email filtering
- ROC-AUC score
- Threshold analysis (0.3, 0.5, 0.7, 0.9)

### 5. Generate Explanations

```bash
python -m src.explain
```

Generates:
- `reports/top_features_phishing.csv` - Top 30 phishing indicators
- `reports/top_features_legit.csv` - Top 30 legitimate indicators

### 5b. Generate Visualizations (Charts & Graphs)

```bash
python -m src.visualize
```

Generates all metrics as high-quality images in the `Metrike/` folder:
- `01_metrics_summary.png` - Bar chart of all performance metrics
- `02_metrics_table.png` - Complete metrics table
- `03_confusion_matrix.png` - Confusion matrix heatmap
- `04_class_distribution.png` - Pie chart of class distribution
- `05_roc_curve.png` - ROC curve with AUC score
- `06_threshold_analysis.png` - Precision/Recall/F1 vs threshold
- `07_feature_importance.png` - Top 20 features for phishing/legitimate

### 6. CLI Inference

```bash
# Simple prediction
python -m src.inference --text "Your account has been suspended. Click here to verify."

# With URLs
python -m src.inference --subject "URGENT" --body "Verify your account" --urls "http://bit.ly/xyz"

# JSON output
python -m src.inference --text "..." --json
```

### 7. Start API Server

```bash
# Development mode
uvicorn src.api:app --reload

# Production mode
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 8. Open Web Interface

After starting the API server, open your browser and navigate to:

```
http://localhost:8000
```

The web interface allows you to:
- Enter email subject, body, and URLs
- Get instant phishing/legitimate classification
- View confidence score with visual indicator
- See detailed analysis reasons
- Review extracted heuristic features

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### Predict Email

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Verify your account",
    "body": "Click the link below to verify your identity immediately.",
    "urls": "http://suspicious-site.xyz/verify"
  }'
```

Response:
```json
{
  "label": "PHISHING",
  "score": 0.87,
  "reasons": [
    "Contains URL with suspicious TLD - indicates phishing",
    "Contains 3 phishing keyword(s) - indicates phishing",
    "Contains term 'verify' - indicates phishing"
  ],
  "model_version": "1.0.0"
}
```

### Detailed Prediction (with heuristics)

```bash
curl -X POST http://localhost:8000/predict/detailed \
  -H "Content-Type: application/json" \
  -d '{"text": "Your account needs verification at http://bit.ly/verify"}'
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t phish-detector .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro phish-detector
```

### Using Docker Compose

```bash
# Start API service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Training with Docker

```bash
# Run training
docker-compose run --rm trainer

# Run evaluation
docker-compose run --rm evaluator
```

## Model Details

### Features

**TF-IDF Features** (up to 30,000):
- Unigrams and bigrams from combined subject + body
- Sublinear TF scaling
- Min document frequency: 3

**Heuristic Features** (10):
| Feature | Description |
|---------|-------------|
| `url_count` | Number of URLs in email |
| `has_url` | Binary: contains URL |
| `exclamation_count` | Count of "!" |
| `question_count` | Count of "?" |
| `text_len` | Character count |
| `word_count` | Word count |
| `caps_ratio` | Ratio of ALL CAPS tokens |
| `keyword_hits` | Phishing keyword matches |
| `suspicious_tld` | URL with .xyz, .tk, .ru, etc. |
| `uses_shortener` | URL shortener (bit.ly, etc.) |

### Classifier

- **Algorithm**: Logistic Regression
- **Class Weighting**: Balanced (handles class imbalance)
- **Solver**: liblinear

## Evaluation Metrics

After training, check `reports/metrics.json`:

```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.97,
  "f1_score": 0.95,
  "false_positive_rate": 0.07,
  "roc_auc": 0.98,
  "confusion_matrix": {
    "true_positives": 1500,
    "false_positives": 100,
    "true_negatives": 1400,
    "false_negatives": 50
  }
}
```

**Important**: The **False Positive Rate (FPR)** indicates what percentage of legitimate emails are incorrectly flagged as phishing. For production email filtering, aim for FPR < 1%.

## Deployment Considerations

### Performance
- **Latency**: ~10-50ms per prediction (depends on text length)
- **Throughput**: Scales horizontally via container replicas
- **Memory**: ~200-500MB depending on vocabulary size

### False Positive Rate
- Critical for email systems - legitimate emails marked as spam
- Consider threshold tuning (see `reports/threshold_metrics.csv`)
- Lower threshold = higher recall but more false positives

### Monitoring
- Track prediction distribution over time
- Monitor for concept drift (phishing tactics evolve)
- Log suspicious patterns for model retraining

### Security
- Validate input size to prevent DoS
- Rate limiting recommended for production
- Don't expose internal model details

## Limitations & Robustness

### Dataset Bias
- Model performance depends on training data representativeness
- CEAS dataset may not cover all phishing tactics
- Regular retraining with new samples recommended

### Concept Drift
- Phishing techniques evolve rapidly
- Model may degrade over time without updates
- Monitor classification accuracy in production

### Obfuscation
- Attackers may use:
  - Character substitution (0 for O, 1 for l)
  - Invisible characters
  - Image-based text
  - Encoded URLs
- Consider preprocessing to normalize text

### Language
- Model trained primarily on English text
- May not perform well on other languages
- Multi-language support requires additional training

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_features.py -v
```

## Configuration

All configuration is centralized in `src/config.py`:

```python
# Key settings
RANDOM_STATE = 42          # Reproducibility seed
TEST_SIZE = 0.2            # Train/test split ratio
TFIDF_MAX_FEATURES = 30000 # Max TF-IDF features
LR_CLASS_WEIGHT = "balanced" # Handle class imbalance
```

## License

This project is provided for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

**Built with**: Python 3.11, scikit-learn, FastAPI, Docker
