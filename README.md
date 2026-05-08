# Complete ML Pipeline for SMS Spam Classification

A reproducible, stage-based machine learning pipeline for classifying SMS messages as spam or ham. The project uses DVC for pipeline orchestration, data/artifact versioning, and experiment tracking.

## Overview

This repository covers the full ML lifecycle:

- Data ingestion from a remote CSV source.
- Text preprocessing (cleaning, tokenization, stopword removal, stemming).
- TF-IDF feature engineering.
- Random Forest training.
- Evaluation with tracked metrics.

The pipeline is implemented in modular Python scripts under `src/` and orchestrated through `dvc.yaml`.

## Pipeline Stages

| Stage | Script | Purpose | Main Outputs |
|---|---|---|---|
| `data_ingestion` | `src/data_ingestion.py` | Download and split source data into train/test | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `src/data_preprocessing.py` | Encode labels, remove duplicates, preprocess text | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| `feature_engineering` | `src/feature_engineering.py` | Convert text to TF-IDF vectors | `data/processed/train_tfidf.csv`, `data/processed/test_tfidf.csv` |
| `model_building` | `src/model_building.py` | Train Random Forest classifier | `models/model.pkl` |
| `model_evaluation` | `src/model_evaluation.py` | Evaluate model and log experiment metrics | `reports/metrics.json`, `dvclive/metrics.json`, `dvclive/plots/` |

## Parameters

Pipeline settings are controlled from `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.35

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 25
  random_state: 2
```

## Experiment Tracking with DVC

This project tracks experiments in two ways:

- Stage orchestration and dependencies via `dvc.yaml`.
- Metric and parameter logging via DVCLive (`live.log_metric`, `live.log_params`).

Useful commands:

```bash
dvc repro                  # Re-run pipeline stages that changed
dvc exp run                # Run and record a new experiment
dvc exp show               # Compare experiments
dvc exp apply <exp-name>   # Apply an experiment result to workspace
```

## Remote Storage (AWS S3)

DVC remote is configured in `.dvc/config`:

- Remote name: `dvcstore`
- Remote URL: `s3://s3-dvc-proj`

Common remote operations:

```bash
dvc push   # Upload tracked data/artifacts to S3
dvc pull   # Download tracked data/artifacts from S3
```

Note: AWS credentials are not stored in this repository. Configure them locally using AWS CLI or environment variables.

## Latest Metrics

From `reports/metrics.json` in the current workspace:

- Accuracy: `0.9463`
- Precision: `0.8368`
- Recall: `0.7605`
- AUC: `0.9410`

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- nltk
- PyYAML
- DVC (`dvc[s3]`)
- DVCLive

## Project Structure

```text
Compelte-ML-PipeLine/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
├── models/
│   └── model.pkl
├── reports/
│   └── metrics.json
├── dvclive/
│   ├── metrics.json
│   ├── params.yaml
│   └── plots/
├── logs/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository.

```bash
git clone <your-repo-url>
cd Compelte-ML-PipeLine
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the complete pipeline:

```bash
dvc repro
```

If you want to run as an experiment entry:

```bash
dvc exp run
```

## Data Source

The ingestion stage reads from:

`https://raw.githubusercontent.com/dhanushpavann/Datasets/refs/heads/main/spam.csv`

## Logging

Each stage writes logs to `logs/`:

- `logs/data_ingestion.log`
- `logs/data_preprocessing.log`
- `logs/feature_engineering.log`
- `logs/model_building.log`
- `logs/model_evaluation.log`

## Notes

- `src/data_preprocessing.py` downloads NLTK resources (`stopwords`, `punkt`) on first run, so internet access is required at least once.
- If `dvc push` or `dvc pull` fails, verify AWS credentials and S3 permissions for `s3://s3-dvc-proj`.

## License

This project is licensed under the terms in `LICENSE`.
