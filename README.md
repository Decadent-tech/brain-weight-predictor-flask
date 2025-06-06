# Brain Weight Predictor

A simple Flask app that predicts brain weight from head size.

## Features
- Input form with validation
- Logging of predictions
- Plot of predictions
- Log summary with filtering
- CSV download
- Flask + Gunicorn + Render deployment

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py


## ✅ `.gitignore`
```txt
__pycache__/
*.pyc
*.pkl
prediction_logs.csv
.env
