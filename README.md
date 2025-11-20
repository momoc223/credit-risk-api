Real-Time Credit Risk Inference API

Project Overview

This project implements a Machine Learning Microservice for real-time credit scoring. It uses an XGBoost classifier trained on financial data to predict loan default probability. The model is served via a high-performance FastAPI endpoint and containerized with Docker.

Features

XGBoost Model: High-accuracy gradient boosting classifier for risk assessment.

FastAPI: Asynchronous, high-performance web framework for model serving (<50ms latency).

Docker: Fully containerized environment for reproducible deployments.

Pydantic: Strict data validation for API inputs.

How to Run

1. Local Testing

First, install dependencies and train the model:

pip install -r requirements.txt
python train_model.py
python app.py


Visit http://localhost:8000/docs to see the Swagger UI and test the API.

2. Run with Docker (Recommended)

Build and run the container:

docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api


API Usage Example

POST /predict

{
  "income": 65000,
  "debt_to_income": 0.3,
  "credit_history_length": 5,
  "num_credit_lines": 4,
  "loan_amount": 20000
}


Response:

{
  "prediction": 0,
  "default_probability": 0.12,
  "recommendation": "Low Risk (Approve)"
}

