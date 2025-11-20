from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Initialize App
app = FastAPI(title="Credit Risk Inference API", version="1.0")

# Load Model
try:
    model = joblib.load('credit_model.pkl')
    print("Model loaded successfully.")
except:
    print("Model not found. Please run train_model.py first.")

# Define Input Schema
class LoanApplication(BaseModel):
    income: float
    debt_to_income: float
    credit_history_length: int
    num_credit_lines: int
    loan_amount: float

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is Live"}

@app.post("/predict")
def predict_risk(application: LoanApplication):
    """
    Predicts loan default risk based on applicant financial data.
    """
    try:
        # Convert input to DataFrame (expected by XGBoost)
        data = pd.DataFrame([application.dict()])
        
        # Inference
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        result = "High Risk (Deny)" if prediction == 1 else "Low Risk (Approve)"
        
        return {
            "prediction": int(prediction),
            "default_probability": float(probability),
            "recommendation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
