import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Generate Synthetic Credit Data (Since we can't upload private bank data)
def generate_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'income': np.random.normal(50000, 15000, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
        'credit_history_length': np.random.randint(1, 30, n_samples),
        'num_credit_lines': np.random.randint(0, 15, n_samples),
        'loan_amount': np.random.normal(15000, 5000, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Logic: High debt + Low Income = Default (1)
    df['default_probability'] = (
        (df['debt_to_income'] * 0.6) + 
        (df['loan_amount'] / df['income'] * 0.4) - 
        (df['credit_history_length'] * 0.01)
    )
    df['target'] = (df['default_probability'] > 0.4).astype(int)
    return df.drop(columns=['default_probability'])

print("Generating synthetic data...")
df = generate_data()

X = df.drop(columns=['target'])
y = df['target']

# 2. Train XGBoost Model
print("Training XGBoost Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train, y_train)

# 3. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.4f}")

# 4. Save Model
joblib.dump(model, 'credit_model.pkl')
print("Model saved to credit_model.pkl")
