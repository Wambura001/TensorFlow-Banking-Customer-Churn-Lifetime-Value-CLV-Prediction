import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(
    title="Bank Churn & CLV Prediction API",
    description="Real-time multi-task TensorFlow scoring service",
    version="1.0"
)

# 2. Load saved artifacts once at startup (keeps latency under ~30ms)
model = tf.keras.models.load_model("notebooks/churn_clv_model.keras")
preprocessor = joblib.load("notebooks/bank_preprocessor.joblib")

# 3. Define strict input schema using Pydantic
class CustomerProfile(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_customer_risk(customer: CustomerProfile):
    # Convert incoming JSON object to a single-row DataFrame
    input_df = pd.DataFrame([customer.model_dump()])
    
    # Apply the exact scaling and encoding learned during training
    processed_features = preprocessor.transform(input_df)
    
    # Run TensorFlow inference
    pred_churn, pred_clv = model.predict(processed_features, verbose=0)
    
    churn_prob = float(pred_churn[0][0])
    clv_value = float(max(pred_clv[0][0], 0.0))
    revenue_at_risk = churn_prob * clv_value
    
    # Assign retention action tier
    if churn_prob >= 0.50 and clv_value >= 1200:
        tier = "1. VIP Rescue"
    elif churn_prob >= 0.50:
        tier = "2. Automated Win-Back"
    elif clv_value >= 1200:
        tier = "3. VIP Nurture"
    else:
        tier = "4. Standard Monitor"

    return {
        "churn_probability": round(churn_prob, 4),
        "predicted_clv_usd": round(clv_value, 2),
        "revenue_at_risk_usd": round(revenue_at_risk, 2),
        "priority_tier": tier
    }
