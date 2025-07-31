import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, RootModel
from typing import List, Dict, Any, Union

# --- Configuration ---
MLFLOW_RUN_ID = "428641126348462805"
MODEL_ID = "m-dcd9213023de4ed6a932eee861e6259d"
MODEL_ARTIFACT_SUBPATH = os.path.join("models", MODEL_ID, "artifacts")

model_pipeline = None

# --- Initialize FastAPI app ---
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using an MLflow-logged model.",
    version="1.0.0"
)

# --- Load model on startup ---
@app.on_event("startup")
def load_model():
    global model_pipeline
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(current_dir, "mlruns")
        model_path = os.path.join(base_path, MLFLOW_RUN_ID, MODEL_ARTIFACT_SUBPATH)
        model_uri = f"file:///{model_path}".replace("\\", "/")

        print(f"Loading model from: {model_uri}")
        model_pipeline = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model_pipeline = None

# --- Single Input Schema ---
class TelcoCustomer(BaseModel):
    customerID: str = Field(..., example="1234-ABCD")
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=24)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=85.0)
    TotalCharges: Union[float, str] = Field(..., example=2040.0)

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "1234-ABCD",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 2040.0
            }
        }

# --- RootModel for batch input ---
class TelcoCustomerBatch(RootModel[List[TelcoCustomer]]):
    class Config:
        json_schema_extra = {
            "example": [
                {
                    "customerID": "1234-ABCD",
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 24,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 85.0,
                    "TotalCharges": 2040.0
                },
                {
                    "customerID": "5678-WXYZ",
                    "gender": "Female",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "Yes",
                    "tenure": 12,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Mailed check",
                    "MonthlyCharges": 55.3,
                    "TotalCharges": 662.5
                }
            ]
        }

# --- Preprocessing helper ---
def preprocess_input_data(input_data_df: pd.DataFrame) -> pd.DataFrame:
    if 'TotalCharges' in input_data_df.columns:
        input_data_df['TotalCharges'] = pd.to_numeric(input_data_df['TotalCharges'], errors='coerce')
        input_data_df['TotalCharges'] = input_data_df['TotalCharges'].fillna(0)
    return input_data_df

# --- Single Prediction Endpoint ---
@app.post("/predict_churn", response_model=Dict[str, Any], summary="Predict churn for a single customer")
async def predict_churn(customer: TelcoCustomer):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    input_df = pd.DataFrame([customer.model_dump()])
    input_df = preprocess_input_data(input_df)

    try:
        prediction = model_pipeline.predict(input_df)[0]
        churn_status = "Churned" if prediction == 1 else "Not Churned"
        return {"prediction": int(prediction), "churn_status": churn_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Batch Prediction Endpoint ---
@app.post("/predict_churn_batch", response_model=Dict[str, Any], summary="Predict churn for a batch of customers")
async def predict_churn_batch(customers: TelcoCustomerBatch):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    input_df = pd.DataFrame([c.model_dump() for c in customers.root])
    input_df = preprocess_input_data(input_df)

    try:
        predictions = model_pipeline.predict(input_df)
        results = [
            {
                "customer_index": i,
                "prediction": int(pred),
                "churn_status": "Churned" if pred == 1 else "Not Churned"
            }
            for i, pred in enumerate(predictions)
        ]
        return {"batch_predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

# --- Health Check ---
@app.get("/")
async def read_root():
    return {"message": "Telco Churn Prediction API is running."}




# To run this API locally using uvicorn (e.g., in your terminal):
# 1. Save this code as 'app.py'
# 2. Make sure your MLflow UI is running in a separate terminal:
#    mlflow ui --port 5000
# 3. Ensure your Telco Churn model is registered in MLflow and set to 'Production' stage.
# 4. In the directory where 'app.py' is saved, run:
#    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# 5. Open your web browser to http://127.0.0.1:8000/docs to test the API.