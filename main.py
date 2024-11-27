from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.predict import get_predictions, get_test_date_range, get_predictions_and_mse, subtract_business_days
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class PredictionRequest(BaseModel):
    symbol: str
    days: int

# Pydantic model for request validation
class MSERequest(BaseModel):
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/predict/")
def predict_stock_prices(request: PredictionRequest):
    """
    Endpoint to predict stock prices.
    """
    symbol = request.symbol
    n_days = request.days

    if n_days <= 0:
        raise HTTPException(status_code=400, detail="Number of days must be greater than 0.")
    
    try:
        predictions = get_predictions(symbol=symbol, n_days=n_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    return {"symbol": symbol, "predictions": predictions}

@app.post("/get-predictions-and-mse/")
def get_predictions_and_mse_endpoint(request: MSERequest):
    """
    Endpoint to get predictions and MSE/MAE scores for a stock symbol and date range.

    Args:
        request (MSERequest): The request containing the stock symbol and date range.

    Returns:
        dict: Predictions, actual data, and MSE/MAE scores.
    """
    symbol = request.symbol
    start_date = request.start_date
    end_date = request.end_date

    range = get_test_date_range()

    if start_date is None or end_date is None:
        # Default to last 30 days if no range is provided
        
        end_date = range["end_date"]

        # Ensure `end_date` is a string
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")
        elif not isinstance(end_date, str):
            end_date = str(end_date)

        print("end_date: => ------- ", end_date)
        
        # Calculate start_date as 10 business days before end_date
        start_date = subtract_business_days(end_date, 10)
        print("start_date (10 business days before end_date): => ------- ", start_date)


    try:
        # Call the get_predictions_and_mse function
        result = get_predictions_and_mse(symbol=symbol, start_date=start_date, end_date=end_date)

        # Convert DataFrames to JSON-compatible format
        result["predictions"] = result["predictions"].reset_index().to_dict(orient="records")
        result["actual"] = result["actual"].reset_index().to_dict(orient="records")

        return {
            "symbol": symbol,
            "start_date": start_date,
            "start_date_limit": range["start_date"],
            "end_date_limit": range["end_date"],
            "end_date": end_date,
            "predictions": result["predictions"],
            "actual": result["actual"],
            "mse_scores": result["mse_scores"],
            "mae_scores": result["mae_scores"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction and evaluation: {str(e)}")

@app.get("/test-date-range/")
def get_test_date_ranges():
    """
    Endpoint to get the start and end dates of the test data range.
    """
    dates = get_test_date_range()
    return dates