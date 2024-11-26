from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.predict import get_predictions, get_test_date_range, get_predictions_and_mse
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

    if start_date is None or end_date is None:
        # If no date range is provided, use the last 30 days as the default range
        range = get_test_date_range()
        start_date = range["start_date"]

        # Ensure `start_date` is converted to string if it is not already
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        elif not isinstance(start_date, str):
            start_date = str(start_date)

        print("start_date: => ------- ", start_date)
        
        # Parse the date string
        original_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Add 10 days
        new_date = original_date + timedelta(days=10)
        end_date = new_date.strftime("%Y-%m-%d")


    try:
        # Call the get_predictions_and_mse function
        result = get_predictions_and_mse(symbol=symbol, start_date=start_date, end_date=end_date)

        # Convert DataFrames to JSON-compatible format
        result["predictions"] = result["predictions"].reset_index().to_dict(orient="records")
        result["actual"] = result["actual"].reset_index().to_dict(orient="records")

        return {
            "symbol": symbol,
            "start_date": start_date,
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