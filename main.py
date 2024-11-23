from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.predict import get_predictions

# FastAPI app
app = FastAPI()

# Request schema
class PredictionRequest(BaseModel):
    symbol: str
    days: int

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