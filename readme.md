# Stocks Predictions AI Model

## To Run the app do
` pip install -r requirements.txt `
` uvicorn main:app --reload `

# To train a model 
`  python scripts/train.py ` 

## Limited Features

The model may be using insufficient or non-optimal features, such as relying only on residuals (`residual_high`, `residual_low`, `residual_close`) and SMA values. While these features may capture basic price dynamics, they might not fully represent market conditions.

### Additional Features
- Trading volume
- Volatility
- External macroeconomic indicators

### Solutions
- Enrich the feature set with more relevant data, such as:
  - Technical indicators (e.g., RSI, MACD, Bollinger Bands)
  - Fundamental features (e.g., earnings reports)

---

## Noise in the Data

Stock prices are inherently noisy and influenced by external factors (e.g., news, global events, market sentiment) that are not captured in historical data or the model's features.

### Contributing Factors
- Unpredictable external events
- Missing features in the dataset

### Solutions
- Accept that some error is inevitable due to the stochastic nature of stock prices.
- Focus on optimizing MAE within reasonable limits rather than aiming for perfect predictions.
