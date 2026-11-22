# Sri Lanka Food Price Predictor

An interactive machine learning application that predicts food prices across Sri Lankan markets using historical data.

## Dataset

- **Source:** [WFP Food Prices – Sri Lanka](https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka)
- **Period:** January 2004 – September 2025
- **Commodities:** 42
- **Markets:** 44
- **Target:** Food price (LKR)

## Models Evaluated

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble
- Hybrid Random Forest + ANN

### Model Performance

| Model           |    MAE |   RMSE |     R² |
| --------------- | -----: | -----: | -----: |
| Stacking        | 120.46 | 174.04 | 0.9290 |
| XGBoost         | 124.36 | 180.67 | 0.9235 |
| CatBoost        | 125.68 | 181.68 | 0.9226 |
| LightGBM        | 126.97 | 177.38 | 0.9262 |
| Random Forest   | 134.51 | 202.19 | 0.9041 |
| Hybrid RF + ANN | 142.57 | 214.45 | 0.8922 |

The Stacking Ensemble was selected for the prediction application based on its overall test performance.

_Note: The standalone models were trained through 2024, while the Stacking and Hybrid base models were trained through 2023._

## Application Features

- **Overview:** Dataset statistics and price summaries.
- **Price Estimator:** Predict prices based on selected inputs.
- **Market Trends:** Explore historical commodity prices.
- **Model Evaluation:** Compare model performance.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Streamlit · Altair

## Run Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sl_food_prices

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Structure

```text
sl_food_prices/
├── data/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

## Disclaimer

Predictions are estimates based on historical data and should not be considered guaranteed future market prices.

## Acknowledgement

Dataset provided by the World Food Programme (WFP) through the Humanitarian Data Exchange (HDX).
