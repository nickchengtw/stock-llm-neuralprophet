import os
import re
import warnings
import json
import argparse

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="neuralprophet.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.utilities.data",
    message=re.escape("Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is")
)

import pandas as pd
from sklearn.model_selection import ParameterGrid
from neuralprophet import NeuralProphet, set_log_level, set_random_seed

from src.model.utils import val_mape
from src.model.features import add_stock_price_feature
from src.config import STOCKS

set_log_level("ERROR")

VALIDATION_PERCENTAGE = 0.2
LAG_REGRESSORS = [
    'volume',
    'high_low_diff',
    'MA',
    'foreign',
    'investment',
    'dealer',
    'ratio_over_400_shares',
    'shareholders_400_to_600',
    'shareholders_600_to_800',
    'shareholders_800_to_1000',
    'ratio_over_1000_shares',
]

with open("src/training_flow/standard.json", "r", encoding="utf-8") as file:
    TUNE_STEPS = json.load(file)


def train_and_eval(df, m):
    print(f"Data range: {df['ds'].iloc[0]} ~ {df['ds'].iloc[-1]}")
    df_train, df_val = m.split_df(df, valid_p=VALIDATION_PERCENTAGE)
    print(f"Validation period: {df_val['ds'].iloc[0]} ~ {df_val['ds'].iloc[-1]}")

    print("Training the model...")
    set_random_seed(0)
    metrics = m.fit(df_train, validation_df=df_val)

    print("Creating future dataframe for forecasting...")
    df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=7)

    print("Generating predictions...")
    forecast = m.predict(df_future)

    rmse = metrics.iloc[-1]["RMSE_val"]
    mape = val_mape(df_val, forecast) * 100
    return rmse, mape


def grid_search(df, param_grid):
    results = []
    # Iterate over each combination of hyperparameters
    for params in ParameterGrid(param_grid):
        # Initialize the NeuralProphet model with current hyperparameters
        print(params)
        init_params = {k: v for k, v in params.items() if k in ['yearly_seasonality', 'weekly_seasonality', 'n_lags']}
        print("init params:", init_params)
        m = NeuralProphet(**init_params)
        
        if params.get('use_holidays', False):
            m = m.add_country_holidays("TW")

        columns = ['ds', 'y']
        for lag_name in LAG_REGRESSORS:
            if params.get(lag_name, 0) > 0:
                print(f"Adding lagged regressor: {lag_name} with n_lags={params.get(lag_name)}")
                m.add_lagged_regressor(lag_name, n_lags=params.get(lag_name))
                columns.append(lag_name)
        
        rmse, mape = train_and_eval(df[columns], m)
        results.append({**params, 'RMSE': rmse, 'MAPE': mape})
    return results


def train_flow(symbol, report_root_dir, stock_data_path):
    print("Loading data...")
    df = pd.read_csv(stock_data_path, parse_dates=True)
    df = add_stock_price_feature(df)
    print(df.info())
    
    optimal_params = {}
    for step_name, step in TUNE_STEPS.items():
        param_grid = {**optimal_params, **step['params']}
        print(f"Grid search: {param_grid}")
        results = grid_search(df, param_grid)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="MAPE")
        print(results_df.head())
        results_df.to_csv(f'{report_root_dir}/{symbol}/{step_name}.csv')
        
        best_candidates = results_df.sort_values(by="MAPE")[step['candidate_cols']].head(step['top_n']).to_dict(orient="list")
        print('best_candidates:', best_candidates)
        optimal_params = best_candidates
        
        with open(f"{report_root_dir}/{symbol}/{step_name}.json", "w") as json_file:
            json.dump(optimal_params, json_file, indent=4)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read stock data from a file')
    parser.add_argument('filepath', type=str, help='Path to the stock data file')

    # Parse arguments
    args = parser.parse_args()
    REPORT_ROOT_DIR = args.filepath

    for symbol in STOCKS.keys():
        print(f"Training flow for {symbol}")
        os.mkdir(f'{REPORT_ROOT_DIR}/{symbol}')
        train_flow(symbol, REPORT_ROOT_DIR, f'data/stocks/{symbol}_stock_data_0630.csv')

if __name__ == "__main__":
    main()
