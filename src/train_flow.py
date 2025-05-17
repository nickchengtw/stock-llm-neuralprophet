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

TUNE_STEPS = {
    "default": {
        "params": {},
        "candidate_cols": [],
        "top_n": 0
    },
    "add_period": {
        "params": {
            "yearly_seasonality": [True],
            "weekly_seasonality": [True],
        },
        "candidate_cols": ['yearly_seasonality', 'weekly_seasonality'],
        "top_n": 1
    },
    "add_ar": {
        "params": {
            'n_lags': range(1, 16)
        },
        "candidate_cols": ['n_lags'],
        "top_n": 4
    },
    "tune_close": {
        "params": {
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'use_holidays': [True, False],
        },
        "candidate_cols": ['yearly_seasonality', 'weekly_seasonality', 'n_lags', 'use_holidays'],
        "top_n": 1
    },
    "lag_vol_price": {
        "params": {
            'volume': [0, 5, 10],
            'high_low_diff': [0, 5, 10],
            'MA': [0, 5, 10]
        },
        "candidate_cols": ['yearly_seasonality', 'weekly_seasonality', 'n_lags', 'use_holidays', 'volume', 'high_low_diff', 'MA'],
        "top_n": 1
    },
    "lag_inst": {
        "params": {
            'foreign': [1, 3, 5],
            'investment': [1, 3, 5],
            'dealer': [1, 3, 5]
        },
        "candidate_cols": [
            'yearly_seasonality',
            'weekly_seasonality',
            'n_lags',
            'use_holidays',
            'volume',
            'high_low_diff',
            'MA',
            'foreign',
            'investment',
            'dealer'
        ],
        "top_n": 1
    },
    "lag_share": {
        "params": {
            'ratio_over_400_shares': [0, 5],
            'shareholders_400_to_600': [0, 5],
            'shareholders_600_to_800': [0, 5],
            'shareholders_800_to_1000': [0, 5],
            'ratio_over_1000_shares': [0, 5],
        },
        "candidate_cols": [
            'yearly_seasonality',
            'weekly_seasonality',
            'n_lags',
            'use_holidays',
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
        ],
        "top_n": 1
    },
}

def train_flow(stock_data_path):
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
        results_df.to_csv(f'reports/{step_name}.csv')
        
        best_candidates = results_df.sort_values(by="MAPE")[step['candidate_cols']].head(step['top_n']).to_dict(orient="list")
        print('best_candidates:', best_candidates)
        optimal_params = best_candidates
        with open(f"reports/{step_name}.json", "w") as json_file:
            json.dump(optimal_params, json_file, indent=4)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read stock data from a file')
    parser.add_argument('filepath', type=str, help='Path to the stock data file')

    # Parse arguments
    args = parser.parse_args()
    file_path = args.filepath

    train_flow(file_path)

if __name__ == "__main__":
    main()
