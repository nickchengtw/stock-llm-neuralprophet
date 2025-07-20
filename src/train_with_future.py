import os
import re
import warnings
import json
import argparse
import yaml

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
    # df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=7)

    print("Generating predictions...")
    forecast = m.predict(df)

    rmse = metrics.iloc[-1]["RMSE_val"]
    mape = val_mape(df_val[:-1], forecast) * 100
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
        
        columns = columns + ['factor']
        m.add_future_regressor('factor')
        
        rmse, mape = train_and_eval(df[columns], m)
        results.append({**params, 'RMSE': rmse, 'MAPE': mape})
    return results


def train_with_future(symbol, name, stock_data_path):
    print("Loading data...")
    df = pd.read_csv(stock_data_path, parse_dates=['ds'])
    df = add_stock_price_feature(df)
    df.info()
    
    print("Loading factors")
    llm_factor = pd.read_csv(f'data/factors/result_{name}.csv', parse_dates=True, index_col=0)
    llm_factor.info()
    
    llm_factor = llm_factor[~llm_factor.index.duplicated(keep='first')]
    llm_factor = llm_factor[['factor']]
    llm_factor.info()
    
    df_merged = df.merge(llm_factor, how='left', left_on='ds', right_index=True)
    # df_merged[df_merged['factor'].isnull()]
    
    df_merged['factor'] = df_merged['factor'].fillna(0)
    df_merged.dropna(inplace=True)
    df_merged.info()
    
    # Open and load JSON file
    with open(f'reports/{symbol}/lag_share.json', 'r', encoding='utf-8') as file:
        optimal_params = json.load(file)
    print("Optimal parameters:", optimal_params)
    param_grid = {**optimal_params, 'factor': [True]}
    results = grid_search(df_merged, param_grid)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="MAPE")
    print(results_df.head())
    results_df.to_csv(f'reports/{symbol}/final_{symbol}.csv')


def main():
    # Load YAML from file
    with open('stocks.yml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Extract all symbol and stock_name pairs
    stock_list = [(stock['symbol'], stock['stock_name']) for stock in data['stocks']]

    # Print the result
    for symbol, name in stock_list:
        print('Training with future regressor', symbol, name)
        train_with_future(symbol, name, f'data/stocks/{symbol}_stock_data_0630.csv')

    # symbol = '2330'
    # name = '台積電'
    # train_with_future(symbol, name, f'data/stocks/{symbol}_stock_data_0317.csv')

if __name__ == "__main__":
    main()
