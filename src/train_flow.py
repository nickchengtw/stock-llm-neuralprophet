import os
import re
import warnings

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
from neuralprophet import NeuralProphet, set_log_level, set_random_seed

from src.model.utils import val_mape

set_log_level("ERROR")

VALIDATION_PERCENTAGE = 0.2


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


def main():
    print("Loading data...")
    df = pd.read_csv("data/stocks/2330_stock_data_0317.csv", parse_dates=True)[["ds", "y"]]

    print("Initializing NeuralProphet model...")
    m = NeuralProphet()
    rmse, mape = train_and_eval(df, m)
    print(f"RMSE={rmse:.2f}, MAPE={mape:.2f}%")


if __name__ == "__main__":
    main()
