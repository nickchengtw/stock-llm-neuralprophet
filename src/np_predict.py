import os
import re
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from neuralprophet import NeuralProphet

from src.model.features import add_stock_price_feature

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning, module="neuralprophet.*")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.utilities.data",
    message=re.escape(
        "Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is"
    ),
)


def load_model(model_path: str) -> NeuralProphet:
    # Cannot use NeuralProphet's load function because it use weights_only=True by default
    m: NeuralProphet = torch.load(
        model_path, weights_only=False, map_location="cpu"
    )
    m.restore_trainer(accelerator="cpu")
    return m


def get_model_columns(m: NeuralProphet):
    lagged_regressors_cols = [reg for reg in  m.config_lagged_regressors]
    return ["ds", "y"] + lagged_regressors_cols


def predict_next_day(df, m):
    df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=1)
    # print(df_future.tail(15))
    # print(m)
    forecast = m.predict(df_future)
    return forecast


def get_next_day_prediction(forecast):
    return forecast["ds"].iloc[-1], forecast["yhat1"].iloc[-1]


def main():
    symbol = "2317"
    
    print("Loading model")
    m = load_model(f"reports/{symbol}/lag_share.np")
    cols = get_model_columns(m)
    print(cols)

    print("Loading data...")
    stock_data_path = f"data/stocks/{symbol}_stock_data_0630.csv"
    df = pd.read_csv(stock_data_path, parse_dates=True, index_col=0)
    df = add_stock_price_feature(df)
    df = df[cols]
    print(f"Data columns after feature addition: {df.columns.tolist()}")

    print("Generating predictions")
    forecast = predict_next_day(df, m)
    print(forecast)
    next_day, prediction = get_next_day_prediction(forecast)
    forecast[forecast["ds"] >= datetime(2024, 12, 18)][["ds", "y", "yhat1"]].plot(x="ds", y=["y", "yhat1"], figsize=(15, 5))
    plt.show()
    
    print(f"Next day: {next_day}, Prediction: {prediction}")


if __name__ == "__main__":
    main()
