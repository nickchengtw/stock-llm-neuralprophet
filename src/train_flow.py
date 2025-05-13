import pandas as pd
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from utils import val_mape

set_log_level("ERROR")

print("Loading data...")
df = pd.read_csv('data/2330_stock_data.csv', parse_dates=True)[['ds', 'y']]
plt = df.plot(x="ds", y="y", figsize=(15, 5))
print("Data loaded and plotted.")

print("Initializing NeuralProphet model...")
m = NeuralProphet()
m.set_plotting_backend("matplotlib")

print("Splitting data into training and validation sets...")
df_train, df_val = m.split_df(df, valid_p=0.2)
print(f"Validation period: {df_val['ds'].iloc[0]} ~ {df_val['ds'].iloc[-1]}")

print("Training the model...")
set_random_seed(0)
metrics = m.fit(df_train, validation_df=df_val)
print("Model training completed.")

print("Creating future dataframe for forecasting...")
df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=7)

print("Generating predictions...")
forecast = m.predict(df_future)

rmse = metrics.iloc[-1]['RMSE_val']
mape = val_mape(df_val, forecast) * 100
print(f"RMSE={rmse:.2f}, MAPE={mape:.2f}%")
