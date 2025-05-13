from sklearn.metrics import mean_absolute_percentage_error

# Calculate MAPE using true and predicted values
def val_mape(df_val, forecast):
    return mean_absolute_percentage_error(df_val['y'], forecast[forecast['ds'].isin(df_val['ds'])]['yhat1'])