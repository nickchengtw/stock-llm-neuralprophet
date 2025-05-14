import pandas as pd

def add_stock_price_feature(df: pd.DataFrame, ma_window=5):
    df['high_low_diff'] = df['high_price'] - df['low_price']
    df['MA'] = df['y'].rolling(window=ma_window).mean()
    return df.dropna()
