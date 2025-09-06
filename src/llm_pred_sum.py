import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_percentage_error

from src.model.features import add_stock_price_feature
from src.config import STOCKS

mapes = []
for SYMBOL in STOCKS.keys():
    df = pd.read_csv(f'data/stocks/{SYMBOL}_stock_data_0630.csv', parse_dates=['ds'])
    df = add_stock_price_feature(df)
    # df.info()

    llm_factor = pd.read_csv(f'reports/pred_llm_rag_full/pred_{SYMBOL}.csv', parse_dates=True, index_col=0)
    # llm_factor.info()

    llm_factor = llm_factor[~llm_factor.index.duplicated(keep='first')]
    llm_factor = llm_factor[['factor']]
    # llm_factor.info()

    df_merged = df.merge(llm_factor, how='left', left_on='ds', right_index=True)
    df_merged[df_merged['factor'].isnull()]

    df_merged['factor'] = df_merged['factor'].fillna(0)
    # df_merged.info()

    df_merged['pred'] = (df_merged['factor']/100 * df_merged['y'] + df_merged['y']).shift(1)

    df_merged.dropna(inplace=True)

    # df_merged

    val_df = df_merged[df_merged['ds'] >= datetime(2024, 12, 18)]

    # val_df

    mape = round(mean_absolute_percentage_error(val_df['y'], val_df['pred']) * 100, 2)
    print(SYMBOL, mape)
    mapes.append(mape)

df = pd.DataFrame({'symbol': STOCKS.keys(), 'mape': mapes})
df.to_csv('reports/llm_pred_sum.csv', index=False)