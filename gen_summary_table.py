import os
import pandas as pd

def read_csvs_with_folder_name(root_dir, target_filename):
    dataframes = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            file_path = os.path.join(dirpath, target_filename)
            try:
                df = pd.read_csv(file_path)
                # Get the folder name (the last part of dirpath)
                folder_name = os.path.basename(dirpath)
                dataframes.append((folder_name, df))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return dataframes

table_index = ['default']
summary_table = {}
dfs = read_csvs_with_folder_name("reports/", "default.csv")
for symbol, df in dfs:
    top_mape = round(df.loc[0, 'MAPE'], 2)
    summary_table[symbol] = [top_mape]

for name in ['add_period', 'add_ar', 'tune_close', 'lag_vol_price', 'lag_inst', 'lag_share']:
    table_index.append(name)
    dfs = read_csvs_with_folder_name("reports/", f"{name}.csv")
    for symbol, df in dfs:
        top_mape = round(df.loc[0, 'MAPE'], 2)
        summary_table[symbol].append(top_mape)

table_index.append('with_future')
dfs = read_csvs_with_folder_name("reports/", "lag_share.csv")
for symbol, _ in dfs:
    print(symbol)
    df = pd.read_csv(f"reports/{symbol}/final_{symbol}.csv")
    top_mape = round(df.loc[0, 'MAPE'], 2)
    summary_table[symbol].append(top_mape)

summary_df = pd.DataFrame(summary_table, index=table_index).T
summary_df['is_lower'] = summary_df['with_future'] < summary_df['lag_share']
print(summary_df)
