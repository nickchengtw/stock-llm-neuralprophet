import os
import json
import pandas as pd

from src.config import STOCKS

with open("src/training_flow/all_step_by_step.json", "r", encoding="utf-8") as file:
    TUNE_STEPS = json.load(file)


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


def main():
    step_names = TUNE_STEPS.keys()
    print(step_names)
    summary_table = {symbol: [] for symbol in STOCKS.keys()}
    for name in step_names:
        dfs = read_csvs_with_folder_name("reports/", f"{name}.csv")
        for symbol, df in dfs:
            top_mape = round(df.loc[0, 'MAPE'], 2)
            summary_table[symbol].append(top_mape)

    summary_df = pd.DataFrame(summary_table, index=step_names).T.sort_index()
    print(summary_df)
    summary_df.to_csv("reports/summary.csv")

if __name__ == "__main__":
    main()
