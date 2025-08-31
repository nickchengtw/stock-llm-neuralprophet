import pandas as pd

from src.config import STOCKS


avg_changes = []

for symbol, data in STOCKS.items():
    company = data['stock_name']

    df = pd.read_csv(f'data/stocks/{symbol}_stock_data_0630.csv', parse_dates=['ds'])
    plt = df.plot(x="ds", y="y", figsize=(15, 5))
    # df.info()

    df['change'] = df['y'].diff()
    df.dropna(inplace=True)
    
    avg_price_change = (df['change'] / df['y']).abs().mean()
    print(f"Average price change for {company} ({symbol}): {avg_price_change:.2%}")

    # Save to list
    avg_changes.append({
        "symbol": symbol,
        "company": company,
        "avg_price_change": avg_price_change
    })

# Convert to DataFrame
avg_change_df = pd.DataFrame(avg_changes)

# Format percentage column for display (optional)
avg_change_df['avg_price_change_pct'] = avg_change_df['avg_price_change'].apply(lambda x: f"{x:.2%}")

avg_change_df.sort_values(by='symbol', inplace=True)

avg_change_df.to_csv('reports/stock_stats.csv', index=False)
print(avg_change_df)
