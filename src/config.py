import yaml


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open("stocks.yml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

START_DATE = config["start_date"]
END_DATE = config["end_date"]
MODEL_NAME = config["model_name"]
RAG_STOCKS = config["stocks"]

STOCKS = {
    str(stock["symbol"]): {
        "stock_name": stock["stock_name"],
        "keywords": stock["keywords"],
    }
    for stock in data["stocks"]
}
