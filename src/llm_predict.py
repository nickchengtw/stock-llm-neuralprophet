from datetime import date, datetime, timedelta
import os
import asyncio
import time

import pandas as pd
from langchain_community.vectorstores import Chroma

from src.rag.embedding import get_embedding_function
from src.rag.retrieval import query_db, get_company_rules
from src.config import MODEL_NAME, STOCKS, MAX_NEWS_USED, CHROMA_PATH, LLM_CALL_DELAY
from src.llm_pred import query_rag, filter_news, clean_news


db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


def process_company_news(news, keywords):
    news = filter_news(news, MAX_NEWS_USED, keywords)[-MAX_NEWS_USED:] # Later news are more relevant
    news = [clean_news(doc) for doc in news]
    return news


async def predict_next_day(symbol, news_date, news_list, df):
    data = STOCKS[str(symbol)]
    company = data['stock_name']
    keywords = [str(symbol), data['stock_name']]
    avg_change = data['avg_change']

    news = process_company_news(news_list, keywords)
    df["pct_change"] = df["y"].pct_change() * 100
    df['volume_change'] = df["volume"].diff()  # Daily volume change
    df.fillna(0, inplace=True) # Fill first row NaN

    if len(news):
        print(f'Find {len(news)} relevant news at {news_date}')
        rules = query_db(news, company, keywords)
        
        # Add company specific rules
        company_rules = get_company_rules(company)
        
        print(f'{len(rules)} rules found {rules}')
        print(f'{len(company_rules)} company rules found {company_rules}')
        try:
            factor = await query_rag(
                company,
                df.loc[datetime(news_date.year, news_date.month, news_date.day), "pct_change"],
                df.loc[datetime(news_date.year, news_date.month, news_date.day), "volume_change"],
                df.loc[datetime(news_date.year, news_date.month, news_date.day), "foreign"],
                avg_change,
                news, MODEL_NAME, rules, company_rules)
            
            time.sleep(LLM_CALL_DELAY) # Avoid LLM rate limit
            
            explanation = "OK"
            print(factor, explanation)
            print("Saving result")
        except asyncio.TimeoutError as e:
            print(f"LLM timeout at {news_date} for {company}: {e}")
        except ValueError as e:
            print(f"Error parsing response at {news_date} for {company}: {e}")
    else:
        print(f"No relevant news found in the DB at {str(news_date)}.")
        factor = None
    return factor


def get_next_day_prediction(news_date, df, factor):
    current_price = df.loc[datetime(news_date.year, news_date.month, news_date.day), "y"]
    prediction = current_price * (factor/100 + 1)
    return prediction


async def main():
    symbol = "2317"
    news_date = date(2025, 6, 30)

    print(f"Loading stock data for {symbol}...")
    df = pd.read_csv(f'data/stocks/{symbol}_stock_data_0630.csv', parse_dates=True, index_col=1)
    df = df[(df.index.date >= news_date-timedelta(days=2)) & (df.index.date <= news_date)]

    filename = f'data/news/news_{news_date}.csv' # TODO : use config
    if not os.path.exists(filename):
        print(f'No news data found for {news_date}')
        return
    news_list = pd.read_csv(filename)['content'].to_list()

    print("Predicting next day factor...")
    factor = await predict_next_day(symbol, news_date, news_list, df)    
    
    next_day = news_date + timedelta(days=1)
    prediction = get_next_day_prediction(news_date, df, factor)
    print(f"Next day: {next_day}, Prediction: {prediction}")


if __name__ == "__main__":
    asyncio.run(main())
