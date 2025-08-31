from datetime import date, datetime
import os
import csv
import re
import asyncio

import pandas as pd
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import yaml

from src.rag.embedding import get_embedding_function
from src.rag.api import get_model, LLMProvider, get_reponse
from src.rag.prompt import get_prompt
from src.rag.utils import generate_date_range
from src.config import START_DATE, END_DATE, MODEL_NAME, RAG_STOCKS, STOCKS, MAX_NEWS_USED

CHROMA_PATH = "chroma"

MAX_CHAR_LENGTH = 7500
RAG_REF_USED = 15

QUERY_TEMPLATE = """
請扮演一位專業且客觀的股市分析師，根據下方所有與 {company} 相關的新聞內容進行整體分析，評估這些消息綜合而言可能對該公司隔日的市場情緒與股價波動產生的影響。

====================

以下是提供的新聞資訊 :

{context}
"""

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


async def main():
    for symbol in RAG_STOCKS:
        data = STOCKS[str(symbol)]
        company = data['stock_name']
        keywords = [str(symbol), data['stock_name']]
        await generate_factor(company, START_DATE, END_DATE, MODEL_NAME, keywords)


async def generate_factor(company, start_date, end_date, model_name, keywords):
    print(f"Generating factor for {company} from {start_date} to {end_date}")
    for news_date in generate_date_range(start_date, end_date):
        filename = f'data/news/news_{news_date}.csv' # TODO : use config
        if not os.path.exists(filename):
            print(f'No news data found for {news_date}')
            continue
        news = get_company_news(filename, keywords)
        
        if len(news):
            print(f'Find {len(news)} relevant news at {news_date}')
            rules = query_db(news, company, keywords)
            company_rules = get_company_rules(company)
            print(f'{len(rules)} rules found {rules}')
            print(f'{len(company_rules)} company rules found {company_rules}')
            try:
                factor, explanation = await query_rag(company, rules, company_rules, news, model_name)
                print(factor, explanation)
                print("Saving result")
                append_row_to_csv(f'./data/factors/result_{company}.csv', news_date, factor, explanation, len(rules)) # TODO : use config
            except asyncio.TimeoutError as e:
                print(f"LLM timeout at {news_date} for {company}: {e}")
            except ValueError as e:
                print(f"Error parsing response at {news_date} for {company}: {e}")
        else:
            print(f"No relevant news found in the DB at {str(news_date)}.")


def get_company_news(filename, keywords):
    news = pd.read_csv(filename)['content'].to_list()
    news = filter_results(news, MAX_NEWS_USED, keywords)[-MAX_NEWS_USED:] # Later news are more relevant
    news = [clean_news(doc) for doc in news]
    return news


def clean_news(doc):
    # Remove line breaks and extra spaces
    pattern = r'https?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'
    cleaned_text = re.sub(pattern, '', doc).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def parse_response(response: str) -> tuple:
    # Remove <think>...</think>
    cleaned_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    fluc_factor, explain = cleaned_text.split("\n", maxsplit=1)
    return float(fluc_factor), explain


def filter_results(results, k, keywords):
    def custom_filter(content):
        return any(keyword.lower() in content for keyword in keywords)

    filtered = [
        doc
        for doc in results
        if custom_filter(doc)
    ]
    # Closest = lowest distance
    return filtered


def get_query(company, results):
    context_text = "\n\n---\n\n".join([doc for doc in results])
    prompt_template = ChatPromptTemplate.from_template(QUERY_TEMPLATE)
    prompt = prompt_template.format(context=context_text, company=company)
    return prompt


def query_db(news, company, keywords):
    query_text = get_query(company, news)
    results = db.similarity_search_with_score(query_text, k=RAG_REF_USED)
    return results


def get_company_rules(company):
    # TODO AD Hoc
    with open("company_stock_rules.yaml", "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    company_rules = rules.get(company)
    positive_rules = [f'如果新聞中有關於 {company} {i}的消息，則隔日股價可能上漲。' for i in company_rules.get("positive")]
    negative_rules = [f'如果新聞中有關於 {company} {i}的消息，則隔日股價可能下跌。' for i in company_rules.get("negative")]
    return positive_rules + negative_rules


async def query_rag(company: str, rules, company_rules, news, model_name):
    prompt = get_prompt(company, rules, company_rules, news)
    if len(prompt) > MAX_CHAR_LENGTH:
        print(f"Prompt length {len(prompt)} exceeds {MAX_CHAR_LENGTH} characters, truncating.")
        prompt = prompt[:MAX_CHAR_LENGTH]
    print(prompt)

    print("Generating response text")
    model = get_model(LLMProvider.OLLAMA, model_name)
    result = await asyncio.wait_for(
        model.ainvoke(prompt),
        timeout=15  # seconds
    )
    response_text = get_reponse(LLMProvider.OLLAMA, result)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}")
    return parse_response(response_text)


def append_row_to_csv(file_path, news_date: date, real_number, text, news_count: int):
    news_date = news_date.strftime('%Y-%m-%d')

    # Get the current timestamp for 'updated_time'
    updated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define the new row to be added
    new_row = [news_date, real_number, text, updated_time, news_count]

    # Check if the file exists to determine if headers are needed
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # If the file does not exist or is empty, write the header
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(['date', 'factor', 'explanation', 'updated_time', 'news_count'])

        # Append the new row
        writer.writerow(new_row)



if __name__ == "__main__":
    asyncio.run(main())
