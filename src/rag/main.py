import argparse
from datetime import date, datetime
import pandas as pd
import os
import csv

from retry import retry
from langchain.vectorstores import Chroma

from src.rag.embedding import get_embedding_function
from src.rag.api import get_model, LLMProvider, get_reponse
from src.rag.prompt import get_prompt
from src.rag.utils import generate_date_range

CHROMA_PATH = "chroma"

QUERY_TEMPLATE = "請對所有與{}相關的新聞進行詳細分析，並判斷這些新聞可能會對股票隔天的市場情緒和股價波動造成什麼影響。"

# Prepare the DB.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def main():
    company = '台積電'
    
    start_date = date(2024, 9, 7)
    end_date = date(2025, 3, 17)
    for news_date in generate_date_range(start_date, end_date):
        results = query_db(news_date, company)
        if len(results):
            factor, explanation = query_rag(company, results)
            print(factor, explanation)
            print("Saving result")
            append_row_to_csv(f'./data/factors/result_{company}.csv', news_date, factor, explanation, len(results)) # TODO : use config
        else:
            print(f"No relevant news found in the DB at {str(news_date)}.")


def parse_response(response: str) -> tuple:
    cleaned_text = response.strip()
    fluc_factor, explain = cleaned_text.split("\n", maxsplit=1)
    return float(fluc_factor), explain

def query_db(news_date, company):
    # Search the DB.
    metadata_filter = {"publish_at": news_date.strftime("%Y-%m-%d")}
    query_text = QUERY_TEMPLATE.format(company)
    results = db.similarity_search_with_score(query_text, k=5, filter=metadata_filter)
    return results

@retry(exceptions=ValueError, tries=10, delay=3)
def query_rag(company: str, results):
    prompt = get_prompt(company, results)
    print(prompt)

    print("Generating response text")
    model = get_model(LLMProvider.OLLAMA, "cwchang/llama-3-taiwan-8b-instruct") # TODO : use config
    response_text = get_reponse(LLMProvider.OLLAMA, model.invoke(prompt))

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
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
    main()
