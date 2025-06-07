from datetime import date, datetime
import os
import csv

import yaml
from retry import retry
from langchain.vectorstores import Chroma

from src.rag.embedding import get_embedding_function
from src.rag.api import get_model, LLMProvider, get_reponse
from src.rag.prompt import get_prompt
from src.rag.utils import generate_date_range

CHROMA_PATH = "chroma"

QUERY_TEMPLATE = "請對所有與{}相關的新聞進行詳細分析，並判斷這些新聞可能會對股票隔天的市場情緒和股價波動造成什麼影響。"

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

def load_stocks():
    with open("stocks.yml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stock_dict = {
        str(stock['symbol']): {
            'stock_name': stock['stock_name'],
            'keywords': stock['keywords']
        }
        for stock in data['stocks']
    }
    return stock_dict


def main():
    # Load YAML config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    START_DATE = config["start_date"]
    END_DATE = config["end_date"]
    MODEL_NAME = config["model_name"]
    
    STOCKS = load_stocks()

    for symbol in config['stocks']:
        data = STOCKS[str(symbol)]
        company = data['stock_name']
        keywords = [str(symbol), data['stock_name']]
        generate_factor(company, START_DATE, END_DATE, MODEL_NAME, keywords)


def generate_factor(company, start_date, end_date, model_name, keywords):
    print(f"Generating factor for {company} from {start_date} to {end_date}")
    for news_date in generate_date_range(start_date, end_date):
        results = query_db(news_date, company, keywords)
        if len(results):
            print(f'Find {len(results)} relevant result')
            factor, explanation = query_rag(company, results, model_name)
            print(factor, explanation)
            print("Saving result")
            append_row_to_csv(f'./data/factors/result_{company}.csv', news_date, factor, explanation, len(results)) # TODO : use config
        else:
            print(f"No relevant news found in the DB at {str(news_date)}.")


def parse_response(response: str) -> tuple:
    cleaned_text = response.strip()
    fluc_factor, explain = cleaned_text.split("\n", maxsplit=1)
    return float(fluc_factor), explain

def filter_results(results, k, keywords):
    def custom_filter(content, metadata, score):
        return any(keyword.lower() in content for keyword in keywords)

    filtered = [
        (doc, score)
        for doc, score in results
        if custom_filter(doc.page_content, doc.metadata, score)
    ]
    # Closest = lowest distance
    return sorted(filtered, key=lambda x: x[1])[:k]


def query_db(news_date, company, keywords):
    metadata_filter = {"publish_at": news_date.strftime("%Y-%m-%d")}
    query_text = QUERY_TEMPLATE.format(company)
    results = db.similarity_search_with_score(query_text, filter=metadata_filter)
    results = filter_results(results, 5, keywords)
    return results


@retry(exceptions=ValueError, tries=10, delay=3)
def query_rag(company: str, results, model_name):
    prompt = get_prompt(company, results)
    print(prompt)

    print("Generating response text")
    model = get_model(LLMProvider.OLLAMA, model_name)
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
