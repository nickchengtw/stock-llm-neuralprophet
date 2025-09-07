from datetime import date, datetime
import os
import csv
import re
import asyncio

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import yaml

from src.rag.embedding import get_embedding_function
from src.rag.api import get_model, get_reponse, LLMProvider
from src.config import START_DATE, END_DATE, MODEL_NAME, RAG_STOCKS, STOCKS, MAX_NEWS_USED, MAX_CHAR_LENGTH, RAG_REF_USED, CHROMA_PATH, PROVIDER


db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


async def main():
    for symbol in RAG_STOCKS:
        data = STOCKS[str(symbol)]
        company = data['stock_name']
        keywords = [str(symbol), data['stock_name']]
        avg_change = data['avg_change']
        await generate_factor(company, symbol, avg_change, START_DATE, END_DATE, MODEL_NAME, keywords)


async def generate_factor(company, symbol, avg_change, start_date, end_date, model_name, keywords):
    print(f"Generating prediction for {company} from {start_date} to {end_date}")
    
    df = pd.read_csv(f'data/stocks/{symbol}_stock_data_0630.csv', parse_dates=True, index_col=1)
    df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

    for i in df.index:
        news_date = date(i.year, i.month, i.day)
        
        filename = f'data/news/news_{news_date}.csv' # TODO : use config
        if not os.path.exists(filename):
            print(f'No news data found for {news_date}')
            continue
        news = get_company_news(filename, keywords)

        df["pct_change"] = df["y"].pct_change() * 100
        df['volume_change'] = df["volume"].diff()  # Daily volume change
        df.fillna(0, inplace=True) # Fill first row NaN

        if len(news):
            print(f'Find {len(news)} relevant news at {news_date}')
            rules = query_db(news, company, keywords)
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
                    news, model_name, rules, company_rules)
                explanation = "OK"
                print(factor, explanation)
                print("Saving result")
                append_row_to_csv(f'./reports/pred_{symbol}.csv', news_date, factor, explanation, len(rules)) # TODO : use config
            except asyncio.TimeoutError as e:
                print(f"LLM timeout at {news_date} for {company}: {e}")
                append_row_to_csv(f'./reports/pred_{symbol}.csv', news_date, None, "TLE", len(rules)) # TODO : use config
            except ValueError as e:
                print(f"Error parsing response at {news_date} for {company}: {e}")
                append_row_to_csv(f'./reports/pred_{symbol}.csv', news_date, None, "REJ", len(rules)) # TODO : use config
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


def parse_response(response: str) -> float:
    # Remove <think>...</think>
    cleaned_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return float(cleaned_text)


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

QUERY_TEMPLATE = """
請扮演一位專業且客觀的股市分析師，根據下方所有與 {company} 相關的新聞內容進行整體分析，評估這些消息綜合而言可能對該公司隔日的市場情緒與股價波動產生的影響。

====================

以下是提供的新聞資訊 :

{context}
"""

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


async def query_rag(company, change_percent, volume_change, foreign_change, avg_change, news, model_name, rules, company_rules):
    prompt = get_prompt(company, change_percent, volume_change, foreign_change, avg_change, news, rules, company_rules)
    if len(prompt) > MAX_CHAR_LENGTH:
        print(f"Prompt length {len(prompt)} exceeds {MAX_CHAR_LENGTH} characters, truncating.")
        prompt = prompt[:MAX_CHAR_LENGTH]
    print(prompt)

    print("Generating response text")
    model = get_model(LLMProvider(PROVIDER), model_name)
    result = await asyncio.wait_for(
        model.ainvoke(prompt),
        timeout=15  # seconds
    )
    response_text = get_reponse(LLMProvider(PROVIDER), result)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}")
    return parse_response(response_text)


PROMPT_TEMPLATE = """
你是一位專業金融分析師。我會提供你 {company} 公司「當日新聞」以及「當日收盤股價資料」。
請你嚴格依照輸出規則，輸出一個浮點數，代表 隔日股價可能的漲跌比例（使用百分比%）。
分析新聞資訊時，請參考提供的「經驗法則」對隔日股價可能的漲跌比例進行判斷。

輸出規則：
- 僅能輸出數值，禁止任何文字、解釋或符號。
- 僅能輸出一個浮點數，範圍必須介於 -10.0 ~ +10.0 之間。
- 上漲輸出正數（例如 2.5），下跌輸出負數（例如 -1.8），沒有明顯傾向輸出 0.0。
- 預測結果應考慮該股票的「平均漲跌幅」，避免預估結果過大或過小。通常預測應落在平均漲跌幅的 ±2 倍區間內，僅在重大消息時才接近 ±10%。

經驗法則：
{rules_text}
{company_rules_text}

=====

範例：

輸入資料：
股價資訊：
漲跌幅：+1.2%
成交量變化：+15萬張
外資買賣超：+8萬張
平均漲跌幅：1.39%

新聞摘要：
台積電宣布擴大先進製程投資，市場信心回升。

輸出：
2.2

-----

輸入資料：
股價資訊：
漲跌幅：-0.8%
成交量變化：-12萬張
外資買賣超：-6萬張
平均漲跌幅：2.06%

新聞摘要：
美國聯準會暗示可能延後降息，投資人情緒轉弱。

輸出：
-2.4

=====

現在開始，請根據我提供的資料直接輸出一個浮點數。

以下是當日收盤股價資料：
漲跌幅：{change_percent}%
成交量變化：{volume_change}萬張
外資買賣超：{foreign_change}萬張
平均漲跌幅：{avg_change}%

以下是當日關於 {company} 公司的新聞：
{context}

"""


def get_prompt(company, change_percent, volume_change, foreign_change, avg_change, news, rules, company_rules):
    context = "\n\n---\n\n".join([doc for doc in news])
    rules_text = "\n".join([doc.page_content for doc, _score in rules])
    company_rules_text = "\n".join(company_rules)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context,
                                    company=company,
                                    change_percent=round(change_percent, 2),
                                    volume_change=round(volume_change/10000, 1),
                                    foreign_change=round(foreign_change/10000, 1),
                                    avg_change=avg_change,
                                    rules_text=rules_text,
                                    company_rules_text=company_rules_text)
    return prompt


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
