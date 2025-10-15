import re

import yaml
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from src.config import RAG_REF_USED, CHROMA_PATH
from src.rag.embedding import get_embedding_function


# QUERY_TEMPLATE = """
# 請扮演一位專業且客觀的股市分析師，根據下方所有與 {company} 相關的新聞內容進行整體分析，評估這些消息綜合而言可能對該公司隔日的市場情緒與股價波動產生的影響。

# ====================

# 以下是提供的新聞資訊 :

# {context}
# """

QUERY_TEMPLATE = """
{context}
"""

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

#TODO duplication
def clean_news(doc):
    # Remove line breaks and extra spaces
    pattern = r'https?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'
    cleaned_text = re.sub(pattern, '', doc).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def get_query(company, results):
    context_text = "\n\n---\n\n".join([doc for doc in results])
    prompt_template = ChatPromptTemplate.from_template(QUERY_TEMPLATE)
    prompt = prompt_template.format(context=context_text, company=company)
    return prompt


def query_db(news, company, keywords):
    query_text = get_query(company, news)
    return db.similarity_search_with_score(query_text, k=RAG_REF_USED)


def get_company_rules(company):
    # TODO AD Hoc
    with open("company_stock_rules.yaml", "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    company_rules = rules.get(company)
    positive_rules = [f'如果新聞中有關於 {company} {i}的消息，則隔日股價可能上漲。' for i in company_rules.get("positive")]
    negative_rules = [f'如果新聞中有關於 {company} {i}的消息，則隔日股價可能下跌。' for i in company_rules.get("negative")]
    return positive_rules + negative_rules
