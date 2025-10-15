from datetime import date, datetime
import os
import csv
import re
import asyncio
import time

import pandas as pd
from langchain.prompts import ChatPromptTemplate
import yaml

from src.rag.embedding import get_embedding_function
from src.rag.api import get_model, get_reponse, LLMProvider
from src.config import START_DATE, END_DATE, MODEL_NAME, RAG_STOCKS, STOCKS, MAX_NEWS_USED, MAX_CHAR_LENGTH, RAG_REF_USED, CHROMA_PATH, PROVIDER


PROMPT_TEMPLATE = """
請協助我清理以下新聞資料，要求如下：
1. 移除所有與新聞無關的廣告文字、行銷用語、推銷連結。
2. 移除明顯的多餘符號，例如多個連續的破折號、亂碼、無意義的標點。
3. 在清理完成後，將新聞內容濃縮到約 500 字以內，保留核心事實與完整事件脈絡。
4. 不要加入主觀判斷、個人評論或推測，僅呈現新聞原本的客觀資訊。

新聞資訊如下：

{context}
"""


def get_summary_prompt(company, news):
    context = "".join([doc for doc in news])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context,
                                    company=company)
    return prompt


async def get_summary(company, news):
    prompt = get_summary_prompt(company, news)
    if len(prompt) > MAX_CHAR_LENGTH:
        print(f"Prompt length {len(prompt)} exceeds {MAX_CHAR_LENGTH} characters, truncating.")
        prompt = prompt[:MAX_CHAR_LENGTH]
    
    model = get_model(LLMProvider(PROVIDER), MODEL_NAME)
    result = await asyncio.wait_for(
        model.ainvoke(prompt),
        timeout=20  # seconds
    )
    response_text = get_reponse(LLMProvider(PROVIDER), result)
    
    return response_text
