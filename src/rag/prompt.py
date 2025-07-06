from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
請扮演一位專業且客觀的股市分析師，根據下方所有與 {company} 相關的新聞內容進行整體分析，評估這些消息綜合而言可能對該公司隔日的市場情緒與股價波動產生的影響。
請將所有新聞內容整合為一個總體結論，禁止逐則分析或引用原文，不要輸出針對其他公司的分析資訊。
輸出必須嚴格包含兩行內容，每行之間使用一個換行字元分隔，嚴禁輸出任何多餘說明、格式標記或原始內容。
第一行請輸出一個介於 0 到 1 之間的實數(例如：0.0、0.3、0.6、1.0)，表示所有新聞對 {company} 股票市場情緒的影響。0 表示市場情緒負面，股票可能會下跌；1 表示市場情緒正面，股票可能會上漲；0.5 則表示中性，無明顯漲跌傾向或資訊無關。
第二行請簡短說明你產出此判斷的主要原因或邏輯依據，僅限一行簡潔文字，不超過一句話。

範例輸出1:
0.85
營收與獲利雙創高點，顯示公司營運狀況良好，預期帶動市場情緒正面。

範例輸出2:
0.1
訂單下修將直接衝擊營收與獲利，引發市場負面情緒。

範例輸出3:
0.4
新產品未引起預期熱度，市場情緒偏保守，短期可能缺乏利多刺激。

====================

以下是提供的新聞資訊 :
{context}
"""


def get_prompt(company, results):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, company=company)
    return prompt
