from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
請對以下所有與{company}相關的新聞進行詳細分析，並判斷這些新聞可能會對股票隔天的市場情緒和股價波動造成什麼影響。
輸出共包含兩行資訊，兩行資訊使用換行字元分隔，輸出需嚴格按照此格式，且不產生其他額外的文字。
在第一行輸出一個介於 -1 到 1 之間的實數值，表示新聞對市場情緒的影響，-1 表示市場情緒負面，股票可能會下跌；1 表示市場情緒正面，股票可能會上漲；0 則表示沒有明顯傾向。
第二行用簡短的句子，說明產生此判斷的原因。

以下是提供的新聞資訊 :
{context}
"""


def get_prompt(company, results):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, company=company)
    return prompt
