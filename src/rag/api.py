import os
from enum import Enum

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Load variables from .env file
load_dotenv()


class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


def get_model(provider: LLMProvider, model_name: str):
    """Function to get the appropriate LLM based on the provider."""

    if provider == LLMProvider.OLLAMA:
        return OllamaLLM(model=model_name)
    elif provider == LLMProvider.OPENAI:
        # Initialize OpenAI LLM with the API key from config
        return ChatOpenAI(
            openai_api_key=os.getenv("API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model=model_name,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def get_reponse(provider: LLMProvider, resp: str):
    """Function to get the appropriate LLM based on the provider."""

    if provider == LLMProvider.OLLAMA:
        return resp
    elif provider == LLMProvider.OPENAI:
        # Initialize OpenAI LLM with the API key from config
        return resp.content[7:-1]
    raise ValueError(f"Unsupported provider: {provider}")

def main():
    prompt_template = ChatPromptTemplate.from_template(
        "何謂台灣股市的三大法人?\n{context}"
    )
    prompt = prompt_template.format(context="")
    print(prompt)

    model = get_model(LLMProvider.OPENAI, "deepseek/deepseek-r1-zero:free")
    resp = get_reponse(LLMProvider.OPENAI, model.invoke(prompt))

    # model = get_model(LLMProvider.OLLAMA, "llama3.2:3b")
    print(resp)


if __name__ == "__main__":
    main()
