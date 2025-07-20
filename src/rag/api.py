import os
from enum import Enum

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from src.config import OLLAMA_BASE_URL

# Load variables from .env file
load_dotenv()


class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


def get_model(provider: LLMProvider, model_name: str, seed: int = 42, temperature: float = 0.0, timeout: float = 10.0):
    """Function to get the appropriate LLM based on the provider."""

    if provider == LLMProvider.OLLAMA:
        return OllamaLLM(base_url=OLLAMA_BASE_URL, model=model_name, seed=seed, temperature=temperature, client_kwargs={"timeout": httpx.Timeout(timeout)})
    elif provider == LLMProvider.OPENAI:
        # Initialize OpenAI LLM with the API key from config
        return ChatOpenAI(
            openai_api_key=os.getenv("API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model=model_name,
            seed=seed,
            temperature=temperature,
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
