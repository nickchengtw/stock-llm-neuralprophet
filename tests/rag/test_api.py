from src.rag.api import get_model, get_reponse, LLMProvider  # adjust import path

def test_ollama_model_output_is_deterministic():
    prompt = "三句話解釋台積電是什麼？"
    model_name = "cwchang/llama-3-taiwan-8b-instruct"

    # Create two models with the same seed
    model1 = get_model(LLMProvider.OLLAMA, model_name)
    model2 = get_model(LLMProvider.OLLAMA, model_name)

    # Invoke both models
    response1 = get_reponse(LLMProvider.OLLAMA, model1.invoke(prompt))
    response2 = get_reponse(LLMProvider.OLLAMA, model2.invoke(prompt))

    assert response1 == response2
