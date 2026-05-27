import config
from typing import Any

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI


def get_ollama(model: str | None = None, base_url: str | None = None) -> Any:
    """Return an Ollama client configured from args or `config`."""
    model = model or getattr(config, 'OLLAMA_MODEL', 'llama3')
    base_url = base_url or getattr(config, 'OLLAMA_BASE_URL', None)
    if base_url:
        return Ollama(model=model, base_url=base_url)
    # Fallback to default constructor (may error if network required)
    return Ollama(model=model)


def get_chat_openai(temperature: float = 0.0, model: str = "gpt-4o", max_tokens: int | None = None) -> Any:
    """Return a ChatOpenAI instance with sane defaults."""
    kwargs = {"temperature": temperature, "model": model}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def invoke_llm(llm: Any, prompt: Any) -> Any:
    """Invoke given LLM with `prompt`.

    Tries common methods in order: `invoke`, `predict`, `generate`.
    Returns the raw LLM response object/string.
    """
    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    if hasattr(llm, "predict"):
        return llm.predict(prompt)
    if hasattr(llm, "generate"):
        return llm.generate(prompt)
    raise RuntimeError("Provided LLM object has no supported invocation method")
