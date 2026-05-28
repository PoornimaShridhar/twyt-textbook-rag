import os
import config
from typing import Any


def _normalize_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    normalized_messages: list[dict[str, Any]] = []
    for item in prompt:
        if isinstance(item, dict):
            role = item.get("role") or item.get("type") or "user"
            content = item.get("content", item)
            normalized_messages.append({"role": role, "content": content})
            continue

        role = getattr(item, "type", "user")
        if role in {"human", "user"}:
            role = "user"
        elif role in {"ai", "assistant"}:
            role = "assistant"
        elif role != "system":
            role = "user"

        content = getattr(item, "content", str(item))
        normalized_messages.append({"role": role, "content": content})

    return normalized_messages


class GroqChatModel:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int | None = None,
        top_p: float = 1.0,
        api_key: str | None = None,
    ):
        self.model = model or getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        from groq import Groq

        resolved_api_key = (
            api_key
            or getattr(config, "GROQ_API_KEY", None)
            or os.getenv("GROQ_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set for this process. Set it in the shell that runs Streamlit, "
                "or place it in a .env file in the project root before launching the app."
            )

        self.client = Groq(api_key=resolved_api_key)

    def invoke(self, prompt: Any) -> str:
        messages = _normalize_messages(prompt)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
            stream=False,
        )
        return completion.choices[0].message.content or ""

    def predict(self, prompt: Any) -> str:
        return self.invoke(prompt)

    def generate(self, prompt: Any) -> str:
        return self.invoke(prompt)


def get_groq_chat(
    temperature: float = 0.0,
    model: str | None = None,
    max_completion_tokens: int | None = None,
    top_p: float = 1.0,
    api_key: str | None = None,
) -> Any:
    """Return a Groq-backed chat wrapper with LangChain-like methods."""
    return GroqChatModel(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        api_key=api_key,
    )


def get_ollama(model: str | None = None, base_url: str | None = None) -> Any:
    """Legacy helper retained for compatibility with older code paths."""
    from langchain_community.llms import Ollama

    model = model or getattr(config, 'OLLAMA_MODEL', 'llama3')
    base_url = base_url or getattr(config, 'OLLAMA_BASE_URL', None)
    if base_url:
        return Ollama(model=model, base_url=base_url)
    return Ollama(model=model)


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
