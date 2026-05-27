from langchain.schema import HumanMessage
import config
import llm_wrapper


def summarize_image(img_base64: str, prompt: str) -> str:
    """Use a chat LLM to summarize an image provided as a base64 string.

    Returns a string summary when possible.
    """
    chat = llm_wrapper.get_groq_chat(
        model=getattr(config, "GROQ_VISION_MODEL", None),
        temperature=0,
        max_completion_tokens=1024,
    )

    human_message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])

    res = llm_wrapper.invoke_llm(chat, [human_message])

    # Groq wrapper may return a string directly
    if hasattr(res, "content"):
        return res.content
    return res
