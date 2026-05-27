from langchain.schema import HumanMessage
import llm_wrapper


def summarize_image(img_base64: str, prompt: str) -> str:
    """Use a chat LLM to summarize an image provided as a base64 string.

    Returns a string summary when possible.
    """
    chat = llm_wrapper.get_chat_openai(model="gpt-4o", temperature=0, max_tokens=1024)

    human_message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])

    res = llm_wrapper.invoke_llm(chat, [human_message])

    # langchain ChatOpenAI may return an object with `.content` or a string
    if hasattr(res, "content"):
        return res.content
    return res
