import streamlit as st
from rag_llm_bot import PromptProcessor
from rag_llm_bot import ytloader
from langchain_community.llms import Ollama
import os
from typing import Literal
import PyPDF2
from langchain_core.messages import HumanMessage, AIMessage
from multimodal_bot import MultimodalPromptProcessor
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
import io
import subprocess

MARGINS = {
    "top": "2.875rem",
    "bottom": "0",
}

STICKY_CONTAINER_HTML = """
<style>
div[data-testid="stVerticalBlock"] div:has(div.fixed-header-{i}) {{
    position: sticky;
    {position}: {margin};
    background-color: white;
    z-index: 999;
}}
</style>
<div class='fixed-header-{i}'/>
""".strip()

# Not to apply the same style to multiple containers
count = 0

def sticky_container(
    *,
    height: int | None = None,
    border: bool | None = None,
    mode: Literal["top", "bottom"] = "top",
    margin: str | None = None,
):
    if margin is None:
        margin = MARGINS[mode]

    global count
    html_code = STICKY_CONTAINER_HTML.format(position=mode, margin=margin, i=count)
    count += 1

    container = st.container(height=height, border=border)
    container.markdown(html_code, unsafe_allow_html=True)
    return container

def main():
    st.sidebar.title("Twyt Summarizer")
    if "history" not in st.session_state:
        st.session_state.history = []
    mp = MultimodalPromptProcessor()
    option = st.sidebar.radio("Select a page:", ["RAGbot", "TextVault", "WebTube Scholar", "Interaction Timeline", "Download PDF Summary"])
    if option == "RAGbot":
        # chatbot_page(processor)
        multimodal_chatbot_page(mp)
    elif option == "TextVault":
        upload_pdf_page()
    elif option == "WebTube Scholar":
        tube_tutor_page()
    elif option == "Interaction Timeline":
        display_timeline()
    elif option == "Download PDF Summary":
        display_pdf_download()

def multimodal_chatbot_page(mp):   
    print("Entering multimodal chatbot page") 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}")
    prompt = st.chat_input("Enter your prompt")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        conversation_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
        )
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            loading_icon = message_placeholder.markdown("|")  
            result = mp.multimodal_process_prompt(prompt, conversation_history)
            loading_icon.empty()
            metadata = result['metadata']
            output = result['output']
            metadata_display = "\n".join(f"- {item}" for item in metadata)
            st.write(metadata_display)
            st.write("----------------------------------------------------------------------------------------------------")
            st.write(output)
            combined_content = f"{metadata_display}\n{'-'*100}\n{output}"
        st.session_state.messages.append({"role": "assistant", "content": combined_content})

def chatbot_page(processor):    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}")
    prompt = st.chat_input("Enter your prompt")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        conversation_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
        )
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            loading_icon = message_placeholder.markdown("|")  
            result = processor.process_prompt(prompt, conversation_history)
            loading_icon.empty()
            metadata = result['metadata']
            output = result['output']
            metadata_display = "\n".join(f"- {item}" for item in metadata)
            st.write(metadata_display)
            st.write("----------------------------------------------------------------------------------------------------")
            st.write(output)
            combined_content = f"{metadata_display}\n{'-'*100}\n{output}"
        st.session_state.messages.append({"role": "assistant", "content": combined_content})

def upload_pdf_page():
    st.subheader("Upload PDF")
    pdf_dir = "reference_texts"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        st.write(f"File Name: {uploaded_file.name}")
        save_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"File saved to {save_path}")

        st.write("Running `multimodal_rag.py` to update the system...")
        try:
            result = subprocess.run(
                ["python", "multimodal_rag.py", "--chunking_strategy", "recursive_sentence"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("System updated successfully!")
                st.text(result.stdout)
            else:
                st.error("An error occurred while updating the system.")
                st.text(result.stderr)
        except Exception as e:
            st.error(f"Failed to execute the script: {e}")

        st.subheader("Available Textbooks")
    pdf_files =  [file for file in os.listdir(pdf_dir) if file.endswith('.pdf')]
    st.write(f"Explore Collection:")
    for pdf_file in pdf_files:
        pdf_url = f"http://128.110.218.28:8888/edit/reference_texts/{pdf_file}"
        st.markdown(f"""
            <a href="{pdf_url}" target="_blank">
                <button style="background-color:#D3D3D3; color:black; border:5px; padding:10px 20px; border-radius:2px; cursor:pointer; margin-bottom: 15px;;">
                    {pdf_file}
                </button>
            </a>
            """, unsafe_allow_html=True)
        
def tube_tutor_page():
    st.subheader("YouTube Video Summarizer")
    youtube_url = st.text_input("Paste YouTube Video Link")
    if st.button("Summarize YouTube Video"):
        result = ytloader(youtube_url)  
        st.session_state.youtube_summary = result
    if "youtube_summary" in st.session_state:
        st.write(st.session_state.youtube_summary)

    # Web search is intentionally commented out for now.
    # st.subheader("Web Search Summarizer")
    # query = st.text_input("Enter a topic or question to search")
    # if st.button("Search Web & Summarize"):
    #     web_results = webSearch(query)
    #     st.session_state.web_summary = web_results
    #     if "web_summary" in st.session_state:
    #         metadata_display = "\n".join(f"- {item}" for item in st.session_state.web_summary["links"])
    #         st.write(metadata_display)
    #         st.write(st.session_state.web_summary["answer"])

def display_timeline():
    st.subheader("Interaction History Timeline")
    for entry in st.session_state.history:
        st.markdown(f"**{entry['feature']}** - {entry['query']}")
        st.write(entry['response'])
        st.write("--------------------------------------------------------------------------------")

def display_pdf_download():
    if st.button("Download Interaction History as PDF"):
        pdf_buffer = generate_pdf()
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="interaction_history.pdf",
            mime="application/pdf"
        )

def generate_pdf():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 12)
    y = 800

    pdf.drawString(100, y, "Interaction History Summary")
    y -= 40

    for entry in st.session_state.history:
        pdf.drawString(100, y, f"{entry['feature']}: {entry['query']}")
        y -= 20
        text = pdf.beginText(100, y)
        text.setFont("Helvetica", 10)
        text.textLines(entry['response'])
        pdf.drawText(text)
        y -= len(entry['response'].split("\n")) * 12 + 20

        if y < 50:  # Start a new page if the content is long
            pdf.showPage()
            y = 800

    pdf.save()
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()
