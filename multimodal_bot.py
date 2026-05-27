from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
import config
if getattr(config, "OPENAI_API_KEY", None):
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
import base64
from langchain.storage import InMemoryStore
from langchain.memory import ConversationBufferMemory
import vectorstore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from IPython.display import HTML, display
from utilities import Utilities
import streamlit as st


class MultimodalPromptProcessor:
    def __init__(self):
        print("-------------------------------------")
        print("ck 1")
        print("-------------------------------------")
        # Initialize components only once in the constructor
        self.utilities = Utilities()
        self.collection_name = config.CHROMA_CLIENT_PATH or "Collections_new_2"
        # PersistentClient path can be configured via `config.CHROMA_CLIENT_PATH`
        # Create vectorstore and client via centralized helper
        self.vectorstore, self.client = vectorstore.create_vectorstore(collection_name=config.DEFAULT_COLLECTION_NAME)
        self.docstore = InMemoryStore()
        self.retriever_multi_vector = vectorstore.make_multi_vector_retriever(self.vectorstore, docstore=self.docstore, k=5)
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.memory = st.session_state.chat_memory
        self.chain_mm_rag = self.multi_modal_rag_chain(vectorstore = self.vectorstore, retriever=self.retriever_multi_vector)

    def multimodal_process_prompt(self, query, conversation_history=None):
        print("-------------------------------------")
        print("ck 2")
        print("-------------------------------------")
        if conversation_history:
            full_prompt = f"{conversation_history}\nUser: {query}"
        else:
            full_prompt = f"User: {query}"
        print("-----------------full prompt--------------------------------")
        print(full_prompt)
        print("-------------------------------------------------------------------\n")
        results = self.vectorstore.similarity_search(query, k=5)
        for r in results:
            print(r)
            print("-----------results----------\n\n\n")

        metadata_list = [doc.metadata for doc in results]
        processed_metadata = self.utilities.process_metadata(metadata_list)
        # chain_mm_rag = self.multi_modal_rag_chain(vectorstore = self.vectorstore, retriever=self.retriever_multi_vector)
        output = self.chain_mm_rag(full_prompt)
        final_output = {
        'metadata': processed_metadata, 
        'output': output  
        }
        print(final_output['output'])
        return  final_output

    def multi_modal_rag_chain(self, vectorstore, retriever, memory=None):
        """Multi-modal RAG chain"""
        print("-------------------------------------")
        print("ck 3")
        print("-------------------------------------")
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        import llm_wrapper
        model = llm_wrapper.get_chat_openai(temperature=0, model="gpt-4o", max_tokens=1024)

        def retrieve_context(query):
            res = vectorstore.similarity_search(query, k=5)
            return res

        # Chain setup to first retrieve context and then process it
        chain = (
            RunnableParallel(
                {
                    "context": RunnableLambda(retrieve_context),  # Run the retriever to get the context dynamically
                    "question": RunnablePassthrough(),  # Just pass through the original question/query
                    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]  # Retrieve chat history
                }
            )
            | RunnableLambda(self.img_prompt_func)  # Construct the final LLM prompt with image handling
            | model  # Pass to the model (LLM)
            | StrOutputParser()  # Parse the output
        )

        def run_chain(query):
            result = chain.invoke(query)
            memory.save_context({"input": query}, {"output": result})
            # chat_history = memory.load_memory_variables({}).get("chat_history", [])
            # chat_history.append({"input": query, "output": result})
            # memory.save_context({"chat_history": chat_history}, {})
            return result
        return run_chain

    def img_prompt_func(self, data_dict):
        """Join the context into a single string"""
        print("-------------------------------------")
        print("ck 4")
        print(data_dict)
        print("-------------------------------------")
        messages = []
        if "context" in data_dict:
            formatted_texts = "\n".join([doc.page_content for doc in data_dict["context"]])
        else:
            formatted_texts = ""  # Default to empty if 'texts' key doesn't exist
        
        for doc in data_dict["context"]:
                if 'image_base64' in doc.metadata:
                    image_message = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{doc.metadata['image_base64']}"}
                    }
                    messages.append(image_message)
        chat_history = data_dict.get("chat_history", [])
        formatted_chat_history = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
        text_message = {
        "type": "text",
        "text": (
            "You are a Research Assistant tasked with answering questions on research articles.\n"
            "You will be given a mix of text, tables, and image(s), usually of tables, charts, or graphs.\n"
            "Use this information to provide accurate information related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
            "Chat History:\n"
            f"{formatted_chat_history}\n\n"
            ),
        }
        messages.append(text_message)
        with open("human_message.txt", "w") as file:
            for message in messages:
                file.write(f"{message}\n")
        return [HumanMessage(content=messages)]        
