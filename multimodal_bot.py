import os
import config
import base64
from vectorstore import InMemoryStore, ConversationBufferMemory
import vectorstore
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage
from utilities import Utilities
import streamlit as st


class MultimodalPromptProcessor:
    def __init__(self):
        print("-------------------------------------")
        print("ck 1")
        print("-------------------------------------")
        # Initialize components only once in the constructor
        self.utilities = Utilities()
        self.collection_name = config.DEFAULT_COLLECTION_NAME or "text_Collection_3"
        # Create vectorstore and client via centralized helper
        self.vectorstore, self.client = vectorstore.create_vectorstore(collection_name=config.DEFAULT_COLLECTION_NAME)
        try:
            print(f"[MultimodalPromptProcessor.__init__] vectorstore={self.vectorstore} client={self.client}")
        except Exception:
            pass
        self.docstore = InMemoryStore()
        self.retriever_multi_vector = vectorstore.make_multi_vector_retriever(self.vectorstore, docstore=self.docstore, k=5)
        try:
            print(f"[MultimodalPromptProcessor.__init__] retriever={self.retriever_multi_vector} docstore={self.docstore}")
        except Exception:
            pass
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
        try:
            print(f"[multimodal_process_prompt] invoking retriever={self.retriever_multi_vector} on vectorstore={self.vectorstore}")
        except Exception:
            pass
        results = self.retriever_multi_vector.invoke(query)
        for r in results:
            print(r)
            print("-----------results----------\n\n\n")

        # Robust metadata extraction for different return shapes
        metadata_list = []
        for doc in results:
            meta = {}
            # langchain Document-like
            if hasattr(doc, 'metadata'):
                try:
                    meta = doc.metadata or {}
                except Exception:
                    meta = {}
            # dict-like
            elif isinstance(doc, dict):
                meta = doc.get('metadata') or doc.get('meta') or {}
            # tuple/list (maybe (doc, score))
            elif isinstance(doc, (list, tuple)) and len(doc) > 0:
                inner = doc[0]
                if hasattr(inner, 'metadata'):
                    meta = getattr(inner, 'metadata') or {}
                elif isinstance(inner, dict):
                    meta = inner.get('metadata') or inner.get('meta') or {}
            # otherwise leave as empty dict
            metadata_list.append(meta)

        print("Retrieved metadata list:", metadata_list)
        processed_metadata = self.utilities.process_metadata(metadata_list)
        # chain_mm_rag = self.multi_modal_rag_chain(vectorstore = self.vectorstore, retriever=self.retriever_multi_vector)
        payload = {
            "question": query,
            "chat_history": conversation_history or "",
        }
        output = self.chain_mm_rag(payload)
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
        model = llm_wrapper.get_groq_chat(temperature=0, model=getattr(config, "GROQ_MODEL", None), max_completion_tokens=1024)

        def retrieve_context(payload):
            query = payload.get("question", payload) if isinstance(payload, dict) else payload
            try:
                print(f"[multi_modal_rag_chain.retrieve_context] using vectorstore passed to chain: {vectorstore}")
            except Exception:
                pass
            res = vectorstore.similarity_search(query, k=5)
            try:
                print(f"[multi_modal_rag_chain.retrieve_context] retrieved {len(res) if hasattr(res, '__len__') else 'N/A'} docs")
            except Exception:
                pass
            return res

        # Chain setup to first retrieve context and then process it
        chain = (
            RunnableParallel(
                {
                    "context": RunnableLambda(retrieve_context),
                    "question": RunnableLambda(lambda x: x.get("question", x) if isinstance(x, dict) else x),
                    "chat_history": RunnableLambda(lambda x: x.get("chat_history", "") if isinstance(x, dict) else ""),
                }
            )
            | RunnableLambda(self.img_prompt_func)  # Construct the final LLM prompt with image handling
            | RunnableLambda(model.invoke)  # Call the custom Groq model through a runnable adapter
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
        if isinstance(chat_history, str):
            formatted_chat_history = chat_history.strip()
        elif isinstance(chat_history, list):
            if chat_history and isinstance(chat_history[0], dict):
                formatted_chat_history = "\n".join(
                    f"{msg.get('role', msg.get('type', 'message'))}: {msg.get('content', msg)}"
                    for msg in chat_history
                )
            else:
                formatted_chat_history = "\n".join([f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', m)}" for m in chat_history])
        else:
            formatted_chat_history = str(chat_history)
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
