import os
import warnings
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# 1. LOAD KEYS FIRST: This must happen before LangChain imports!
load_dotenv(find_dotenv())

# 2. FIX HUGGING FACE WARNING: Pass the token to the system environment
if "HF_TOKEN" in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]

# 3. SILENCE DEPRECATION WARNINGS
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 4. NOW IMPORT LANGCHAIN
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.title("Ask Rashid Dental AI Assistant")

    # --- DEBUGGING MENU (Helps you verify your keys are loading) ---
    with st.sidebar:
        st.subheader("System Diagnostics")
        st.write(f"`.env` file found: **{os.path.exists('.env')}**")
        st.write(f"Groq Key Loaded: **{'Yes' if os.environ.get('GROQ_API_KEY') else 'No'}**")
        st.write(f"HF Token Loaded: **{'Yes' if os.environ.get('HF_TOKEN') else 'No'}**")
    # ---------------------------------------------------------------

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                st.error("GROQ_API_KEY is missing! Check the sidebar diagnostics.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.3-70b-versatile",  # Updated to active model
                    temperature=0.0,
                    groq_api_key=groq_key,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_docs = response["source_documents"]
            result_to_show = result + "\n\n**Source Docs:**\n" + str(source_docs)
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


# for run the project
# python -m pipenv run streamlit run medibot.py