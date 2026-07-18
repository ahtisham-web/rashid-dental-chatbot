# rashid-dental-chatbot
AI assistant for Rashid Dental Clinic. The chatbot will answer questions using clinic-approved Markdown files, provide information about dental services, collect appointment requests, and direct users toward clinic staff when human assistance is required.

# Medical RAG Chatbot

A Streamlit chatbot that answers questions from *The Gale Encyclopedia of Medicine* 
using RAG (FAISS + HuggingFace embeddings + Groq LLM).

## Setup
1. `pipenv install` (or `pip install -r requirements.txt`)
2. Create a `.env` file with:
3. Place source PDFs in `data/`
4. Run `python create_memory_of_lim.py` to build the FAISS vector store
5. Run `streamlit run medibot.py` to launch the chatbot

## Files
- `create_memory_of_lim.py` — builds FAISS index from PDFs in `data/`
- `connect_memory_with_llm.py` — CLI test script (HuggingFace endpoint)
- `medibot.py` — Streamlit UI (uses Groq)
