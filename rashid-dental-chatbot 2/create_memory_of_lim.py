from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Step 1: PDF file upload configuration
DATA_PATH = "data/"

def load_pdf_files(data):
    # Fixed the typo: loader_cls now correctly uses PyPDFLoader
    loader = DirectoryLoader(
        data, 
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Running the loader
documents = load_pdf_files(data=DATA_PATH)

# Verification step (prints how many documents were successfully loaded)
print(f"Success! Loaded {len(documents)} document pages.")

# create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))

# create vector embedding
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# CREATE A FAISS 
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)