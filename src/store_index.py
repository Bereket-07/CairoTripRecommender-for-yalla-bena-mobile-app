from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



#  extract_data_from_the_pdf
def load_pdf(data):
    print("loading document .....")
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    print("loading document ............. completed")
    return documents


# creating text chunks 
def text_split(extracted_data):
    print("text splitting .......")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print("text splitting ..... completed")
    return text_chunks



def download_hugging_face_embeddings():
    print("downloading embedding model from hugging face")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("downloadind ...... completed")
    return embedding