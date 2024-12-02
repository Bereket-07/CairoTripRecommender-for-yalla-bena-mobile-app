from langchain_huggingface import HuggingFaceEmbeddings


def download_hugging_face_embeddings():
    print("downloading embedding model from hugging face")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("downloadind ...... completed")
    return embedding
