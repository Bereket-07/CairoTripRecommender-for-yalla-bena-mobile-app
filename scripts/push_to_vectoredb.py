from store_index import load_csv , text_split , download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
import time

import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINCONE_API_ENV = os.environ.get('PINECONE_API_ENV')

def load_and_split_data():
    tourist_place_data = load_csv(['../data/preprocessed_data-preprocessed_data.csv'])
    hotel_documnet = load_csv(['../scripts/data2.csv'])
    all_documents = tourist_place_data + hotel_documnet
    text_chunks = text_split(all_documents)
    embeddings = download_hugging_face_embeddings()
    return text_chunks , embeddings
def store_to_vectore_db():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    text_chunks , embeddings = load_and_split_data()
    index_name = "yallabena"
    existing_indexs = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexs:
        print(f"there is no index name {index_name}")
        pc.create_index(
            name= index_name,
            dimension=384,
            metric="cosine",
            spec = ServerlessSpec(cloud="aws",region="us-east-1"),   
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        index = pc.Index(index_name)

        vectore_store = PineconeVectorStore(index=index , embedding=embeddings)
        uuids = [str(uuid4()) for _ in range(len(text_chunks))]
        vectore_store.add_documents(documents=text_chunks,ids=uuids)

        retriver = vectore_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs={"k":3,"score_threshold":0.5},
        )
        return retriver
    else:
        index = pc.Index(index_name)
        print(f"the index name {index_name} is there")
        vectore_store = PineconeVectorStore(index=index,embedding=embeddings)
        retriver = vectore_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":3,"score_threshold":0.5},
        )
    return retriver
    