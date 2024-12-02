import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


#  extract_data_from_the_pdf
def load_csv(file_paths):
    documents = []
    print("loading the data")
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # convert each roe to a text format with relevant
        for _, row in df.iterrows():
            if 'Type of Activity' in df.columns: # Tourist plaes data set
                text = f"Name: {row['Name']}, Type: {row['Type of Activity']}, Description: {row['Description']}, Location: {row['Location']}, Opening Times: {row['OpeningTimes']}, OpeningTime : {row['OpeningTime']} , ClosingTime: {row['ClosingTime']} , Duration: {row['Duration']}, Is_Special_Schedule: {row['Is_Special_Schedule']}, Fee Adult: {row['Fee_Adult_EGP']}, Fee Kid: {row['Fee_kid_EGP']}"
            else: # Hotels and resturants dataset
                text = f"Name: {row['name']}, State: {row['state']}, District: {row['district']}, Street: {row['street']}, Coordinates: ({row['lon']}, {row['lat']}), Categories: {row['categories']}"
            documents.append(Document(page_content=text))
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
