from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage , HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
import os
from helper import download_hugging_face_embeddings
from flask import Flask , request , render_template , jsonify

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

embedding = download_hugging_face_embeddings()
store = {}

def main_ai_chat_bot(data):
    load_dotenv()
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "yallabena"
    index = pc.Index(index_name)
    print(f"the index name {index_name} is there")
    vectore_store = PineconeVectorStore(index=index , embedding=embedding)
    retriver = vectore_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )
    
    llm = ChatGroq(model="llama3-8b-8192")

    # contextualize question 
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm , retriver ,contextualize_q_prompt
    )
    system_prompt = (
    "You are a travel assistant designed to provide day-by-day trip recommendations based on user preferences. "
    "Use the following user-provided details: travel duration (start and end dates), budget, travel companion type "
    "(alone, family, or friends), and interests. Budget allocation should be as follows: 30% for accommodation, "
    "20% for meals, and the rest for transportation and activities. "
    "Retrieve relevant information from the database of local hotels, restaurants, and tourist spots. "
    "For each day of the trip, recommend options for hotels, meals, and activities, considering proximity, budget, and interests. "
    "Provide brief, day-specific suggestions for each aspect to create a clear itinerary for the user’s specified dates."
    "\n\n"
    "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human" , "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    ### statefully manage chat history

    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    response = conversational_rag_chain.invoke(
        {"input":data},
        config={
            "configurable" : {"session_id" : "abc123"}
        },
    )["answer"]
    return response

app = Flask(__name__)


@app.route("/recommender" , methods=["POST"])
def chat():
    msg = request.json.get("message")
    response = main_ai_chat_bot(msg)
    return jsonify({"reply":response})

if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=8000,debug=True)