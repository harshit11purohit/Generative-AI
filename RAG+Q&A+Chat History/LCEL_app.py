import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=os.getenv("GROQ_API_KEY"))

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

docs=PyPDFDirectoryLoader("research_papers").load()

splits=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

vectorstore=Chroma.from_documents(splits,embeddings)

retriever=vectorstore.as_retriever()


contextualize_prompt=ChatPromptTemplate.from_messages([
("system","Rewrite the question using chat history. Do not answer."),
MessagesPlaceholder("chat_history"),
("human","{input}")
])


question_rewriter=contextualize_prompt|llm|StrOutputParser()


def retrieve_docs(inputs):
    question=question_rewriter.invoke(inputs)
    docs=retriever.invoke(question)
    return docs


qa_prompt=ChatPromptTemplate.from_messages([
("system","Answer only from context:\n{context}"),
MessagesPlaceholder("chat_history"),
("human","{input}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain=(
{
"context":retrieve_docs|format_docs,
"input":RunnablePassthrough(),
"chat_history":lambda x:x["chat_history"]
}
|qa_prompt
|llm
|StrOutputParser()
)


store={}


def get_session_history(session_id):

    if session_id not in store:
        store[session_id]=ChatMessageHistory()

    return store[session_id]


conversation_chain=RunnableWithMessageHistory(
rag_chain,
get_session_history,
input_messages_key="input",
history_messages_key="chat_history"
)


response=conversation_chain.invoke(
{"input":"How does it work?"},
config={"configurable":{"session_id":"user1"}}
)


print(response)