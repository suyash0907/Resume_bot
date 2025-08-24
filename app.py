import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import os

DB_DIR = "db"

st.set_page_config(page_title="Resume Chatbot - Ask about Suyash", page_icon="🤖")
st.title("🤖 Resume Chatbot – Ask me anything about Suyash")

# 🔑 Hugging Face API token
hf_token = st.secrets["hf_token"]

# Load vector DB & embeddings (from your prebuilt db in repo)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Cloud-friendly LLM via Hugging Face Inference API
llm = HuggingFaceHub(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.4,
    max_new_tokens=512,
)




SYSTEM_PROMPT = """You are a helpful assistant that answers strictly using the provided context about 'Suyash'.
If the answer is not in the context, say you don't know based on the resume.
Keep answers concise, factual, and interview-friendly.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nUser question: {question}")
])

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.chat_input("e.g., Tell me about Suyash's projects, skills, internships…")
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if user_q:
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = chain.invoke(user_q)
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))











