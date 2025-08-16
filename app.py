import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os


# HuggingFace API key (you can get free from https://huggingface.co/settings/tokens)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token = st.secrets["hf_token"]

# Load resume
with open("resume.txt", "r") as f:
    resume_text = f.read()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_text(resume_text)

# Embed resume into vector database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(docs, embeddings)

# Use free HuggingFace model for Q&A
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0, "max_length":256})

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.set_page_config(page_title="Suyash's Resume Chatbot", page_icon="🤖")
st.title("🤖 Resume Chatbot - Ask me anything about Suyash!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask me about my skills, projects, or experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    answer = qa.run(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

