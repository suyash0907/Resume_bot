import streamlit as st
from langchain_community.vectorstores import FAISS
# Updated imports for new packages
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import os

DB_DIR = "db"

st.set_page_config(page_title="Resume Chatbot - Ask about Suyash", page_icon="🤖")
st.title("🤖 Resume Chatbot – Ask me anything about Suyash")

# 1. Load vector DB & embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 2. Use the new HuggingFaceEndpoint class for the LLM
# This is the modern way to connect to Hugging Face models
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=st.secrets["hf_token"], # Token is passed directly
    temperature=0.4,
    max_new_tokens=512,
)

# 3. Define the prompt template
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

# 4. Build the RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Set up the Streamlit Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if user_q := st.chat_input("e.g., Tell me about Suyash's projects..."):
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = chain.invoke(user_q)
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))






