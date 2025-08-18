import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
import os

# 1. Page config
st.set_page_config(page_title="🤖 Resume Chatbot", layout="wide")
st.title("🤖 Resume Chatbot – Ask me anything about Suyash!")

# 2. Load Resume PDF directly (no upload needed)
with open("Suyash_Dombe_Resume.pdf", "rb") as f:
    pdf_reader = PdfReader(f)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

# 3. Split text
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=800, chunk_overlap=100, length_function=len
)
chunks = text_splitter.split_text(text)

# 4. Embeddings & Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)

# 5. Hugging Face API token (from secrets)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["hf_token"]

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

# 6. Conversational Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
)

# 7. Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask a question about my resume:")
if query:
    result = qa({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))

# Display conversation
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")


