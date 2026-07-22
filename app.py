import os
import time
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="Generative AI Chatbot",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #0b0f19;
            color: #f5f7fb;
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #253044;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            position: fixed;
            top: 0.8rem;
            left: 0.8rem;
            z-index: 999999;
            background: #172033;
            border: 1px solid #334155;
            border-radius: 10px;
            color: #f8fafc;
        }

        [data-testid="stSidebarCollapsedControl"]:hover {
            background: #24324a;
            border-color: #4f6b95;
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        .app-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .app-header h1 {
            color: #f8fafc;
            font-size: 2rem;
            margin-bottom: 0.35rem;
            letter-spacing: -0.04em;
        }

        .sidebar-title {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 1.25rem;
        }

        .sidebar-card {
            background: linear-gradient(135deg, #172033, #111827);
            border: 1px solid #2b3952;
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0;
        }

        .sidebar-card h3 {
            color: #f8fafc;
            font-size: 1rem;
            margin: 0 0 0.6rem;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border: none;
            padding: 0.55rem 0;
        }

        [data-testid="stChatMessageContent"] {
            border-radius: 18px;
            padding: 0.9rem 1.1rem;
            line-height: 1.6;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) [data-testid="stChatMessageContent"] {
            background: #1d4ed8;
            color: #ffffff;
            border-bottom-right-radius: 5px;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-assistant"]
        ) [data-testid="stChatMessageContent"] {
            background: #151d2d;
            border: 1px solid #26334a;
            color: #e2e8f0;
            border-bottom-left-radius: 5px;
        }

        [data-testid="stChatInput"] {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 0.2rem 0.6rem;
        }

        [data-testid="stChatInput"] textarea {
            color: #f8fafc !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #334155;
            background: #172033;
            color: #e2e8f0;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background: #24324a;
            border-color: #4f6b95;
            color: #ffffff;
        }

        .stSpinner > div {
            border-top-color: #60a5fa !important;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return Ollama(model="qwen2.5:3b", temperature=0.1)


def create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    embeddings = load_embeddings()
    database = FAISS.from_documents(chunks, embeddings)

    return database


def clear_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">✦ Document Workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-subtitle">Upload a PDF to start asking questions.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            pdf_path = tmp_file.name

        try:
            with st.spinner("Analyzing and indexing your document..."):
                database = create_vectorstore(pdf_path)

            st.session_state.retriever = database.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10},
            )
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            st.success("PDF ready for questions.")
        except Exception as error:
            st.error(f"Unable to process PDF: {error}")
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    st.markdown(
        """
        <div class="sidebar-card">
            <h3>ℹ️ How it works</h3>
            <p style="color:#9a9cb0; font-size:13px; line-height:1.5; margin-bottom:0;">
                Your PDF is split into chunks, embedded, and stored in a local
                FAISS index. Each question retrieves the most relevant chunks
                and passes them to the LLM as context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "⌫ Clear Chat",
        use_container_width=True,
        on_click=clear_chat,
    )


st.markdown(
    """
    <div class="app-header">
        <h1>Generative AI Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

llm = load_llm()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retriever is None:
    st.info("Upload a PDF from the sidebar to begin.")

query = st.chat_input(
    "Ask a question about your document...",
    disabled=st.session_state.retriever is None,
)

if query and st.session_state.retriever:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant sections..."):
            documents = st.session_state.retriever.invoke(query)

        context = "\n\n".join(document.page_content[:700] for document in documents)
        history = "".join(
            f"{message['role']}: {message['content']}\n"
            for message in st.session_state.messages[-6:]
        )

        prompt = f"""
You are a helpful AI assistant. Use ONLY the provided context.

Rules:
- Use the chat history when answering follow-up questions.
- If the answer is not present in the context, say:
  "I don't have enough information in the document."
- Keep answers concise and clear.

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(prompt):
                full_response += chunk
                response_placeholder.markdown(f"{full_response}▍")
                time.sleep(0.01)
        except Exception:
            full_response = llm.invoke(prompt)
            response_placeholder.markdown(full_response)

        response_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )