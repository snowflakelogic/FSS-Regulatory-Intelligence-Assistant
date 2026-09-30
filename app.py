import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Your notebook saves the FAISS index as "faiss_index".
# This also supports the "Data/faiss_index" layout shown in
# your project explorer.
FAISS_INDEX_CANDIDATES = [
    BASE_DIR / "faiss_index",
    BASE_DIR / "Data" / "faiss_index",
]

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def find_faiss_index() -> Path | None:
    """Find the existing FAISS index created by the notebook."""
    for path in FAISS_INDEX_CANDIDATES:
        if (path / "index.faiss").exists() and (path / "index.pkl").exists():
            return path
    return None


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents for the RAG prompt."""
    formatted = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        formatted.append(
            f"Document {i + 1} "
            f"(Source: {source}, Page: {page}):\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


# ---------------------------------------------------------
# Load RAG system once
# ---------------------------------------------------------

@st.cache_resource
def load_rag():
    """Load embeddings, FAISS, retriever, and Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. "
            "Add GROQ_API_KEY=your_key to the project's .env file."
        )

    faiss_path = find_faiss_index()

    if faiss_path is None:
        raise FileNotFoundError(
            "FAISS index was not found.\n\n"
            "Run the FAISS creation cells in "
            "RAG-WITH-LANGCHAIN-FAISS.ipynb first. "
            "The index should contain index.faiss and index.pkl."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        str(faiss_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the supplied context.
- If the context does not contain enough information to answer,
  say that the information was not found in the provided document.
- Do not invent facts.
- Give a clear and concise answer.

Answer:"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, faiss_path


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="FSS Rules RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 FSS Rules RAG Assistant")
st.caption("Ask questions about the Food Safety and Standards Rules, 2011.")

with st.sidebar:
    st.header("RAG Configuration")
    st.write(f"**LLM:** `{MODEL_NAME}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write("**Retriever:** Similarity search, top 3 documents")
    st.divider()
    st.info(
        "The answer is generated from the documents retrieved "
        "from your local FAISS index."
    )

try:
    rag_chain, retriever, faiss_path = load_rag()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.success(f"FAISS index loaded from `{faiss_path}`")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question
question = st.chat_input(
    "Ask a question about the FSS Rules, 2011..."
)

if question:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                answer = rag_chain.invoke(question)
                source_docs = retriever.invoke(question)

                st.markdown(answer)

                # Show retrieved sources
                with st.expander("📖 Retrieved sources"):
                    seen = set()

                    for i, doc in enumerate(source_docs, start=1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "Unknown")

                        source_key = (str(source), str(page))

                        if source_key in seen:
                            continue

                        seen.add(source_key)

                        st.markdown(
                            f"**Source {i}:** `{Path(str(source)).name}`  "
                            f"— **Page {page}**"
                        )

                        st.caption(doc.page_content[:500] + "...")

            except Exception as e:
                answer = f"Error while generating the answer: {e}"
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# Clear chat button
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()