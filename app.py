import streamlit as st
import os

# LangChain & AstraDB components
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # ✅ Replaces ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Reranker component
from sentence_transformers import CrossEncoder


# --- 1. LOAD SECRETS SAFELY ---

def load_secret(key_name: str) -> str:
    """
    Safely load a secret value from Streamlit secrets.
    Raises a descriptive error if the key is missing.
    """
    try:
        return st.secrets[key_name]
    except Exception:
        st.error(f"❌ Missing secret: `{key_name}`. Please add it to .streamlit/secrets.toml.")
        st.stop()


# --- 2. SET UP MODELS & DATABASE (Load once with cache) ---

@st.cache_resource
def load_models_and_db():
    """
    Load embedding model, reranker model, LLM, and connect to the Vector DB.
    """
    print("🔄 Loading models and connecting to AstraDB...")

    # Load Embedding Model (same as ingest_data.py)
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    # Load Reranker Model
    reranker = CrossEncoder(
        "BAAI/bge-reranker-large",
        max_length=512,
        device='cpu'
    )

    # ✅ Load LLM via OpenRouter (through ChatOpenAI)
    llm = ChatOpenAI(
        api_key=load_secret("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        base_url="https://openrouter.ai/api/v1"
    )

    # ✅ Connect to AstraDB Vector Store
    vstore = AstraDBVectorStore(
        embedding=embedder,
        collection_name="pubmed_data",
        token=load_secret("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=load_secret("ASTRA_DB_API_ENDPOINT"),
    )

    # Create retriever
    retriever = vstore.as_retriever(search_kwargs={"k": 20})

    print("✅ Models and database connected successfully.")
    return llm, reranker, retriever, vstore


# --- 3. DEFINE RAG PIPELINE ---

def format_docs(docs):
    """
    Combine document content into a single string for LLM context.
    Include PMID, title, journal, and date for citation.
    """
    return "\n\n".join(
        f"--- Start of Context (Source) ---\n"
        f"PMID: {doc.metadata.get('pmid', 'N/A')}\n"
        f"Title: {doc.metadata.get('title', 'N/A')}\n"
        f"Journal: {doc.metadata.get('journal', 'N/A')}\n"
        f"Published Date: {doc.metadata.get('published_date', 'N/A')}\n"
        f"Abstract:\n{doc.page_content}\n"
        f"--- End of Context (Source) ---"
        for doc in docs
    )


def run_rag_pipeline(query, llm, reranker, retriever):
    """
    Executes the full RAG pipeline: Retrieve → Rerank → Generate.
    """

    # --- PHASE 1: RETRIEVAL ---
    retrieved_docs = retriever.invoke(query)

    # --- PHASE 2: RERANKING ---
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    rerank_scores = reranker.predict(pairs)
    scored_docs = zip(retrieved_docs, rerank_scores)
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    top_5_docs = [doc for doc, score in sorted_docs[:5]]

    # --- PHASE 3: GENERATION ---
    context = format_docs(top_5_docs)

    prompt_template = """[INST]
**Important Disclaimer:** This information is for educational and informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

You are a clinical assistant specializing in interpreting recent scientific findings. Your primary goal is to provide accurate, context-grounded, evidence-based responses.

**Task:** Answer the following medical question *solely* based on the provided research context.

**Instructions:**
1. Use ONLY the information in the context below. Do not infer, speculate, or introduce external knowledge.
2. Be precise and direct. Clearly answer the question.
3. If there is not enough information, explicitly state that.
4. Cite your answer with PMID and Title, e.g., [PMID: 123456 - Title of Article].
5. Respond concisely in one paragraph.

---

**Question:**
{question}

**Context:**
{context}

---

**Answer:**
[/INST]"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"context": context, "question": query})
    return answer, top_5_docs


# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Cardio RAG", page_icon="🩺", layout="wide")
st.title("🩺 Advanced RAG for Medical Research (CVD, Stroke, Diabetes)")
st.caption("Powered by AstraDB, LangChain, Mistral, and BGE-Reranker")

# Load models and database
try:
    llm, reranker, retriever, vstore = load_models_and_db()
except Exception as e:
    st.error(f"❌ Failed to load models or connect to database: {e}")
    st.stop()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user query
if query := st.chat_input("Ask a question about CVD, Stroke, or Diabetes..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Retrieving, Reranking, Generating)"):
            try:
                answer, sources = run_rag_pipeline(query, llm, reranker, retriever)
                response_content = f"{answer}"

                st.markdown(response_content)

                # Show top sources used
                with st.expander("Show Sources Used (Top 5 Reranked Results)"):
                    for i, doc in enumerate(sources):
                        st.info(f"**Source {i+1}: {doc.metadata.get('title', 'N/A')}**")
                        st.write(f"**PMID:** {doc.metadata.get('pmid', 'N/A')}")
                        st.write(f"**Journal:** {doc.metadata.get('journal', 'N/A')}")
                        st.write(f"**Date:** {doc.metadata.get('published_date', 'N/A')}")
                        st.write(f"**Link:** {doc.metadata.get('source_url', 'N/A')}")
                        st.write(doc.page_content[:300] + "...")

                # Save the response to session
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Sorry, an error occurred: {e}"}
                )
