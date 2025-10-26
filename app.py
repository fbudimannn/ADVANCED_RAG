import streamlit as st
import os
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# --- 1. LOAD SECRETS SAFELY ---
def load_secret(key_name: str) -> str:
    try:
        return st.secrets[key_name]
    except Exception:
        st.error(f"❌ Missing secret: `{key_name}`. Please add it to .streamlit/secrets.toml.")
        st.stop()

# --- 2. SET UP MODELS & DATABASE (Cache to avoid reloads) ---
@st.cache_resource
def load_models_and_db():
    print("🔄 Loading models and connecting to AstraDB...")

    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    reranker = CrossEncoder(
        "BAAI/bge-reranker-large",
        max_length=512,
        device='cpu'
    )

    llm = ChatOpenAI(
        api_key=load_secret("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        base_url="https://openrouter.ai/api/v1"
    )

    vstore = AstraDBVectorStore(
        embedding=embedder,
        collection_name="pubmed_data",
        token=load_secret("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=load_secret("ASTRA_DB_API_ENDPOINT"),
    )

    retriever = vstore.as_retriever(search_kwargs={"k": 20})
    print("✅ Models and database connected successfully.")
    return llm, reranker, retriever, vstore


# --- 3. RAG PIPELINE ---
def format_docs(docs):
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
    with st.spinner("🔍 Retrieving relevant research..."):
        retrieved_docs = retriever.invoke(query)

    with st.spinner("⚖️ Reranking top results..."):
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        rerank_scores = reranker.predict(pairs)
        scored_docs = zip(retrieved_docs, rerank_scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        top_5_docs = [doc for doc, score in sorted_docs[:5]]

    with st.spinner("🧠 Generating evidence-based response..."):
        context = format_docs(top_5_docs)
        prompt_template = """[INST]
**Important Disclaimer:** This information is for educational and informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

You are a clinical assistant specializing in interpreting recent scientific findings. Your goal is to provide accurate, evidence-based, and context-grounded responses.

**Task:** Answer the following medical question *solely* based on the provided research context.

**Instructions:**
1. Use ONLY the context below. Do not speculate or use external knowledge.
2. Be concise and clear.
3. If information is insufficient, say so explicitly.
4. Cite answers with PMID and Title, e.g., [PMID: 123456 - Title].
5. One paragraph only.

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

# Custom CSS Styling
st.markdown("""
    <style>
    body { background-color: #f8fafc; }
    .stChatMessage { background-color: #f9fafb; border-radius: 10px; padding: 8px; margin-bottom: 10px; }
    .user-msg { background-color: #e0f7fa; border-left: 5px solid #00acc1; }
    .assistant-msg { background-color: #f1f8e9; border-left: 5px solid #7cb342; }
    .stExpander { background-color: #f5f5f5; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About Cardio RAG")
    st.markdown("""
    **Cardio RAG** is an AI-powered research assistant focused on:
    - 🫀 Cardiovascular Disease  
    - 🧠 Stroke  
    - 💉 Diabetes  

    It uses:
    - **Mistral 7B** (via OpenRouter)
    - **AstraDB Vector Store**
    - **BGE-Reranker-Large**
    - **LangChain RAG Framework**
    """)
    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
    st.caption("⚠️ This app does not provide medical advice.")

# Title and caption
st.title("🩺 Cardio RAG – Medical Research Assistant")
st.caption("Answering evidence-based clinical questions using scientific literature (PubMed).")

# Load models
try:
    llm, reranker, retriever, vstore = load_models_and_db()
except Exception as e:
    st.error(f"❌ Failed to load models or connect to database: {e}")
    st.stop()

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖"):
        bubble_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# Chat input
if query := st.chat_input("Ask a question about CVD, Stroke, or Diabetes..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"<div class='user-msg'>{query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            answer, sources = run_rag_pipeline(query, llm, reranker, retriever)
            response = f"{answer}"
            st.markdown(f"<div class='assistant-msg'>{response}</div>", unsafe_allow_html=True)

            with st.expander("📚 Sources (Top 5 Reranked Results)"):
                for i, doc in enumerate(sources):
                    st.markdown(f"### 🔹 Source {i+1}: {doc.metadata.get('title', 'N/A')}")
                    st.write(f"**PMID:** {doc.metadata.get('pmid', 'N/A')}")
                    st.write(f"**Journal:** {doc.metadata.get('journal', 'N/A')}")
                    st.write(f"**Date:** {doc.metadata.get('published_date', 'N/A')}")
                    if doc.metadata.get("source_url"):
                        st.markdown(f"[🔗 Read More]({doc.metadata.get('source_url')})")
                    st.caption(doc.page_content[:250] + "…")
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Sorry, an error occurred: {e}"}
            )
