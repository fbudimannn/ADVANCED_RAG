import streamlit as st
import os
import time
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# --- 1. LOAD SECRETS SAFELY ---
def load_secret(key_name: str) -> str:
    """Safely load a secret value from Streamlit secrets."""
    try:
        return st.secrets[key_name]
    except Exception:
        st.error(f"❌ Missing secret: `{key_name}`. Please add it to .streamlit/secrets.toml.")
        st.stop()

# --- 2. SETUP MODELS & DATABASE (Optimized for Streamlit Cloud) ---
@st.cache_resource(show_spinner=False, max_entries=1)
def load_models_and_db():
    """Load models and connect to AstraDB."""
    print("🔄 Loading models and connecting to AstraDB...")

    # ✅ Lightweight embedding model
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # ✅ Lightweight reranker
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        device='cpu'
    )

    # ✅ Default LLM (Mistral 7B)
    llm = ChatOpenAI(
        api_key=load_secret("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        base_url="https://openrouter.ai/api/v1"
    )

    # ✅ AstraDB connection
    vstore = AstraDBVectorStore(
        embedding=embedder,
        collection_name="pubmed_data",
        token=load_secret("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=load_secret("ASTRA_DB_API_ENDPOINT"),
    )

    # ✅ Lighter retriever
    retriever = vstore.as_retriever(search_kwargs={"k": 10})

    print("✅ Models and database connected successfully.")
    return llm, reranker, retriever, vstore


# --- 3. RAG PIPELINE ---
def format_docs(docs):
    """Format documents for LLM context."""
    return "\n\n".join(
        f"--- Start of Context (Source) ---\n"
        f"PMID: {doc.metadata.get('pmid', 'N/A')}\n"
        f"Title: {doc.metadata.get('title', 'N/A')}\n"
        f"Journal: {doc.metadata.get('journal', 'N/A')}\n"
        f"Published Date: {doc.metadata.get('published_date', 'N/A')}\n"
        f"Abstract:\n{doc.page_content[:800]}\n"
        f"--- End of Context (Source) ---"
        for doc in docs
    )


def run_rag_pipeline(query, llm, reranker, retriever):
    """Full RAG pipeline with retry + fallback handling."""
    with st.spinner("🔍 Retrieving relevant studies..."):
        retrieved_docs = retriever.invoke(query)

    with st.spinner("⚖️ Reranking top results..."):
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        rerank_scores = reranker.predict(pairs)
        scored_docs = zip(retrieved_docs, rerank_scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        top_5_docs = [doc for doc, _ in sorted_docs[:5]]

    with st.spinner("🧠 Generating an evidence-based answer..."):
        context = format_docs(top_5_docs)

        prompt_template = """[INST]
**Disclaimer:** This content is for educational use only and not medical advice.

You are a clinical assistant interpreting scientific findings. 
Provide accurate, context-based, evidence-supported responses.

**Instructions:**
1. Use ONLY the context below.
2. Be concise and precise.
3. If insufficient info, state that clearly.
4. Cite with PMID and Title.
5. Respond in one concise paragraph.

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

        # ✅ Retry mechanism
        for attempt in range(3):
            try:
                answer = rag_chain.invoke({"context": context, "question": query})
                break
            except Exception as e:
                if "rate-limited" in str(e).lower():
                    st.warning(f"⏳ Model is busy (attempt {attempt+1}/3). Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    raise e
        else:
            # ✅ Fallback model if retry exhausted
            st.warning("⚠️ Mistral overloaded. Switching to Meta-Llama 3...")
            fallback_llm = ChatOpenAI(
                api_key=load_secret("OPENROUTER_API_KEY"),
                model="meta-llama/Meta-Llama-3-8B-Instruct:free",
                base_url="https://openrouter.ai/api/v1"
            )
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | fallback_llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke({"context": context, "question": query})

    return answer, top_5_docs


# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Cardio RAG", page_icon="🩺", layout="wide")

# ✅ Clean styling
st.markdown("""
<style>
body { background-color: #f8fafc; }
.user-bubble {
    background-color: #e3f2fd;
    padding: 10px 14px; border-radius: 12px; 
    border-left: 4px solid #2196f3; margin: 8px 0;
}
.assistant-bubble {
    background-color: #f1f8e9;
    padding: 10px 14px; border-radius: 12px; 
    border-left: 4px solid #7cb342; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💡 About Cardio RAG")
    st.markdown("""
    AI-powered **Medical Research Assistant** for:
    - 🫀 Cardiovascular Diseases  
    - 🧠 Stroke  
    - 💉 Diabetes  
    
    **Powered by:**
    - LangChain + AstraDB  
    - Mistral-7B (OpenRouter)  
    - MiniLM Reranker  
    """)
    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.caption("⚠️ For educational use only — not for medical diagnosis.")

st.title("🩺 Cardio RAG – Medical Research Assistant")
st.caption("Evidence-based answers generated from PubMed studies by Fakhri.")

# Load backend
try:
    llm, reranker, retriever, vstore = load_models_and_db()
except Exception as e:
    st.error(f"❌ Failed to load models or connect to database: {e}")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (✅ now includes sources)
for msg in st.session_state.messages:
    role = msg["role"]
    css_class = "user-bubble" if role == "user" else "assistant-bubble"
    avatar = "🧑‍⚕️" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)
        if role == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Show Top 5 Sources Used"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"### 🔹 Source {i+1}: {doc.metadata.get('title', 'N/A')}")
                    st.write(f"**PMID:** {doc.metadata.get('pmid', 'N/A')}")
                    st.write(f"**Journal:** {doc.metadata.get('journal', 'N/A')}")
                    st.write(f"**Date:** {doc.metadata.get('published_date', 'N/A')}")
                    if doc.metadata.get("source_url"):
                        st.markdown(f"[🔗 View Article]({doc.metadata.get('source_url')})")
                    st.caption(doc.page_content[:150] + "…")
                    st.divider()

# Handle new user query
if query := st.chat_input("Ask a question about CVD, Stroke, or Diabetes..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"<div class='user-bubble'>{query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            answer, sources = run_rag_pipeline(query, llm, reranker, retriever)
            st.markdown(f"<div class='assistant-bubble'>{answer}</div>", unsafe_allow_html=True)

            # 📚 Display sources
            with st.expander("📚 Show Top 5 Sources Used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"### 🔹 Source {i+1}: {doc.metadata.get('title', 'N/A')}")
                    st.write(f"**PMID:** {doc.metadata.get('pmid', 'N/A')}")
                    st.write(f"**Journal:** {doc.metadata.get('journal', 'N/A')}")
                    st.write(f"**Date:** {doc.metadata.get('published_date', 'N/A')}")
                    if doc.metadata.get("source_url"):
                        st.markdown(f"[🔗 View Article]({doc.metadata.get('source_url')})")
                    st.caption(doc.page_content[:150] + "…")
                    st.divider()

            # ✅ Save to session (with sources)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Sorry, an error occurred: {e}", "sources": []}
            )
