
import re
import time

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# ---------------------- CONFIG STREAMLIT ----------------------
st.set_page_config(page_title="PDF-RAG (Student Version)", page_icon="📘", layout="wide")
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
st.title("Local PDF-RAG – Student Version")

# ---------------------- SESSION STATE ----------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "llm" not in st.session_state:
    st.session_state.llm = Ollama(model="llama3.2")

# Prompt commun (utilisé par le RAG classique ET le RAG amélioré)
RAG_PROMPT = """You must answer ONLY using the context below.
If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""


# ---------------------- TODO 1 : QUERY EXPANSION ----------------------
def expand_query(query, llm):
    """
    - Utilise le LLM pour générer 3 reformulations de la question
    - Retourne une liste de requêtes (question originale + reformulations)
    """
    prompt = f"""Generate 3 alternative questions that mean the same as the question below.
Use different words or angles to help a search engine find relevant passages.
Return ONLY the 3 questions, one per line, without numbering and without any explanation.

Question: {query}
"""
    try:
        response = llm.invoke(prompt)
    except Exception:
        return [query]

    reformulations = []
    for line in response.split("\n"):
        # Nettoyage : retire numérotation / puces ("1.", "1)", "-", "*")
        line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        # On garde uniquement les lignes qui ressemblent à une question
        if len(line) > 10 and line.lower() != query.lower():
            reformulations.append(line)

    # Question originale en premier + max 3 reformulations
    return [query] + reformulations[:3]


# ---------------------- TODO 2 : RERANKING ----------------------
def rerank_chunks(query, docs, llm, top_k=5):
    """
    - Supprime les doublons
    - Fait scorer chaque chunk (0 à 10) par le LLM
    - Retourne les top_k chunks les plus pertinents
    """
    # 1) Dédoublonnage (les requêtes multiples ramènent souvent les mêmes chunks)
    unique_docs = []
    seen = set()
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # 2) Scoring par le LLM
    scored = []
    for doc in unique_docs:
        prompt = f"""You are a relevance judge.
Rate how useful the passage below is to answer the question.
Answer with ONLY one integer between 0 (irrelevant) and 10 (perfectly relevant).

Question: {query}

Passage:
{doc.page_content[:1000]}

Score (0-10):"""
        try:
            response = llm.invoke(prompt)
            match = re.search(r"\d+(\.\d+)?", response)
            score = min(float(match.group()), 10.0) if match else 0.0
        except Exception:
            score = 0.0
        scored.append((score, doc))

    # 3) Tri décroissant et sélection des top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------- RAG CLASSIQUE ----------------------
def classic_rag(query, db, llm):
    """RAG classique : retrieval top-k -> LLM (via RetrievalQA)."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]


# ---------------------- RAG AMÉLIORÉ ----------------------
def improved_rag(query, db, llm):
    """RAG amélioré : Query Expansion -> Multi-search -> Reranking -> LLM."""
    # Query expansion
    expanded_queries = expand_query(query, llm)

    # Multi-search
    retriever = db.as_retriever(search_kwargs={"k": 5})
    all_docs = []
    for q in expanded_queries:
        all_docs.extend(retriever.invoke(q))

    # Reranking
    selected_docs = rerank_chunks(query, all_docs, llm, top_k=5)

    # Prompt final
    context = "\n\n".join(doc.page_content for doc in selected_docs)
    answer = llm.invoke(RAG_PROMPT.format(context=context, query=query))

    return answer, selected_docs, expanded_queries, len(all_docs)


def show_sources(docs):
    for doc in docs:
        page = doc.metadata.get("page", "?")
        st.write(f"- {doc.metadata.get('source', '?')} (page {page})")


# ---------------------- PDF PROCESSING ----------------------
if uploaded_file is not None and st.session_state.processed_file != uploaded_file.name:
    with st.spinner("Processing PDF..."):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        st.session_state.db = Chroma.from_documents(texts, embeddings)
        st.session_state.processed_file = uploaded_file.name

        st.success("PDF processed successfully!")

# ---------------------- QUESTION / ANSWER ----------------------
query = st.chat_input("Ask a question about the PDF:")

if query and st.session_state.db:
    st.write(f"**Question :** {query}")
    col_classic, col_improved = st.columns(2)

    # ---------- RAG CLASSIQUE ----------
    with col_classic:
        st.subheader(" RAG classique")
        with st.spinner("Generating classic answer..."):
            t0 = time.time()
            answer_c, docs_c = classic_rag(query, st.session_state.db, st.session_state.llm)
            time_c = time.time() - t0

        st.write("### Answer")
        st.write(answer_c)
        st.write("### Sources")
        show_sources(docs_c)
        st.caption(f"⏱ {time_c:.1f} s")

    # ---------- RAG AMÉLIORÉ ----------
    with col_improved:
        st.subheader(" RAG amélioré (Expansion + Reranking)")
        with st.spinner("Generating improved answer..."):
            t0 = time.time()
            answer_i, docs_i, expanded_queries, n_retrieved = improved_rag(
                query, st.session_state.db, st.session_state.llm
            )
            time_i = time.time() - t0

        with st.expander("Requêtes générées (Query Expansion)"):
            for q in expanded_queries:
                st.write(f"- {q}")

        st.write("### Answer")
        st.write(answer_i)
        st.write("### Sources")
        show_sources(docs_i)
        st.caption(f"⏱ {time_i:.1f} s | {n_retrieved} chunks récupérés → 5 conservés après reranking")

    # ---------- COMPARAISON ----------
    st.divider()
    st.write("###  Comparaison")
    st.table({
        "Critère": ["Temps de réponse (s)", "Nombre de requêtes", "Chunks utilisés"],
        "RAG classique": [f"{time_c:.1f}", 1, len(docs_c)],
        "RAG amélioré": [f"{time_i:.1f}", len(expanded_queries), len(docs_i)],
    })

elif query:
    st.warning("Please upload a PDF first.")