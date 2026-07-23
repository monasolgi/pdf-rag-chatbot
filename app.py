import os
import tempfile

import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about its content.")

#@st.cache_resource
#def load_embedding_model():
#    return SentenceTransformer("all-MiniLM-L6-v2")
#

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    )
embedding_model = load_embedding_model()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        api_key = st.sidebar.text_input("Groq API key", type="password")
    return Groq(api_key=api_key) if api_key else None

def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": page_number, "text": text.strip()})
    return pages

def create_chunks(pages, chunk_size=1000, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        split_texts = splitter.split_text(page["text"])

        for text in split_texts:
            chunks.append({
                "page": page["page"],
                "text": text,
            })

    return chunks
def create_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, top_k=4):
    question_embedding = embedding_model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(question_embedding, min(top_k, len(chunks)))
    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        item = chunks[chunk_index].copy()
        item["score"] = float(score)
        results.append(item)
    return results

def build_context(retrieved_chunks):
    return "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}" for item in retrieved_chunks
    )

def format_chat_history(history, max_messages=6):
    recent = history[-max_messages:]
    if not recent:
        return "No previous conversation."
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in recent
    )

def rewrite_question(question, history, client):
    if not history:
        return question
    prompt = f"""
Rewrite the latest user question as a complete standalone question.
Use the conversation history only to resolve references such as it, they,
the first one, that method, or this dataset.
Do not answer the question. Return only the rewritten question.

Conversation history:
{format_chat_history(history)}

Latest question:
{question}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def generate_answer(question, standalone_question, retrieved_chunks, history, client):
    system_prompt = """
You are a document question-answering assistant.
Answer using only the supplied document context.
Use chat history only to understand the discussion, not as a factual source.
If the context is insufficient, say so clearly.
Cite relevant page numbers and keep the answer concise but complete.
"""
    user_prompt = f"""
CONVERSATION HISTORY:
{format_chat_history(history)}

DOCUMENT CONTEXT:
{build_context(retrieved_chunks)}

CURRENT QUESTION:
{question}

STANDALONE QUESTION USED FOR RETRIEVAL:
{standalone_question}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

for key, value in {
    "messages": [],
    "chunks": None,
    "faiss_index": None,
    "uploaded_filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

client = get_groq_client()
top_k = st.sidebar.slider("Number of retrieved chunks", 1, 10, 4)
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_filename:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name
        try:
            pages = extract_pages(temp_path)
            chunks = create_chunks(pages)
            if not chunks:
                st.error("No usable text was found in the PDF.")
            else:
                st.session_state.chunks = chunks
                st.session_state.faiss_index = create_faiss_index(chunks)
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []
                st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks.")
        finally:
            os.remove(temp_path)

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}** — similarity: {source['score']:.3f}")
                    st.write(source["text"])

question = st.chat_input("Ask a question about the PDF")
if question:
    if client is None:
        st.error("Please enter your Groq API key.")
    elif st.session_state.chunks is None:
        st.error("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                standalone_question = rewrite_question(question, history, client)
                sources = retrieve_chunks(
                    standalone_question,
                    st.session_state.chunks,
                    st.session_state.faiss_index,
                    top_k,
                )
                answer = generate_answer(
                    question, standalone_question, sources, history, client
                )
            st.markdown(answer)
            with st.expander("Sources"):
                st.caption(f"Retrieval query: {standalone_question}")
                for source in sources:
                    st.markdown(f"**Page {source['page']}** — similarity: {source['score']:.3f}")
                    st.write(source["text"])
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
