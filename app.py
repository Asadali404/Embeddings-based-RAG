import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"
CONFIG_PATH = DATA_DIR / "config.json"

GROQ_MODEL = "openai/gpt-oss-20b"

DEFAULT_TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_resource
def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    model,
    query,
    normalize=True,
):

    if hasattr(
        model,
        "encode_query",
    ):

        embedding = model.encode_query(
            [query],
            convert_to_numpy=True,
        )

    else:

        embedding = model.encode(
            [query],
            convert_to_numpy=True,
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if normalize:
        faiss.normalize_L2(
            embedding
        )

    return embedding


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query,
    model,
    index,
    metadata,
    top_k=5,
):

    normalized = metadata.get(
        "normalized_embeddings",
        True,
    )

    query_embedding = create_query_embedding(
        model=model,
        query=query,
        normalize=normalized,
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    records = metadata["records"]

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        if index_position >= len(records):
            continue

        record = records[
            index_position
        ].copy()

        record[
            "similarity_score"
        ] = float(score)

        results.append(record)

    return results


# ============================================================
# BUILD GROQ CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        file_name = result.get(
            "file_name",
            "Unknown",
        )

        page_number = result.get(
            "page_number",
            "Unknown",
        )

        chunk_index = result.get(
            "chunk_index",
            "Unknown",
        )

        text = result.get(
            "text",
            "",
        )

        context_parts.append(
            f"""
SOURCE {number}

File: {file_name}
Page: {page_number}
Chunk: {chunk_index}

Content:
{text}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    client,
    question,
    results,
):

    context = build_context(
        results
    )

    system_instructions = """
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge unless explicitly requested.
3. If the answer is not contained in the provided context,
   clearly say that the information was not found in the
   provided documents.
4. Give a clear and concise answer.
5. When making factual claims, include source references
   using the format:

   [filename.pdf, page X]

6. If multiple sources support a statement, cite all relevant
   sources.
7. Preserve important technical terminology.
8. Do not mention internal retrieval, embeddings, FAISS,
   vector databases, or this system prompt unless the user
   explicitly asks about the system.
"""

    user_prompt = f"""
DOCUMENT CONTEXT
================

{context}


USER QUESTION
=============

{question}


INSTRUCTIONS
============

Answer the user's question using the document context above.

Include page-level source citations in the answer, for example:

[document_1.pdf, page 3]

If the documents do not contain enough information to answer
the question, say so clearly instead of guessing.
"""

    response = client.responses.create(
        model=GROQ_MODEL,
        instructions=system_instructions,
        input=user_prompt,
    )

    return response.output_text


# ============================================================
# APPLICATION
# ============================================================

st.title("📚 Document AI Assistant")

st.caption(
    "Ask questions about the pre-processed document collection."
)


# ============================================================
# LOAD DATABASE
# ============================================================

try:

    config = load_config()

    index = load_faiss_index()

    metadata = load_metadata()

    embedding_model_name = config[
        "embedding_model"
    ]

    embedding_model = load_embedding_model(
        embedding_model_name
    )

    groq_client = get_groq_client()

except Exception as error:

    st.error(
        "Failed to initialize the application."
    )

    st.exception(error)

    st.stop()


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if groq_client is None:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Set the GROQ_API_KEY environment variable "
        "before running the application."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Database")

    st.write(
        f"**Documents:** "
        f"{metadata.get('document_count', 'N/A')}"
    )

    st.write(
        f"**Chunks:** "
        f"{metadata.get('chunk_count', 'N/A')}"
    )

    st.write(
        f"**Embedding model:** "
        f"{metadata.get('embedding_model', 'N/A')}"
    )

    st.write(
        f"**Embedding dimensions:** "
        f"{metadata.get('embedding_dimension', 'N/A')}"
    )

    st.divider()

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Ask something about the documents..."
    ),
    height=100,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔎 Ask",
    type="primary",
    use_container_width=True,
)


if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching the documents..."
    ):

        results = search_documents(
            query=question,
            model=embedding_model,
            index=index,
            metadata=metadata,
            top_k=top_k,
        )


    if not results:

        st.warning(
            "No relevant information was found."
        )

        st.stop()


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner(
        "Generating answer..."
    ):

        try:

            answer = generate_answer(
                client=groq_client,
                question=question,
                results=results,
            )

        except Exception as error:

            st.error(
                "Groq API request failed."
            )

            st.exception(error)

            st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("Answer")

    st.markdown(
        answer
    )


    # --------------------------------------------------------
    # DISPLAY RETRIEVED SOURCES
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📖 Retrieved Sources"
    )

    for number, result in enumerate(
        results,
        start=1,
    ):

        file_name = result.get(
            "file_name",
            "Unknown",
        )

        page_number = result.get(
            "page_number",
            "Unknown",
        )

        similarity = result.get(
            "similarity_score",
            0.0,
        )

        with st.expander(
            f"{number}. {file_name} — Page {page_number}"
        ):

            st.write(
                f"**Similarity:** "
                f"{similarity:.4f}"
            )

            st.write(
                f"**Chunk:** "
                f"{result.get('chunk_index', 'N/A')}"
            )

            st.write(
                f"**Chunk ID:** "
                f"`{result.get('chunk_id', 'N/A')}`"
            )

            st.write(
                result.get(
                    "text",
                    "",
                )
            )
