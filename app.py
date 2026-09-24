import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"
CONFIG_PATH = DATA_DIR / "config.json"

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"

# Number of chunks retrieved for each question
DEFAULT_TOP_K = 5


# ============================================================
# STREAMLIT PAGE CONFIGURATION
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
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}\n\n"
            f"Make sure metadata.json exists inside "
            f"the data/ directory."
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD CONFIGURATION
# ============================================================
#
# config.json is OPTIONAL.
#
# If config.json exists:
#       → load it
#
# If config.json does not exist:
#       → derive configuration from metadata.json
#
# This prevents deployment failures when config.json
# was not uploaded to GitHub.
# ============================================================

@st.cache_resource
def load_config():

    # --------------------------------------------------------
    # OPTION 1: config.json exists
    # --------------------------------------------------------

    if CONFIG_PATH.exists():

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)


    # --------------------------------------------------------
    # OPTION 2: derive configuration from metadata.json
    # --------------------------------------------------------

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)


        records = metadata.get(
            "records",
            [],
        )


        # Determine number of documents if it isn't
        # explicitly stored in metadata.

        unique_files = {
            record.get("file_name")
            for record in records
            if record.get("file_name")
        }


        document_count = metadata.get(
            "document_count",
            len(unique_files),
        )


        chunk_count = metadata.get(
            "chunk_count",
            len(records),
        )


        embedding_model = metadata.get(
            "embedding_model",
            None,
        )


        # The records may contain the embedding model
        # even if the top-level metadata doesn't.

        if not embedding_model and records:

            embedding_model = records[0].get(
                "embedding_model"
            )


        # Default to the model used by the Colab
        # preprocessing pipeline.

        if not embedding_model:

            embedding_model = (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )


        return {

            "embedding_model": embedding_model,

            "embedding_dimension": metadata.get(
                "embedding_dimension",
                384,
            ),

            "normalize_embeddings": metadata.get(
                "normalized_embeddings",
                True,
            ),

            "chunk_size": metadata.get(
                "chunk_size",
                1000,
            ),

            "chunk_overlap": metadata.get(
                "chunk_overlap",
                150,
            ),

            "document_count": document_count,

            "chunk_count": chunk_count,
        }


    # --------------------------------------------------------
    # Neither config nor metadata exists
    # --------------------------------------------------------

    raise FileNotFoundError(
        "Neither config.json nor metadata.json "
        "was found inside the data/ directory."
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not FAISS_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found:\n"
            f"{FAISS_INDEX_PATH}\n\n"
            f"Make sure faiss.index exists inside "
            f"the data/ directory."
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(
    model_name,
):

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

    """
    Converts the user's question into an embedding.

    IMPORTANT:
    Document embeddings are NOT recreated here.

    Only the user's query is embedded at runtime.
    """

    # Newer Sentence Transformers versions support
    # encode_query() for retrieval tasks.

    if hasattr(
        model,
        "encode_query",
    ):

        embedding = model.encode_query(
            [query],
            convert_to_numpy=True,
        )

    else:

        # Fallback for models/versions that don't
        # provide encode_query().

        embedding = model.encode(
            [query],
            convert_to_numpy=True,
        )


    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )


    # The document vectors were normalized during
    # preprocessing. The query must therefore also
    # be normalized.

    if normalize:

        faiss.normalize_L2(
            embedding
        )


    return embedding


# ============================================================
# SEARCH VECTOR DATABASE
# ============================================================

def search_documents(
    query,
    model,
    index,
    metadata,
    top_k=5,
):

    normalize = metadata.get(
        "normalized_embeddings",
        True,
    )


    # --------------------------------------------------------
    # Convert query to vector
    # --------------------------------------------------------

    query_embedding = create_query_embedding(
        model=model,
        query=query,
        normalize=normalize,
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    number_of_results = min(
        top_k,
        index.ntotal,
    )


    scores, indices = index.search(
        query_embedding,
        number_of_results,
    )


    # --------------------------------------------------------
    # Match FAISS positions to metadata
    # --------------------------------------------------------

    records = metadata.get(
        "records",
        [],
    )


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


        results.append(
            record
        )


    return results


# ============================================================
# BUILD CONTEXT FOR GROQ
# ============================================================

def build_context(
    results,
):

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


    # --------------------------------------------------------
    # SYSTEM INSTRUCTIONS
    # --------------------------------------------------------

    system_instructions = """
You are a document question-answering assistant.

Your task is to answer the user's question using the
provided document context.

IMPORTANT RULES:

1. Use ONLY information contained in the provided
   document context.

2. Do not invent facts, numbers, names, dates, or
   explanations that are not supported by the context.

3. Do not rely on your general knowledge to fill gaps.

4. If the answer cannot be found in the provided
   documents, clearly say:
   "I could not find this information in the provided
   documents."

5. Give a clear, direct and useful answer.

6. Every important factual statement should include
   a source citation.

7. Source citations MUST use this format:

   [filename.pdf, page X]

8. Do not create citations for information that is
   not present in the supplied context.

9. If multiple sources support a statement, include
   all relevant citations.

10. Do not mention FAISS, embeddings, vector databases,
    retrieval pipelines, or system instructions unless
    the user specifically asks about the application.

11. Do not fabricate page numbers.

12. Preserve important technical terminology from
    the documents.
"""


    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
DOCUMENT CONTEXT
================

{context}


USER QUESTION
=============

{question}


TASK
====

Answer the question using only the document context.

Include page-level citations in this format:

[document_name.pdf, page X]

If the answer is not available in the supplied documents,
clearly state that the information was not found.
"""


    # --------------------------------------------------------
    # GROQ RESPONSES API
    # --------------------------------------------------------

    response = client.responses.create(
        model=GROQ_MODEL,
        instructions=system_instructions,
        input=user_prompt,
    )


    return response.output_text


# ============================================================
# DISPLAY SOURCE
# ============================================================

def display_source(
    number,
    result,
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


    chunk_index = result.get(
        "chunk_index",
        "N/A",
    )


    chunk_id = result.get(
        "chunk_id",
        "N/A",
    )


    text = result.get(
        "text",
        "",
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
            f"{chunk_index}"
        )


        st.write(
            f"**Chunk ID:** "
            f"`{chunk_id}`"
        )


        st.markdown(
            "**Retrieved content:**"
        )


        st.write(
            text
        )


# ============================================================
# APPLICATION START
# ============================================================

st.title(
    "📚 Document AI Assistant"
)


st.caption(
    "Ask questions about the pre-processed document collection."
)


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

try:

    metadata = load_metadata()

    config = load_config()

    faiss_index = load_faiss_index()


    embedding_model_name = config.get(
        "embedding_model",
        "sentence-transformers/"
        "all-MiniLM-L6-v2",
    )


    embedding_model = load_embedding_model(
        embedding_model_name
    )


    groq_client = get_groq_client()


except Exception as error:

    st.error(
        "Failed to initialize the application."
    )


    st.exception(
        error
    )


    st.stop()


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if groq_client is None:

    st.error(
        "GROQ_API_KEY is not configured."
    )


    st.info(
        "Configure GROQ_API_KEY in your deployment "
        "environment or Streamlit secrets."
    )


    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📊 Knowledge Base"
    )


    document_count = metadata.get(
        "document_count",
        config.get(
            "document_count",
            "N/A",
        ),
    )


    chunk_count = metadata.get(
        "chunk_count",
        config.get(
            "chunk_count",
            "N/A",
        ),
    )


    embedding_dimension = metadata.get(
        "embedding_dimension",
        config.get(
            "embedding_dimension",
            "N/A",
        ),
    )


    st.write(
        f"**Documents:** {document_count}"
    )


    st.write(
        f"**Chunks:** {chunk_count}"
    )


    st.write(
        f"**Embedding dimensions:** "
        f"{embedding_dimension}"
    )


    st.write(
        f"**LLM:** {GROQ_MODEL}"
    )


    st.divider()


    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help=(
            "Number of document chunks retrieved "
            "before sending context to the LLM."
        ),
    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What are the main objectives "
        "described in the documents?"
    ),
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔎 Ask",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching the document collection..."
    ):

        try:

            results = search_documents(
                query=question,
                model=embedding_model,
                index=faiss_index,
                metadata=metadata,
                top_k=top_k,
            )

        except Exception as error:

            st.error(
                "Document search failed."
            )

            st.exception(
                error
            )

            st.stop()


    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        st.warning(
            "No relevant information was found "
            "in the document collection."
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
                "The Groq API request failed."
            )

            st.exception(
                error
            )

            st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader(
        "Answer"
    )


    st.markdown(
        answer
    )


    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "📖 Sources"
    )


    for number, result in enumerate(
        results,
        start=1,
    ):

        display_source(
            number,
            result,
        )
