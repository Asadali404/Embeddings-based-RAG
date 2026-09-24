
UNIVERSITY KNOWLEDGE BASE
=========================

Created:
2026-09-24T06:39:52.158698+00:00


DOCUMENTS
=========

Number of documents:
6


CHUNKS
======

Number of chunks:
106


EMBEDDING
=========

Model:
sentence-transformers/all-MiniLM-L6-v2

Dimension:
384


CHUNKING
========

Chunk size:
700

Chunk overlap:
120


FAISS
=====

Index:
faiss_index/index.faiss

Index type:
IndexFlatIP


METADATA
========

File:
metadata/metadata.json


The metadata contains:

- chunk ID
- document index
- file name
- source path
- page number
- chunk number
- chunk text
- character count
- word count
- embedding model
- chunk size
- chunk overlap
- document SHA-256 hash


RUNTIME ARCHITECTURE
====================

The deployed application does NOT need the
original PDF documents.

The application loads:

1. FAISS index
2. metadata.json
3. embedding model

Then:

User query
    ↓
Query embedding
    ↓
FAISS similarity search
    ↓
Vector IDs
    ↓
metadata.json
    ↓
Retrieved source text
    ↓
Groq LLM
    ↓
Answer + source citation


VECTOR MAPPING
==============

FAISS vector position corresponds to
the same position in metadata["chunks"].

Example:

FAISS vector 0
    ↓
metadata["chunks"][0]

FAISS vector 1
    ↓
metadata["chunks"][1]


IMPORTANT
=========

The same embedding model must be used
during application runtime for user queries.

Current model:

sentence-transformers/all-MiniLM-L6-v2
