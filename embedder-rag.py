import faiss
import numpy as np
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Ładowanie modelu Z CACHEM - wykona się tylko raz przy uruchomieniu aplikacji
@st.cache_resource
def get_embedding_model():
    embed_model_id = "sentence-transformers/all-MiniLM-L6-v2" 
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    return HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_embedding, k=3):
        D, I = self.index.search(query_embedding, k)
        results = []
        for idx in I[0]:
            if idx != -1: 
                results.append(self.metadata[idx])
        return results

def create_index(chunked_documents):
    """Tworzy indeks FAISS na podstawie PODZIELONYCH dokumentów (chunks)."""
    if not chunked_documents:
        return None
        
    embeddings = get_embedding_model()
    
    # Wyciągamy treść fragmentów do zmapowania na wektory
    texts = [doc["text"] for doc in chunked_documents]
    metadata = chunked_documents 

    # Zamiana fragmentów tekstu na wektory
    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    dimension = embeddings_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    """Przeszukuje bazę FAISS dla zadanego pytania."""
    if faiss_index is None:
        return []
        
    embeddings = get_embedding_model()
    
    # Embeddowanie zapytania użytkownika
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    
    # Wyszukanie k najbliższych fragmentów
    results = faiss_index.similarity_search(query_embedding, k=k)
    return results