import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_embedding, k=3):
        # FAISS wyszukuje najbliższych sąsiadów
        D, I = self.index.search(query_embedding, k)
        results = []
        for idx in I[0]:
            # FAISS zwraca -1, jeśli poprosimy o więcej wyników niż jest dokumentów w bazie
            if idx != -1: 
                results.append(self.metadata[idx])
        return results

# Bardzo popularny, lekki model Sentence-Transformers (dobry na CPU)
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2" 
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    # 1. Załadowanie modelu embeddingowego
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    
    # 2. Zakładamy, że 'documents' to lista słowników, np. [{"filename": "plik.txt", "text": "treść"}]
    texts = [doc["text"] for doc in documents]
    metadata = documents # Nasze metadane to po prostu całe oryginalne słowniki

    # 3. Zamiana tekstu na wektory (macierz)
    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    # 4. Pobranie wymiaru wektora (dla modelu all-MiniLM to 384)
    dimension = embeddings_matrix.shape[1]

    # 5. Ustawienie indeksu przeszukiwania (IndexFlatL2 to standardowy pomiar odległości Euklidesowej)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    # 1. Załadowanie tego samego modelu embeddingowego
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    
    # 2. Embeddowanie zapytania (query)
    # UWAGA: FAISS wymaga wejściowej macierzy 2D, dlatego zamykamy wektor w dodatkowe nawiasy []
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    
    # 3. Zwrócenie wyników przeszukiwania
    results = faiss_index.similarity_search(query_embedding, k=k)
    
    return results