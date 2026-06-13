import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """Pobiera tekst z pliku PDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Błąd podczas ładowania PDF {file_path}: {e}")
        return ""

def load_txt(file_path):
    """Pobiera tekst z pliku TXT."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Błąd podczas ładowania TXT {file_path}: {e}")
        return ""

def load_documents_from_folder(folder_path):
    """Ładuje wszystkie pliki PDF i TXT z podanego folderu."""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} nie istnieje.")
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ignorujemy podfoldery
        if not os.path.isfile(file_path):
            continue
            
        text = ""
        if filename.lower().endswith(".pdf"):
            text = load_pdf(file_path)
        elif filename.lower().endswith(".txt"):
            text = load_txt(file_path)
            
        # Dodajemy tylko, jeśli udało się wyciągnąć jakiś tekst
        if text.strip():
            documents.append({"filename": filename, "text": text})
            
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Dzieli długie dokumenty na mniejsze fragmenty (chunks).
    chunk_overlap (zakładka) sprawia, że nie ucinamy zdania/myśli w połowie.
    """
    # Używamy standardowego i bardzo dobrego narzędzia z LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Priorytety podziału (najpierw akapity, potem linie)
    )
    
    chunked_docs = []
    for doc in documents:
        # text_splitter dzieli tekst na listę stringów
        chunks = text_splitter.split_text(doc["text"])
        
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })
            
    return chunked_docs