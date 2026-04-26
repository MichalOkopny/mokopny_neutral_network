import streamlit as st
from openai import OpenAI
import PyPDF2  # Przywracamy obsługę PDF

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]

# ==========================================
# PASEK BOCZNY
# ==========================================
with st.sidebar:
    st.header("Ustawienia")
    
    # Poprawiona lista modeli
    selected_model = st.selectbox(
        "Select a model", 
        "gemini-2.5-flash", 
        index=0
    )
    
    st.divider()
    
    # Przenosimy uploader na pasek boczny
    uploaded_files = st.file_uploader(
        "Dołącz pliki (opcjonalnie)", 
        accept_multiple_files=True, 
        help="Załącz pliki tekstowe lub PDF. Ich treść zostanie dołączona do Twojego pytania."
    )

# ==========================================
# GŁÓWNY CZAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

# Wyświetlanie historii
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Obsługa nowego zapytania
if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Tworzymy bazowy tekst pytania
    full_prompt = prompt
    
    # Jeśli są pliki, dołączamy ich treść do pytania (w tle)
    if uploaded_files:
        full_prompt += "\n\n--- ZAŁĄCZONE PLIKI ---\n"
        
        for f in uploaded_files:
            file_extension = f.name.split(".")[-1].lower()
            
            try:
                # Obsługa PDF
                if file_extension == "pdf":
                    pdf_reader = PyPDF2.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            pdf_text += extracted + "\n"
                    
                    snippet = pdf_text if len(pdf_text) <= 5000 else pdf_text[:5000] + "\n...[przycięte]"
                    full_prompt += f"\nPlik PDF ({f.name}):\n{snippet}\n"
                    
                # Obrazy (tylko informacyjnie, sam tekstowy model ich nie "zobaczy" w ten sposób)
                elif f.type and f.type.startswith("image/"):
                    full_prompt += f"\n[Użytkownik załączył obraz: {f.name}, ale jako asystent tekstowy nie możesz go zobaczyć]\n"
                    
                # Zwykły tekst / kod
                else:
                    data = f.getvalue()
                    text = data.decode("utf-8")
                    snippet = text if len(text) <= 5000 else text[:5000] + "\n...[przycięte]"
                    full_prompt += f"\nPlik tekstowy ({f.name}):\n{snippet}\n"
                    
            except Exception as e:
                st.error(f"Błąd przy odczycie pliku {f.name}: {e}")

    # Pokazujemy użytkownikowi tylko to, co wpisał (żeby nie zaśmiecać UI)
    st.chat_message("user").write(prompt)
    
    # Ale do historii dodajemy pełny prompt z plikami, żeby model go dostał
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    
    # Komunikacja z API
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=st.session_state.messages
        )

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        
    except Exception as e:
        st.error(f"Wystąpił błąd API: {e}")
        st.stop()