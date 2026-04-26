import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = st.selectbox(
    "Select a model", 
    "gemini-2.5-flash", 
    index=0
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
uploaded_files = st.file_uploader("Dołącz pliki (opcjonalnie)", accept_multiple_files=True, help="Załącz pliki; ich treść zostanie dołączona do wiadomości przed wysłaniem.")
if uploaded_files:
    for f in uploaded_files:
        try:
            data = f.getvalue()
            # próba dekodowania jako tekst
            if isinstance(data, (bytes, bytearray)):
                try:
                    text = data.decode("utf-8")
                    snippet = text if len(text) <= 5000 else text[:5000] + "\n...[przycięte]"
                    st.write(f"Załączono plik tekstowy: {f.name}")
                    st.code(snippet)
                    st.session_state.messages.append({"role": "user", "content": f"Załącznik {f.name}:\n{snippet}"})
                    continue
                except UnicodeDecodeError:
                    pass

            # jeśli to obraz — wyświetl i dodaj notkę
            if f.type and f.type.startswith("image/"):
                st.image(data, caption=f.name)
                st.session_state.messages.append({"role": "user", "content": f"Załączono obraz {f.name} (wyświetlony)."})
            else:
                # plik binarny lub nieobsługiwany — dodaj notkę bez treści
                st.warning(f"Nie można odczytać zawartości pliku jako tekst: {f.name}. Dołączono notatkę zamiast treści.")
                st.session_state.messages.append({"role": "user", "content": f"Załączono plik: {f.name} (zawartość nie dołączona)."})
        except Exception as e:
            st.error(f"Błąd przy wczytywaniu pliku {f.name}: {e}")

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=st.session_state.messages
        )

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        # To wyświetli dokładny komunikat błędu z serwera na czerwonym tle w aplikacji
        st.error(f"Wystąpił błąd API: {e}")
        st.stop()