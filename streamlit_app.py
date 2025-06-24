# streamlit_app.py

import os
import pandas as pd
import streamlit as st
from random import shuffle
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- App Setup ---
st.set_page_config(page_title="🎵 Mood-Based Song Recommender", layout="centered")
st.title("🎵 Mood-Based Song Recommender")

# --- Input API Key ---
api_key = st.text_input("Masukkan GOOGLE_API_KEY kamu:", type="password")
if not api_key:
    st.warning("🚨 Harap masukkan GOOGLE_API_KEY kamu terlebih dahulu.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Load Model ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV lagu kamu", type=["csv"])

# --- Helper Functions ---
def is_music_dataset(df: pd.DataFrame) -> bool:
    required_cols = {"track_name", "track_artist"}
    return required_cols.issubset(set(df.columns))

def detect_language(text: str) -> str:
    prompt = f"What language is this? Respond only with ISO code like 'id' or 'en'.\n\n{text}"
    return llm.invoke(prompt).content.strip().lower()

def translate_input(text: str) -> str:
    prompt = f"Translate this to English if it's not already:\n\n{text}"
    return llm.invoke(prompt).content.strip()

def translate_back(text: str, lang_code: str) -> str:
    prompt = f"You will translate the following text to language code '{lang_code}'. Text in double braces {{like this}} should not be translated.\n\nText:\n{text}"
    return llm.invoke(prompt).content.strip()

def map_genre(query: str) -> str:
    prompt = f"From this user mood or query, infer potential music genre (pop, rock, acoustic, dance, sadcore, etc).\n\nQuery: {query}"
    return llm.invoke(prompt).content.strip().lower()

def randomize_results(results, k=3):
    shuffle(results)
    return results[:k]

# --- Main Execution ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    if not is_music_dataset(df):
        st.warning("🚫 Dataset tidak valid. Harap upload file lagu dengan kolom seperti 'track_name' dan 'track_artist'.")
        st.stop()

    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(df["combined_text"].tolist())]
    vectorstore = FAISS.from_documents(documents, embedding_model)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.last_recommendation = ""

    user_input = st.text_input("Ketikkan suasana hatimu atau jenis lagu yang kamu inginkan:")

    if user_input:
        lang = detect_language(user_input)
        input_en = translate_input(user_input) if lang != "en" else user_input

        genre = map_genre(input_en)
        search_query = f"{input_en}, genre: {genre}"
        results = vectorstore.similarity_search(search_query, k=10)
        selected = randomize_results(results)

        song_lines = [f"🎶 {doc.page_content}" for doc in selected]
        st.session_state.last_recommendation = "\n".join(song_lines)

        context = "\n".join([f"User: {u}\nBot: {b}" for u, b in st.session_state.chat_history[-2:]])

        explain_prompt = f"""
You are an emotionally-aware music recommender chatbot.

Conversation history:
{context}

User input: {input_en}
Recommended songs:
{st.session_state.last_recommendation}

Explain why these songs are suitable. Include breakdowns of lyrics, genre, and mood.
        """
        explanation = llm.invoke(explain_prompt).content.strip()

        full_response = f"Here are songs I picked for you:\n\n{chr(10).join(song_lines)}\n\n{explanation}"
        if lang != "en":
            full_response = translate_back(full_response, lang)

        st.session_state.chat_history.append((user_input, full_response))

        st.markdown("---")
        st.subheader("🤖 Rekomendasi Lagu:")
        st.markdown(full_response)

        st.markdown("---")
        with st.expander("Lanjut? Ganti mood?"):
            follow_up = st.text_input("Tanya atau ganti mood di sini:", key="follow")
            if follow_up:
                st.session_state.chat_history.append(("[Mood Switch] " + follow_up, ""))
                st.experimental_rerun()
