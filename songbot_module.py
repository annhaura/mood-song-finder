# songbot_module.py

import os
import pandas as pd
from random import shuffle
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
import re

# Load CSV (from GitHub)
CSV_URL = "https://raw.githubusercontent.com/annhaura/mood-song-recommender/main/spotify_songs.csv"

@staticmethod
def normalize(text):
    return text.lower().strip() if isinstance(text, str) else ""

def load_vectorstore(api_key):
    df = pd.read_csv(CSV_URL)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    docs = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(df["combined_text"])]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)

def create_agent(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    vectorstore = load_vectorstore(api_key)

    # Tool 1: Genre Detector
    def detect_genre(query):
        prompt = f"Guess the genre from this mood or query:\n\n{query}\n\nRespond with a genre (e.g., pop, chill, rock):"
        return llm.invoke(prompt).content.strip()

    # Tool 2: Simple Search by keyword
    def simple_search(query):
        df = pd.read_csv(CSV_URL)
        mask = df["track_name"].str.contains(query, case=False, na=False) | df["track_artist"].str.contains(query, case=False, na=False)
        results = df[mask].head(5)
        if results.empty:
            return "Tidak ditemukan lagu yang cocok dengan kata kunci."
        return "\n".join([f"🎵 {row['track_name']} - {row['track_artist']}" for _, row in results.iterrows()])

    # Tool 3: Vector Search
    def rag_song_search(query):
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "Tidak ada lagu yang mirip ditemukan."
        return "\n".join([f"🎶 {doc.page_content}" for doc in docs])

    # Tool 4: Explain Reason
    def explain_reason(query):
        results = rag_song_search(query)
        prompt = f"""
You're a music expert. Based on user's mood/query: "{query}", these songs were recommended:
{results}

Explain why each song fits. Include genre, lyrics, and vibe. Make it warm and human.
"""
        return llm.invoke(prompt).content.strip()

    # Tool 5: Mood to Vibe
    def mood_to_vibe(query):
        prompt = f"Convert this mood into a short vibe sentence (no more than 1 line):\n{query}"
        return llm.invoke(prompt).content.strip()

    # Tool 6: Random Playlist Generator
    def random_playlist(_):
        df = pd.read_csv(CSV_URL)
        sample = df.sample(3)
        return "Here's a random vibe check:\n" + "\n".join([f"🔀 {row['track_name']} - {row['track_artist']}" for _, row in sample.iterrows()])

    tools = [
        Tool(name="GenreDetectorTool", func=detect_genre, description="Deteksi genre dari input mood atau perasaan."),
        Tool(name="SimpleSearchTool", func=simple_search, description="Cari lagu berdasarkan kata kunci judul atau artis."),
        Tool(name="RAGSongTool", func=rag_song_search, description="Cari lagu mirip menggunakan RAG dan vectorstore."),
        Tool(name="ExplainRecommendationTool", func=explain_reason, description="Jelaskan alasan lagu-lagu cocok dengan mood."),
        Tool(name="MoodToVibeTool", func=mood_to_vibe, description="Ubah input mood jadi vibe singkat."),
        Tool(name="RandomPlaylistTool", func=random_playlist, description="Berikan playlist acak sebagai kejutan.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools=tools, llm=llm, memory=memory, agent="zero-shot-react-description", verbose=False)

    return agent
