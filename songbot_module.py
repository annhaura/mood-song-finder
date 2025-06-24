# songbot_module.py

import os
import pandas as pd
import random
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatGoogleGenerativeAI

CSV_URL = "https://raw.githubusercontent.com/annhaura/mood-song-recommender/main/spotify_songs.csv"
df = pd.read_csv(CSV_URL)
df.dropna(subset=["track_name", "track_artist"], inplace=True)
df["full_text"] = df["track_name"] + " by " + df["track_artist"]
docs = [Document(page_content=row, metadata={"index": i}) for i, row in enumerate(df["full_text"])]

def create_agent(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def detect_mood(text):
        prompt = f"What is the user's mood from this sentence?\n\n{text}"
        return llm.invoke(prompt).content.strip().lower()

    def map_genre(mood):
        prompt = f"Given the user's mood '{mood}', what genre of music would suit them? Just give genre."
        return llm.invoke(prompt).content.strip().lower()

    def retrieve_songs(query):
        result = retriever.get_relevant_documents(query)
        if result:
            return "\n".join([f"🎵 {doc.page_content}" for doc in result])
        return fallback_random_songs(query)

    def fallback_random_songs(_):
        sampled = random.sample(df["full_text"].tolist(), 3)
        return "🎲 Lagu acak untukmu:\n" + "\n".join([f"🎵 {s}" for s in sampled])

    def explain_recommendation(query):
        prompt = f"""
Kamu adalah chatbot musik yang pintar dan empatik.

User bilang: {query}
Berikut lagu yang kamu rekomendasikan:
{retrieve_songs(query)}

Jelaskan kenapa lagu-lagu ini cocok dengan kondisi user, berdasarkan genre, mood, atau lirik.
"""
        return llm.invoke(prompt).content.strip()

    def translate_if_needed(text, original_input):
        prompt = f"Translate this to the same language as user input: {original_input}\n\n{text}"
        return llm.invoke(prompt).content.strip()

    tools = [
        Tool(name="DetectUserMood", func=detect_mood, description="Deteksi mood dari input user."),
        Tool(name="MapGenreFromMood", func=map_genre, description="Pilih genre dari mood user."),
        Tool(name="RetrieveSongsRAG", func=retrieve_songs, description="Ambil lagu sesuai query dari FAISS."),
        Tool(name="FallbackRandomSongs", func=fallback_random_songs, description="Lagu acak jika RAG tidak dapat hasil."),
        Tool(name="ExplainRecommendation", func=explain_recommendation, description="Jelaskan alasan pemilihan lagu."),
        Tool(name="TranslateOutput", func=translate_if_needed, description="Terjemahkan hasil jika perlu.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )

    return agent
