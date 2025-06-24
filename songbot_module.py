# songbot_module.py

import os
import pandas as pd
from tqdm import tqdm
from random import shuffle
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# CSV publik dari GitHub
CSV_FILE_PATH = "https://raw.githubusercontent.com/annhaura/mood-song-recommender/main/spotify_songs.csv"

# Identitas sistem agent
system_identity = (
    "You are an emotionally-aware music recommender chatbot that responds with empathy, "
    "adapts to user's language, and explains song selections insightfully."
)

def create_vectorstore(df, embedding_model, batch_size=100):
    combined = df["combined_text"].tolist()
    documents = []

    print("🔄 Embedding songs in batches...")

    for i in tqdm(range(0, len(combined), batch_size)):
        batch = combined[i:i + batch_size]
        try:
            embeddings = embedding_model.embed_documents(batch)
            for j, emb in enumerate(embeddings):
                doc = Document(page_content=batch[j], metadata={"index": i + j})
                documents.append(doc)
        except Exception as e:
            print(f"⚠️ Skipped batch {i}-{i + batch_size} due to error: {e}")

    print(f"✅ Total embedded documents: {len(documents)}")
    return FAISS.from_documents(documents, embedding_model)

def create_agent(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key

    # LLM dan embedding
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load data CSV
    df = pd.read_csv(CSV_FILE_PATH)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    vectorstore = create_vectorstore(df, embedding_model)

    chat_history = []
    last_recommendation = ""

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

    def retrieve_similar_songs(query: str):
        nonlocal last_recommendation
        results = vectorstore.similarity_search(query, k=10)
        selected = randomize_results(results)

        song_lines = []
        for doc in selected:
            raw_text = doc.page_content
            escaped = raw_text.replace("{", "{{").replace("}", "}}")
            display = raw_text.replace("{", "").replace("}", "")
            song_lines.append({
                "escaped": f"🎶 {escaped}",
                "display": f"🎶 {display}"
            })

        last_recommendation = "\n".join([line["escaped"] for line in song_lines])
        return "\n".join([line["display"] for line in song_lines])

    def explain_recommendation(query: str, context: str = "") -> str:
        prompt = f"""
{system_identity}

Conversation history (if any):
{context}

User input: {query}
Recommended songs:
{last_recommendation}

Explain why these songs are suitable. Include breakdowns of lyrics, genre, and mood. Maintain user language style.
"""
        return llm.invoke(prompt).content.strip()

    def smart_rag_response(user_input: str) -> str:
        lang = detect_language(user_input)
        input_en = translate_input(user_input) if lang != "en" else user_input

        inferred_genre = map_genre(input_en)
        songs = retrieve_similar_songs(f"{input_en}, genre: {inferred_genre}")
        context = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history[-2:]])
        explanation = explain_recommendation(input_en, context)

        full_response = f"\nHere are songs I picked for you:\n\n{songs}\n\n{explanation}"
        if lang != "en":
            full_response = translate_back(full_response, lang)

        chat_history.append((user_input, full_response))
        return full_response

    return smart_rag_response
