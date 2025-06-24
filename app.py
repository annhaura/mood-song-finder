import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from random import shuffle

st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood, powered by AI.")

api_key = st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to start.", icon="🔑")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

csv_url = "https://raw.githubusercontent.com/annhaura/mood-song-recommender/main/spotify_songs.csv"
df = pd.read_csv(csv_url)
df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(df["combined_text"])]
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embedding_model)

def retrieve_similar_songs(query: str, k=10) -> str:
    results = vectorstore.similarity_search(query, k=k)
    songs = [f"🎵 {doc.page_content}" for doc in results]
    return "\n".join(songs)

def infer_genre(query: str) -> str:
    prompt = f"Given this user mood or query, suggest a suitable music genre (pop, rock, acoustic, dance, nostalgic, etc):\n\n{query}"
    return llm.invoke(prompt).content.strip()

def explain_choice(query: str, songs: str) -> str:
    prompt = f"You are a thoughtful music recommender.\n\nUser mood/query: {query}\nRecommended songs:\n{songs}\n\nExplain briefly why these songs fit the mood."
    return llm.invoke(prompt).content.strip()

def classify_mood(query: str) -> str:
    prompt = f"Classify the emotional mood of this text (examples: happy, sad, nostalgic, energetic, calm, angry, romantic):\n\n{query}"
    return llm.invoke(prompt).content.strip().lower()

def randomize_list(text_block: str) -> str:
    lines = text_block.strip().splitlines()
    shuffle(lines)
    return "\n".join(lines)

tools = [
    Tool(name="RetrieveSimilarSongs", func=retrieve_similar_songs, description="Finds songs matching the user's mood or genre."),
    Tool(name="InferGenre", func=infer_genre, description="Suggests a music genre based on the user's mood or query."),
    Tool(name="ExplainChoice", func=explain_choice, description="Explains why recommended songs are suitable."),
    Tool(name="MoodClassifier", func=classify_mood, description="Classifies emotional mood from user input."),
    Tool(name="Randomizer", func=randomize_list, description="Randomizes order of recommendations for variation."),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("Thinking..."):
        response = agent.run(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", response))

for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)
