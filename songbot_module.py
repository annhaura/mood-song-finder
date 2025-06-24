import os, pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import re

CSV_URL = "https://raw.githubusercontent.com/annhaura/mood-song-recommender/main/spotify_songs.csv"

def load_data_and_vectorstore():
    df = pd.read_csv(CSV_URL)
    df["combined"] = df["track_name"] + " by " + df["track_artist"]
    docs = [Document(page_content=t, metadata={"source": t}) for t in df["combined"]]
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = FAISS.from_documents(docs, embed)
    return df, vs

def create_agent(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    df, vectorstore = load_data_and_vectorstore()
    memory = ConversationBufferMemory(memory_key="chat_history")

    def genre_mapper(query):
        p = PromptTemplate(
            input_variables=["text"],
            template="Infer a song genre from this mood or preference. Respond with one word: {text}"
        )
        return llm.invoke(p.format(text=query)).content.strip()

    def lang_detector(query):
        p = PromptTemplate(input_variables=["text"],
            template="Detect language of this: {text}. Just give ISO code.")
        return llm.invoke(p.format(text=query)).content.strip()

    def translator(query):
        p = PromptTemplate(input_variables=["text"],
            template="Translate this to English: {text}")
        return llm.invoke(p.format(text=query)).content.strip()

    def rag_retriever(query):
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])

    def explain_song(query):
        p = PromptTemplate(input_variables=["songs", "query"],
            template="Explain why these songs match '{query}':\n{songs}")
        return llm.invoke(p.format(songs=query, query=query)).content.strip()

    tools = [
        Tool(name="GenreMapper", func=genre_mapper,
             description="Detect song genre from user mood"),
        Tool(name="LangDetector", func=lang_detector,
             description="Detect user's language code"),
        Tool(name="Translator", func=translator,
             description="Translate non-English to English"),
        Tool(name="RAGRetriever", func=rag_retriever,
             description="Retrieve top song matches from dataset"),
        Tool(name="ExplainSong", func=explain_song,
             description="Explain why songs fit the mood"),
    ]

    llm_chain_prompt = ("You are a music recommender. Use tools based on user query."
    " Always follow this process: detect language, translate if needed, map genre, retrieve songs, explain recommendations.")

    agent = initialize_agent(
        tools, llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )
    return agent
    
import streamlit as st

@st.cache_resource
def load_data_and_vectorstore():
    df = pd.read_csv(CSV_URL)
    df["combined"] = df["track_name"] + " by " + df["track_artist"]
    docs = [Document(page_content=t, metadata={"source": t}) for t in df["combined"]]
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = FAISS.from_documents(docs, embed)
    return df, vs

