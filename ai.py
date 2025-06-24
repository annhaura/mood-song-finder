# ai.py
import streamlit as st
from songbot_module import create_agent

st.set_page_config(page_title="🎧 Mood-Based Song Recommender", page_icon="🎶")
st.title("🎶 Mood-Based Song Recommender")

GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Masukkan API key terlebih dahulu.")
    st.stop()

agent = create_agent(GOOGLE_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🎤 Hai! Ceritain suasana hatimu, nanti aku rekomendasikan lagu yang cocok~"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Masukkan mood, genre, atau suasana..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
