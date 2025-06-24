import streamlit as st
from songbot_module import create_agent

st.set_page_config(page_title="🎧 Mood Song Recommender", page_icon="🎶")
st.title("🎶 Mood Song Recommender")

apikey = st.text_input("Masukkan API Key Gemini:", type="password")
if not apikey:
    st.warning("Masukkan API Key dulu ya!")
    st.stop()

agent = create_agent(apikey)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Ceritain mood-mu, ya."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ceritain mood atau genre..."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role":"assistant", "content":response})
