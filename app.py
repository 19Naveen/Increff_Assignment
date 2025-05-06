import streamlit as st
from chatbot import process_query
from setup import setup

setup()
st.title("AI Assistant Chatbot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for i, message in enumerate(st.session_state['messages']):
    if message['role'] == 'user':
        st.markdown(f"**You**: {message['content']}")
    else:
        st.markdown(f"**Bot**: {message['content']}")

user_message = st.text_input("Ask a question:")
if st.button('Send'):
    if user_message:
        st.session_state['messages'].append({"role": "user", "content": user_message})
        response = process_query(user_message)
        st.session_state['messages'].append({"role": "bot", "content": response})
