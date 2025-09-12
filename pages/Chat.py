import streamlit as st

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": ""},
    ]

if prompt := st.chat_input("Write a message..."):
    with st.chat_message("user"):
        st.markdown(prompt)