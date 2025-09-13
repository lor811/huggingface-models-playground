import streamlit as st

pages = [
    st.Page("pages/Hub.py", title="🤗 Hub"),
    st.Page("pages/Chat.py", title="💬 Chat"),
    st.Page("pages/Text_Generation.py", title="📝 Text Generation"),
]

current_page = st.navigation(pages, position="top", expanded=True)
current_page.run()
