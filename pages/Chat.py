import re
import streamlit as st
import json

from src.use_cases.chat_service import ChatService

if "chat_service" not in st.session_state:
    st.session_state.chat_service = ChatService()
    st.session_state.chat_service.append_message(
        role="system", content="You are a helpful assistant."
    )
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

chat_service = st.session_state.chat_service

models = chat_service.get_conversational_assistants_list()

if not models:
    st.warning(
        "No conversational model found. You can go to the 🤗 Hub and install one."
    )
else:
    with st.sidebar:
        st.title("💬 Chat")

        @st.cache_resource(show_spinner=False)
        def set_assistant_cached(id: str):
            return chat_service.set_assistant(id)

        with st.spinner("Loading model..."):
            selected_model_id = st.selectbox(label="Model:", options=models)
            set_assistant_cached(selected_model_id)

        chat_service.set_system_message(
            st.text_input(
                label="System message:",
                placeholder="You are a helpful assistant.",
            )
            or "You are a helpful assistant."
        )
        stream = st.checkbox("Stream response")

        chat_buttons_cols = st.columns(2, gap="small", vertical_alignment="center")
        with chat_buttons_cols[0]:
            st.download_button(
                "Save history",
                icon="⬇️",
                data=json.dumps(chat_service.get_messages(), indent=2),
                file_name="chat_history.json",
                mime="application/json",
            )
        with chat_buttons_cols[1]:
            if st.button("Clear history", icon="🧹"):
                chat_service.clear_messages()
                chat_service.append_message(
                    role="system", content="You are a helpful assistant."
                )

        uploaded_file = st.file_uploader(
            label="⬆️ Load history:",
            type="json",
        )
        if uploaded_file and uploaded_file.file_id != st.session_state.uploaded_file_id:
            st.session_state.uploaded_file_id = uploaded_file.file_id
            try:
                messages = json.loads(uploaded_file.read().decode("utf-8"))

                correct_messages = []
                for m in messages:
                    if isinstance(m, dict) and set(m.keys()) == {"role", "content"}:
                        correct_messages.append(m)

                if not correct_messages:
                    raise Exception("Not a valid JSON format.")

                if correct_messages[0]["role"] != "system":
                    correct_messages.insert(
                        0, {"role": "system", "content": "You are a helpful assistant."}
                    )

                chat_service.set_messages(correct_messages)
            except Exception as e:
                chat_service.clear_messages()
                chat_service.append_message(
                    role="system", content="You are a helpful assistant."
                )
                st.error(f"Failed to load JSON: {e}")

        st.divider()
        st.title("🔧 Generation Config")

        badge_placeholder = st.empty()

        max_new_tokens = st.number_input(
            "max_new_tokens", min_value=20, max_value=4096, value=1024, step=100
        )

        do_sample = st.checkbox("do_sample", value=False)

        if not do_sample:
            badge_placeholder.badge("greedy decoding")
        elif do_sample:
            badge_placeholder.badge("multinomial sampling")

        temperature = 1.0
        top_k = 50
        top_p = 1.0
        if do_sample:
            temperature = st.number_input(
                "temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.1
            )
            top_k = st.number_input(
                "top_k", min_value=0, max_value=100, value=50, step=5
            )
            top_p = st.number_input(
                "top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.1
            )

        repetition_penalty = st.number_input(
            "repetition_penalty", min_value=1.0, max_value=2.0, value=1.0, step=0.1
        )

    def reasoning_expander(placeholder, reasoning_text, expanded=False):
        if reasoning_text != "":
            with placeholder.container():
                with st.expander(label="Reasoning", icon="🤔", expanded=expanded):
                    st.markdown(reasoning_text)

    for msg in chat_service.get_messages():
        if msg["role"] == "system":
            with st.chat_message("assistant"):
                st.caption(f"⚙️ System: {msg["content"]}")
                st.markdown("Let's chat! 👇")
            continue

        with st.chat_message(msg["role"]):
            reasoning_placeholder = st.empty()
            markdown_placeholder = st.empty()

            if msg["role"] == "assistant":
                reasoning_text = " ".join(
                    re.findall(r"<think>(.*?)</think>", msg["content"], flags=re.DOTALL)
                )
                message_text = re.sub(
                    r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL
                )
                reasoning_expander(reasoning_placeholder, reasoning_text, True)
                markdown_placeholder.markdown(message_text)
            else:
                markdown_placeholder.markdown(msg["content"])

    if prompt := st.chat_input("Write a message..."):
        st.html(
            """
                <style>
                section[data-testid="stSidebar"] {
                    pointer-events: none;
                    opacity: 0.5;
                }
                div[data-baseweb="textarea"] textarea {
                    pointer-events: none;
                    opacity: 0.5;
                    display: none;
                }
                </style>
                
            """
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }

            response_placeholder = st.empty()
            full_response = ""
            try:
                if stream:
                    streamer = chat_service.send(prompt, stream=True, **kwargs)
                    for chunk in streamer:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    with st.spinner("Generating response..."):
                        full_response = chat_service.send(
                            prompt, stream=False, **kwargs
                        )
                    response_placeholder.markdown(full_response)
            except Exception as e:
                response_placeholder.error(f"Something went wrong: {e}")
            finally:
                chat_service.append_message("assistant", full_response)
        st.rerun()
