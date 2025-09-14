import streamlit as st

from src.use_cases.text_generation_service import TextGenerationService

if "text_generation_service" not in st.session_state:
    st.session_state.text_generation_service = TextGenerationService()

text_generation_service = st.session_state.text_generation_service

models = text_generation_service.get_conversational_assistants_list()

if not models:
    st.warning(
        "No text-generation model found. You can go to the 🤗 Hub and install one."
    )
else:
    with st.sidebar:
        st.title("📝 Text Generation")

        @st.cache_resource(show_spinner=True)
        def get_assistant_cached(id: str):
            assistant = text_generation_service.set_assistant(id)
            return assistant

        with st.spinner("Loading model..."):
            selected_model_id = st.selectbox(label="Model:", options=models)
            cached_assistant = get_assistant_cached(selected_model_id)

        stream = st.checkbox("Stream response")

        st.divider()
        st.title("🔧 Generation Config")

        badge_placeholder = st.empty()

        max_new_tokens = st.number_input(
            "max_new_tokens", min_value=20, max_value=256, value=30, step=10
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
                "temperature", min_value=0.0, max_value=1.0, value=0.8, step=0.1
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

    with st.chat_message("assistant"):
        st.markdown("Give me a prompt! 👇")
    
    text_input_palceholder = st.empty()
    assistant_response_placeholder = st.container()
    if prompt := text_input_palceholder.text_input("Prompt:"):
        with assistant_response_placeholder:
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
                        streamer = text_generation_service.send(prompt, stream=True, **kwargs)
                        for chunk in streamer:
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    else:
                        with st.spinner("Generating response..."):
                            full_response = text_generation_service.send(
                                prompt, stream=False, **kwargs
                            )
                        response_placeholder.markdown(full_response)
                except Exception as e:
                    response_placeholder.error(f"Something went wrong: {e}")