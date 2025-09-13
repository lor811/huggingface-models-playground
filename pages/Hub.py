from requests import HTTPError
import streamlit as st

from src.use_cases.hf_api_service import HfApiService

if "hf_api_service" not in st.session_state:
    st.session_state.hf_api_service = HfApiService()

hf_api_service = st.session_state.hf_api_service

with st.sidebar:
    st.title("🤗 Hub")

    hf_api_service.set_token(st.text_input(label="Token:", type="password"))

    st.title("🔎 Search Filters")

    query = st.text_input("Name:", placeholder="gpt, llama, deepseek...")
    author = st.text_input("Author:", placeholder="openai, Qwen, meta, google...")

    tag_options = hf_api_service.get_tag_options()
    tag = tag_options.get(
        st.selectbox(label="Task:", options=tag_options.keys(), index=0)
    )

    sort_options = hf_api_service.get_sort_options()
    sort = sort_options.get(
        st.selectbox(label="Sort by:", options=sort_options.keys(), index=0)
    )

    parameters_options = hf_api_service.get_num_params_options().keys()
    min_parameters, max_parameters = st.select_slider(
        "Parameter size:", options=parameters_options, value=("<1B", ">500B")
    )

    show_limit = st.select_slider("Show:", options=range(1, 101), value=20)


download_status = st.empty()
download_progress = st.empty()

st.subheader("Browse and download Hugging Face models")
try:
    with st.spinner("Loading models..."):
        models = hf_api_service.search_models(
            query, author, min_parameters, max_parameters, tag, sort, show_limit
        )

    def download_model_repo_callback(repo_id, pipeline_tag):
        try:
            files = hf_api_service.get_repo_files(repo_id)
            i = 0
            for filename in files:
                if i < len(files):
                    download_progress.progress(
                        value=(i + 1) / len(files),
                        text=f"Downloading { files[i] } ({i+1}/{len(files)})",
                    )

                elapsed, file_size = hf_api_service.download_file(
                    repo_id, filename, pipeline_tag
                )
                st.toast(
                    f"Downloaded {filename} in {elapsed} ({file_size})",
                    icon="✅",
                )
                i += 1

            st.success(
                f"Successfully downloaded {repo_id}",
                icon="✅",
            )
        except Exception as e:
            st.error(f"Download failed: {e}")

    for i, model in enumerate(models):
        modelId = model["modelId"]
        downloads = model["downloads"]
        likes = model["likes"]
        parameter_size = model["parameter_size"]
        download_size = model["download_size"]
        pipeline_tag = model["pipeline_tag"]
        if pipeline_tag == "text-generation" and "conversational" in model["tags"]:
            pipeline_tag = "conversational"

        def format_number(num):
            if num >= 1_000_000_000:
                return f"{num/1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif num >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return str(num)

        with st.container(
            border=True, vertical_alignment="center", horizontal_alignment="center"
        ):
            col1, col2 = st.columns([1, 14])

            with col1:
                st.markdown(f"## 🤗")

            with col2:
                st.markdown(f"### **[{modelId}](https://huggingface.co/{modelId})**")

                cols = st.columns(4)
                with cols[0]:
                    st.metric("Downloads", f"{format_number(downloads)}")

                with cols[1]:
                    st.metric("Likes", format_number(likes))

                with cols[2]:
                    st.metric(
                        "Parameters",
                        format_number(parameter_size) if parameter_size else "N/A",
                    )

                with cols[3]:
                    st.button(
                        f"⬇️ {download_size}",
                        key=modelId + str(i),
                        on_click=lambda model_id=modelId, pipeline_tag=pipeline_tag: download_model_repo_callback(
                            model_id, pipeline_tag
                        ),
                    )
except HTTPError as e:
    if e.response.status_code == 401:
        st.error("HTTP 401 - Unauthorized -> Token may not be valid.")
    else:
        st.error(str(e))
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    raise e
