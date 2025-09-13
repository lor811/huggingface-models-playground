import os
import time
from huggingface_hub import HfApi, snapshot_download
import requests

from ..domain.models.constants import MODELS_DIR


class HfApiService:
    def __init__(self):
        self.hf_api = HfApi()
        self.token = None

    def set_token(self, token: str):
        self.token = token if token else None

    def get_tag_options(self):
        return {
            "Conversational": "conversational",
            "Text Generation": "text-generation",
        }

    def get_sort_options(self):
        return {
            "Trending": "trendingScore",
            "Most likes": "likes",
            "Most downloads": "downloads",
            "Recently created": "createdAt",
            "Recently updated": "lastModified",
        }

    def get_num_params_options(self):
        return {
            "<1B": 0,
            "3B": 3,
            "6B": 6,
            "9B": 9,
            "12B": 12,
            "24B": 24,
            "32B": 32,
            "64B": 64,
            "128B": 128,
            "256B": 256,
            ">500B": 10000,
        }

    def _get_model_infos(self, id):
        return self.hf_api.model_info(
            repo_id=id,
            files_metadata=True,
            token=self.token,
        )

    def _format_bytes(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024 ** 2):.2f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"

    def _calculate_download_size(self, siblings):
        if siblings and len(siblings) > 0:
            download_size = sum(file.size for file in siblings if file.size is not None)
            download_size = self._format_bytes(download_size)
        else:
            download_size = "N/A"
        return download_size

    def search_models(
        self,
        query,
        author,
        min_parameters,
        max_parameters,
        pipeline_tag="conversational",
        sort="trendingScore",
        limit=20,
    ):
        min_parameters = self.get_num_params_options().get(min_parameters)
        max_parameters = self.get_num_params_options().get(max_parameters)
        num_parameters = ""
        if min_parameters == 0 and max_parameters != 10000:
            num_parameters = f"max:{max_parameters}B"
        elif max_parameters == 10000 and min_parameters != 0:
            num_parameters = f"min:{min_parameters}B"
        elif min_parameters == 0 and max_parameters == 10000:
            num_parameters = ""
        else:
            num_parameters = f"min:{min_parameters}B,max:{max_parameters}B"

        filter = ""
        if pipeline_tag == "conversational":
            filter = pipeline_tag
            pipeline_tag = "text-generation"

        params = {
            "search": query,
            "author": author,
            "pipeline_tag": pipeline_tag,
            "filter": filter,
            "sort": sort,
            "num_parameters": num_parameters,
            "library": ["transformers", "diffusers"],
            "limit": limit,
        }

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.get(
            "https://huggingface.co/api/models",
            params=params,
            headers=headers,
        )
        response.raise_for_status()

        response = response.json()
        for model in response:
            model_info = self._get_model_infos(model["modelId"])
            if model_info.safetensors and model_info.safetensors.total:
                params = model_info.safetensors.total
            else:
                params = None

            model["parameter_size"] = params

            model["download_size"] = self._calculate_download_size(model_info.siblings)
        return response

    def get_repo_files(self, repo_id):
        return self.hf_api.list_repo_files(repo_id, token=self.token)

    def _download_file_from_repo(self, repo_id, filename, download_folder):
        return self.hf_api.hf_hub_download(
            repo_id,
            repo_type="model",
            filename=filename,
            local_dir=download_folder,
            force_download=False,
            token=self.token,
        )

    def download_file(self, repo_id, filename, pipeline_tag):
        download_folder = os.path.join(
            MODELS_DIR, pipeline_tag, repo_id.split("/")[1].replace("/", "_")
        )
        start_time = time.time()
        file_path = self._download_file_from_repo(repo_id, filename, download_folder)
        elapsed = time.time() - start_time
        file_size = os.path.getsize(file_path)
        return f"{elapsed:.2f}s", self._format_bytes(file_size)
