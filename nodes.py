import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time
import torch

import folder_paths
from comfy.utils import ProgressBar

from .modules.hibiki_inferencer import HibikiInferencer


custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
# custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "my-custom-nodes")
custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "my-custom-nodes")


def get_category_name():
    return "ComfyUI hibiki"


class HibikiModelLoader:
    #  Define the input parameters of the node here.
    @classmethod
    def INPUT_TYPES(s):
        repo_ids = ["kyutai/hibiki-1b-pytorch-bf16", "kyutai/hibiki-2b-pytorch-bf16"]

        return {
            #  If the key is "required", the value must be filled.
            "required": {
                "repo_id": (repo_ids,),
                "device": (['cuda', 'cpu'],),
                "dtype": (['bf16', 'fp16'],),
            },
        }

    RETURN_TYPES = ("HIBIKI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = get_category_name()

    def load_model(self,
                   repo_id: str,
                   device: str,
                   dtype: str = None,
                   ):
        hibiki_inferencer = HibikiInferencer()
        hibiki_inferencer.load_model(
            hf_repo=repo_id,
            device=device,
            dtype=dtype,
        )

        return (hibiki_inferencer, )


class SpeechToSpeechTranslation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HIBIKI_MODEL", ),
                "audio": ("AUDIO", ),
                "batch_size": ("INT", {"default": 5}),
                "cfg_coef": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "predict"
    CATEGORY = get_category_name()

    def predict(self,
                model: HibikiInferencer,
                audio: Union[dict, torch.Tensor],
                batch_size: int,
                cfg_coef: float,
                ):
        if isinstance(audio, dict):
            audio, sr = audio["waveform"], audio["sample_rate"]
        result, sr = model.predict(
            input=audio,
            batch_size=batch_size,
            cfg_coef=cfg_coef,
        )

        return ({"waveform": result, "sample_rate": sr},)

