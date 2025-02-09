import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time
import torch
import sphn

import folder_paths
from comfy.utils import ProgressBar

from .modules.hibiki_inferencer import HibikiInferencer

custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
# custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "my-custom-nodes")
# custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "my-custom-nodes")


def get_category_name():
    return "ComfyUI hibiki"


class HibikiModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        repo_ids = ["kyutai/hibiki-1b-pytorch-bf16", "kyutai/hibiki-2b-pytorch-bf16"]

        return {
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
                "audio": ("STRING", ),
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
                audio: Union[dict, str],
                batch_size: int,
                cfg_coef: float,
                ):
        if isinstance(audio, dict):
            audio, sr = audio["waveform"], audio["sample_rate"]
        else:
            audio, sr = audio, None

        outputs, sr = model.predict(
            input=audio,
            sample_rate=sr,
            batch_size=batch_size,
            cfg_coef=cfg_coef,
        )

        # Temporary pick first item only in ComfyUI.
        # The issue is about https://github.com/jhj0517/ComfyUI-hibiki/issues/3
        first_output_audio = outputs[0]
        first_output_audio = first_output_audio.unsqueeze(0).float()

        return ({"waveform": first_output_audio, "sample_rate": sr},)


class GetAudioFilePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {"default": None}),  # Adds an audio upload widget
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "return_str"
    CATEGORY = get_category_name()

    def return_str(self, audio_file: str):
        return (audio_file, )


# This node is ported from https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/8629188458dc6cb832f871ece3bd273507e8a766/videohelpersuite/nodes.py#L622
# Which is Kosinkadink's ComfyUI-VideoHelperSuite node.
class LoadAudioSphn:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                "audio": (sorted(files),),
            },
        }

    CATEGORY = get_category_name()

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio_sphn"

    def load_audio_sphn(self, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(self.strip_path(kwargs['audio']))
        return (self.read_with_sphn(audio_file, None), )

    def read_with_sphn(self, audio: str, sample_rate):
        return sphn.read(audio, sample_rate=sample_rate)

    @staticmethod
    def strip_path(path):
        path = path.strip()
        if path.startswith("\""):
            path = path[1:]
        if path.endswith("\""):
            path = path[:-1]
        return path
