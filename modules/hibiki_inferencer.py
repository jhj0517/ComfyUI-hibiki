from typing import Optional, Union
import torch
import sphn
from pathlib import Path

from moshi.run_inference import *

# To fix https://github.com/jhj0517/ComfyUI-hibiki/issues/1
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class HibikiInferencer:
    def __init__(self):
        self.model = None
        self.mimi = None
        self.text_tokenizer = None
        self.lm = None
        self.device = None

    def load_model(self,
                   hf_repo: str,
                   device: str = "cuda",
                   dtype: str = "bf16",
                   ):
        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        self.device = device

        self.model = loaders.CheckpointInfo.from_hf_repo(
            hf_repo
        )
        self.mimi = self.model.get_mimi(device=device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.lm = self.model.get_moshi(device=device, dtype=dtype)

    def predict(self,
                input: Union[str, torch.Tensor],
                sample_rate: Optional[int] = None,
                batch_size: int = 8,
                cfg_coef: float = 1.0
                ):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model with `load_model()` first.")

        if sample_rate is None:
            sample_rate = self.mimi.sample_rate

        if isinstance(input, str):
            in_pcms, _ = sphn.read(input, sample_rate=sample_rate)
            in_pcms = torch.from_numpy(in_pcms).to(device=self.device)
            in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)
        else:
            in_pcms = input.to(device=self.device)
            in_pcms = in_pcms.expand(batch_size, -1, -1)

        state = InferenceState(
            self.model.model_type, self.mimi, self.text_tokenizer, self.lm,
            batch_size, cfg_coef, self.device, **self.model.lm_gen_config
        )
        out_items = state.run(in_pcms)

        out_items = [audio for _, audio in out_items]

        return out_items, sample_rate