from typing import Optional, Union
import torch
import sphn

from moshi.run_inference import *


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

        state = InferenceState(
            self.model.model_type, self.mimi, self.text_tokenizer, self.lm,
            batch_size, cfg_coef, self.device, **self.model.lm_gen_config)
        out_items = state.run(in_pcms)

        return out_items, sample_rate

        # outfile = Path(output_path)
        # for index, (_, out_pcm) in enumerate(out_items):
        #     if len(out_items) > 1:
        #         outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
        #     else:
        #         outfile_ = outfile
        #     duration = out_pcm.shape[1] / self.mimi.sample_rate
        #     sphn.write_wav(str(outfile_), out_pcm[0].numpy(), sample_rate=self.mimi.sample_rate)

        # return outfile


# hibiki_inferencer = HibikiInferencer()
# hibiki_inferencer.load_model(
#     hf_repo="kyutai/hibiki-1b-pytorch-bf16",
#     dtype=torch.float16,
# )
# result, sr = hibiki_inferencer.predict(
#     input=r"C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-hibiki\output.mp3",
# )




