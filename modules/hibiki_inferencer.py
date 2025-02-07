from typing import Optional
import sphn

from ..moshi.moshi.moshi.run_inference import *


class HibikiInferencer:
    def __init__(self):
        self.model = None
        self.mimi = None
        self.text_tokenizer = None
        self.lm = None

    def load_model(self,
                   hf_repo: str,
                   device: str = "cuda",
                   dtype: str = "bf16",
                   ):
        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16

        self.model = loaders.CheckpointInfo.from_hf_repo(
            hf_repo
        )
        self.mimi = self.model.get_mimi(device=device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.lm = self.model.get_moshi(device=device, dtype=dtype)

    def predict(self,
                input_path: str,
                output_path: str,
                device: str = "cuda",
                batch_size: int = 8,
                cfg_coef: float = 1.0
                ):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model with `load_model()` first.")

        in_pcms, _ = sphn.read(input_path, sample_rate=self.mimi.sample_rate)
        in_pcms = torch.from_numpy(in_pcms).to(device=device)
        in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)

        state = InferenceState(
            self.model.model_type, self.mimi, self.text_tokenizer, self.lm,
            batch_size, cfg_coef, device, **self.model.lm_gen_config)
        out_items = state.run(in_pcms)

        outfile = Path(output_path)
        for index, (_, out_pcm) in enumerate(out_items):
            if len(out_items) > 1:
                outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            else:
                outfile_ = outfile
            duration = out_pcm.shape[1] / self.mimi.sample_rate
            log("info", f"writing {outfile_} with duration {duration:.1f} sec.")
            sphn.write_wav(str(outfile_), out_pcm[0].numpy(), sample_rate=self.mimi.sample_rate)

        return outfile



hibiki_inferencer = HibikiInferencer()
hibiki_inferencer.load_model(
    hf_repo="kyutai/hibiki-1b-pytorch-bf16"
)
hibiki_inferencer.predict(
    input_path=r"C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-hibiki\data_sample_fr_hibiki_crepes.mp3",
    output_path=r"C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-hibiki\output.wav"
)




