from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import os
from tqdm import tqdm
from glob import glob
import random
import json

cosyvoice2 = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=True, load_onnx=False, load_trt=False
)


def infer_zero_shot(target_text, prompt_speech_wav_path, prompt_text, output_path):
    prompt_speech_16k = load_wav(prompt_speech_wav_path, 16000)
    for i, j in enumerate(
        cosyvoice2.inference_zero_shot(
            target_text,
            prompt_text,
            prompt_speech_16k,
            stream=False,
            text_frontend=False,
        )
    ):
        torchaudio.save(output_path, j["tts_speech"], cosyvoice2.sample_rate)


if __name__ == "__main__":
    # path = "/storage/zhangxueyao/workspace/CosyVoice/results/cosyvoice2_cross_lingual/en2zh.json"
    # save_root = "/storage/zhangxueyao/workspace/CosyVoice/results/cosyvoice2_cross_lingual/en2zh"

    path = "/storage/zhangxueyao/workspace/CosyVoice/results/cosyvoice2_cross_lingual/zh2en.json"
    save_root = "/storage/zhangxueyao/workspace/CosyVoice/results/cosyvoice2_cross_lingual/zh2en"


    os.makedirs(save_root, exist_ok=True)

    with open(path, "r") as f:
        evalset = json.load(f)

    for item in tqdm(evalset):
        output_path = os.path.join(save_root, item["output_path"])

        if os.path.exists(output_path):
            continue

        infer_zero_shot(
            item["input"]["text"],
            item["prompt"]["wav_path"],
            item["prompt"]["text"],
            output_path,
        )
