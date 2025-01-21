import sys

sys.path.append("/storage/zhangxueyao/workspace/CosyVoice")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import os
from tqdm import tqdm
from glob import glob
import random
import json
import argparse

cosyvoice2 = CosyVoice2(
    "/storage/zhangxueyao/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B",
    load_jit=True,
    load_onnx=False,
    load_trt=False,
)


def infer_zero_shot(
    target_text,
    prompt_speech_wav_path,
    prompt_text,
    output_path,
    temperature=1.0,
    top_p=0.8,
    top_k=25,
):
    prompt_speech_16k = load_wav(prompt_speech_wav_path, 16000)
    for i, j in enumerate(
        cosyvoice2.inference_zero_shot(
            target_text,
            prompt_text,
            prompt_speech_16k,
            stream=False,
            text_frontend=False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    ):
        torchaudio.save(output_path, j["tts_speech"], cosyvoice2.sample_rate)


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_setting", type=str, required=True)
    args = parser.parse_args()

    save_root = args.save_root
    model_name = args.model_name
    eval_setting = args.eval_setting
    save_root = os.path.join(save_root, model_name)
    os.makedirs(save_root, exist_ok=True)

    # Raw CosyVoice2 sampling: top_p = 0.8, top_k = 25, temperature = 1.0
    top_k = 25
    top_p = 0.8
    temp = 1.0

    # ========= EvalSet =========
    evalset_root = "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/tts"

    if eval_setting == "seedtts":
        group2eval = {
            "seedtts_en": "seedtts_en",
            "seedtts_zh": "seedtts_zh",
        }
    elif eval_setting == "hard":
        group2eval = {
            "seedtts_en_hard": "seedtts_en_hard",
            "seedtts_zh_hard": "seedtts_zh_hard",
        }
    elif eval_setting == "crosslingual":
        group2eval = {
            "crosslingual_en2zh": "crosslingual_en2zh",
            "crosslingual_zh2en": "crosslingual_zh2en",
        }
    elif eval_setting == "codeswitching":
        group2eval = {
            "codeswitching_from_en": "codeswitching_from_en",
            "codeswitching_from_zh": "codeswitching_from_zh",
        }
    else:
        group2eval = {
            "seedtts_en": "seedtts_en",
            "seedtts_zh": "seedtts_zh",
            "seedtts_en_hard": "seedtts_en_hard",
            "seedtts_zh_hard": "seedtts_zh_hard",
            "crosslingual_en2zh": "crosslingual_en2zh",
            "crosslingual_zh2en": "crosslingual_zh2en",
            "codeswitching_from_en": "codeswitching_from_en",
            "codeswitching_from_zh": "codeswitching_from_zh",
        }

    print("-" * 20)
    print("Evalset: ", list(group2eval.keys()))
    print("-" * 20)

    for group in group2eval.keys():
        print("\nFor {}...".format(group))
        save_dir = os.path.join(save_root, group)
        os.makedirs(save_dir, exist_ok=True)

        evalset_json_path = os.path.join(evalset_root, group, "evalset.json")
        with open(evalset_json_path, "r", encoding="utf-8") as f:
            evalset = json.load(f)

        for item in tqdm(evalset):
            src_text = item["input"]["text"]
            ref_text = item["prompt"]["text"]
            ref_wav_path = os.path.join(
                evalset_root, group, "wav", "{}.wav".format(item["prompt"]["uid"])
            )
            output_filename = "{}-{}-{}".format(
                model_name, group2eval[group], item["output_path"]
            )

            output_path = os.path.join(save_dir, output_filename)
            if os.path.exists(output_path):
                continue

            infer_zero_shot(
                target_text=src_text,
                prompt_speech_wav_path=ref_wav_path,
                prompt_text=ref_text,
                output_path=output_path,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )
