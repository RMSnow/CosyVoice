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
    parser.add_argument("--start", type=int, help="起始pair索引", default=0)
    parser.add_argument(
        "--end", type=int, required=True, help="结束pair索引", default=12000
    )
    args = parser.parse_args()

    # Raw CosyVoice2 sampling: top_p = 0.8, top_k = 25, temperature = 1.0

    # top_p = 1.0
    # top_k = 40
    # temp_list = [0.6, 0.8, 1.0, 1.2, 1.4]

    # save_root = "/storage/zhangxueyao/dataset/data_rlhf/rlhfv1_repetition"
    # os.makedirs(save_root, exist_ok=True)

    top_p = 0.8
    top_k = 25
    temp_list = [1.0]

    save_root = "/storage/zhangxueyao/dataset/data_rlhf/rlhfv1_mistakable_prosody"
    os.makedirs(save_root, exist_ok=True)

    group_dirs = glob(os.path.join(save_root, "ar_soundstorm/*"))
    group_dirs.sort()

    for group_dir in tqdm(group_dirs):
        group_name = os.path.basename(group_dir)

        pair_dirs = glob(os.path.join(group_dir, "*"))
        pair_dirs.sort()

        for pair_dir in tqdm(pair_dirs):
            pair_name = os.path.basename(pair_dir)

            # 获取pair索引并检查是否在指定范围内
            pair_index = int(pair_name.split("_")[-1])
            if pair_index < args.start or pair_index >= args.end:
                continue

            pair_save_root = os.path.join(
                save_root, "cosyvoice2", group_name, pair_name
            )
            os.makedirs(pair_save_root, exist_ok=True)

            with open(os.path.join(pair_dir, "text.txt"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]

                prompt_text = lines[0]
                target_text = lines[1]

            # 1. Save prompt.mp3 and text.txt
            if pair_dir != pair_save_root:
                os.system(
                    'cp "{}" "{}"'.format(
                        os.path.join(pair_dir, "prompt.mp3"),
                        os.path.join(pair_save_root, "prompt.mp3"),
                    )
                )
                os.system(
                    'cp "{}" "{}"'.format(
                        os.path.join(pair_dir, "text.txt"),
                        os.path.join(pair_save_root, "text.txt"),
                    )
                )

            for temp in temp_list:
                output_audio_file = os.path.join(
                    pair_save_root, "output_recovered_audio_temp{}.wav".format(temp)
                )
                if os.path.exists(output_audio_file):
                    continue

                infer_zero_shot(
                    target_text=target_text,
                    prompt_speech_wav_path=os.path.join(pair_dir, "prompt.mp3"),
                    prompt_text=prompt_text,
                    output_path=output_audio_file,
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                )
