from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import os
from tqdm import tqdm
from glob import glob
import random
import json

# cosyvoice2 = CosyVoice2(
#     "pretrained_models/CosyVoice2-0.5B", load_jit=True, load_onnx=False, load_trt=False
# )


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


def get_text_wav_pairs(root):
    with open(os.path.join(root, "evalset.json"), "r", encoding="utf-8") as f:
        evalset = json.load(f)

    pairs = []
    for item in evalset:
        for t in ["input", "prompt"]:
            duration = item[t]["duration"]
            if duration < 5:
                continue

            text = item[t]["text"]
            uid = item[t]["uid"]

            wav_path = os.path.join(root, "wav", f"{uid}.wav")
            pairs.append((text, wav_path, uid, duration))

    return pairs


if __name__ == "__main__":
    root = "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/tts/"
    en_pairs = get_text_wav_pairs(os.path.join(root, "seedtts_en"))
    zh_pairs = get_text_wav_pairs(os.path.join(root, "seedtts_zh"))
    print("en_pairs:", len(en_pairs))
    print("zh_pairs:", len(zh_pairs))

    save_dir = (
        "/storage/zhangxueyao/workspace/CosyVoice/results/cosyvoice2_cross_lingual"
    )
    os.makedirs(save_dir, exist_ok=True)

    random.seed(42)
    N = 2000

    # EN prompt, ZH text
    res = []
    for prompt_text, prompt_wav_path, prompt_uid, prompt_duration in tqdm(en_pairs):
        for target_text, target_wav_path, target_uid, target_duration in zh_pairs:
            res.append(
                {
                    "input": {
                        "uid": target_uid,
                        "duration": target_duration,
                        "text": target_text,
                        "wav_path": target_wav_path,
                    },
                    "prompt": {
                        "uid": prompt_uid,
                        "duration": prompt_duration,
                        "text": prompt_text,
                        "wav_path": prompt_wav_path,
                    },
                    "output_path": f"{prompt_uid}#{target_uid}.wav",
                }
            )

    res = random.sample(res, N)
    with open(os.path.join(save_dir, "en2zh.json"), "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    print("en2zh", len(res))

    # ZH prompt, EN text
    res = []
    for prompt_text, prompt_wav_path, prompt_uid, prompt_duration in tqdm(zh_pairs):
        for target_text, target_wav_path, target_uid, target_duration in en_pairs:
            res.append(
                {
                    "input": {
                        "uid": target_uid,
                        "duration": target_duration,
                        "text": target_text,
                        "wav_path": target_wav_path,
                    },
                    "prompt": {
                        "uid": prompt_uid,
                        "duration": prompt_duration,
                        "text": prompt_text,
                        "wav_path": prompt_wav_path,
                    },
                    "output_path": f"{prompt_uid}#{target_uid}.wav",
                }
            )

    res = random.sample(res, N)
    with open(os.path.join(save_dir, "zh2en.json"), "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    print("zh2en", len(res))
