from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import os
from tqdm import tqdm
from glob import glob

# from modelscope import snapshot_download

# snapshot_download("iic/CosyVoice2-0.5B", local_dir="pretrained_models/CosyVoice2-0.5B")

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=True, load_onnx=False, load_trt=False
)


def conversion(content_wav_path, reference_wav_path, output_path):
    # vc usage
    prompt_speech_16k = load_wav(reference_wav_path, 16000)
    source_speech_16k = load_wav(content_wav_path, 16000)
    for i, j in enumerate(
        cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)
    ):
        torchaudio.save(output_path, j["tts_speech"], 22050)


if __name__ == "__main__":
    # ========= EvalSet =========
    save_root = "/mnt/workspace/zhangxueyao/VersaVoice/results"
    model_name = "2024_CosyVoice2_VC"
    save_root = os.path.join(save_root, model_name)
    os.makedirs(save_root, exist_ok=True)

    evalset_root = "/mnt/workspace/zhangxueyao/VersaVoice/Evalset"
    group2eval = {"g0": "VCTK", "g1": "SeedEval", "g2": "L2Arctic", "g3": "ESD"}

    # ========= g0, g1 =========
    for group in ["g0", "g1"]:
        print("\nFor {}...".format(group))
        for i in tqdm(range(200)):
            filename = "{:04}.wav".format(i + 1)
            cont_name = "{:04}".format(i + 1)
            ref_name = cont_name

            content_wav_path = os.path.join(evalset_root, group, "content", filename)
            reference_wav_path = os.path.join(
                evalset_root, group, "reference", filename
            )

            save_dir = os.path.join(save_root, group)
            os.makedirs(save_dir, exist_ok=True)
            output_filename = "{}-{}-{}-{}.wav".format(
                model_name, group2eval[group], cont_name, ref_name
            )
            output_path = os.path.join(save_dir, output_filename)

            assert os.path.exists(content_wav_path)
            assert os.path.exists(reference_wav_path)
            # print(output_path)

            # Conversion
            conversion(content_wav_path, reference_wav_path, output_path)

    # ========= g2, g3 =========
    for group in ["g2", "g3"]:
        print("\nFor {}...".format(group))

        content_files = glob(os.path.join(evalset_root, group, "content", "*.wav"))
        reference_files = glob(os.path.join(evalset_root, group, "reference", "*.wav"))
        content_files.sort()
        reference_files.sort()

        assert len(content_files) == 30
        assert len(reference_files) == 6

        conversion_num = 0
        for content_wav_path in tqdm(content_files):
            for reference_wav_path in reference_files:
                cont_name = os.path.basename(content_wav_path).split(".")[0]
                ref_name = os.path.basename(reference_wav_path).split(".")[0]

                save_dir = os.path.join(save_root, group)
                os.makedirs(save_dir, exist_ok=True)
                output_filename = "{}-{}-{}-{}.wav".format(
                    model_name, group2eval[group], cont_name, ref_name
                )
                output_path = os.path.join(save_dir, output_filename)

                assert os.path.exists(content_wav_path)
                assert os.path.exists(reference_wav_path)
                # print(output_path)

                # Conversion
                conversion(content_wav_path, reference_wav_path, output_path)
                conversion_num += 1

        assert conversion_num == 180
        print("#Conversion = {}".format(conversion_num))
