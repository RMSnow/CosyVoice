from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=True, load_onnx=False, load_trt=False
)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav("zero_shot_prompt.wav", 16000)
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        # "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        # "推荐的多次推荐的，指数是标普五，百指数这。一指数以，大盘股为。主包含五，百只成分股。然，而它并非单，纯按照市值。规模选股，其中包含百分之九十的。大公司和百，分之十。的中型公司。",
        # "推荐的多次推荐的！指数是标普五！百指数这。一指数以！大盘股为。主包含五，百只成分股。然？而它并非单，纯按照市值！规模选股，其中包含百分之九十的？大公司和百！分之十？的中型公司。",
        # "推见嘚哆次推见嘚，指枢是标扑午，摆指枢这。一指枢以，打盘股唯。猪包寒午，摆只成份股。然，尔它并匪单，存按照市值。规某选股，其中包寒摆分之久十嘚。打公厮和摆，分之十。嘚钟型公厮",
        "坐玉石倚玉枕拂金徽谪仙何处无人伴我白螺杯我为灵芝仙草不为朱唇丹脸长啸亦何为醉舞下山去明月逐人归。",
        "希望你以后能够做的比我还好呦。",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save(
        "zero_shot_{}_yuancheng.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(
#     cosyvoice.inference_cross_lingual(
#         "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
#         prompt_speech_16k,
#         stream=False,
#     )
# ):
#     torchaudio.save(
#         "fine_grained_control_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )

# # instruct usage
# for i, j in enumerate(
#     cosyvoice.inference_instruct2(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         "用四川话说这句话",
#         prompt_speech_16k,
#         stream=False,
#     )
# ):
#     torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)
