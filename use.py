import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

#if you learning new model, please update PATH
PATH = "/home/pengejeen/paymon_v0.0.1/paimon_llama2-enhanced/tts_paimon/TTS/recipes/ljspeech/xtts_v2/run/training/paimon_ko-May-26-2024_11+23PM-58e6b6a"
TOKENIZER_PATH = "/home/pengejeen/paymon_v0.0.1/paimon_llama2-enhanced/tts_paimon/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"

print("Loading model...")
config = XttsConfig()
config.load_json(f"{PATH}/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir = f"{PATH}",
    vocab_path = TOKENIZER_PATH,
    use_deepspeed=False  
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["./datasets/wavs/2_audio.wav"]
)


print("Inference...")
out = model.inference(
    "아니 이게 정말 맞는거야?",
    "ko",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7
)
torchaudio.save("./output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)