import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "speaker1"

    wav_files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]
    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

    print("Formatting corpus files for MFA alignment...")
    for wav_name in tqdm(wav_files):
        base_name = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(in_dir, wav_name)
        lab_path = os.path.join(in_dir, f"{base_name}.lab")

        if os.path.exists(lab_path):
            with open(lab_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            text = _clean_text(text, cleaners)

            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            wav = wav / max(abs(wav) + 1e-7) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, f"{base_name}.wav"),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, f"{base_name}.lab"),
                "w",
                encoding="utf-8",
            ) as f1:
                f1.write(text)

