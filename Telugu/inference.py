import argparse
import json
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from utils.model import get_model, get_vocoder
from text.telugu_text import text_to_phones
from text import text_to_sequence

plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI Historic', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def synthesize_and_plot(model, vocoder, model_config, preprocess_config, test_sentences, restore_step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]

    aud_out = "../Results/Telugu/test_generated_audios"
    dur_out = "../Results/Telugu/variance_adaptor_plots/duration"
    pitch_out = "../Results/Telugu/variance_adaptor_plots/pitch"
    energy_out = "../Results/Telugu/variance_adaptor_plots/energy"

    for d in [aud_out, dur_out, pitch_out, energy_out]:
        os.makedirs(d, exist_ok=True)

    for tag, text in test_sentences.items():
        print(f"Synthesizing {tag}: {text}")
        
        phones_res = text_to_phones(text)
        phones_str = " ".join(phones_res) if isinstance(phones_res, list) else str(phones_res)
        raw_text = "{" + phones_str + "}"
        
        clean_seq = np.array(text_to_sequence(raw_text, preprocess_config["preprocessing"]["text"]["text_cleaners"]))
        tokens = torch.from_numpy(clean_seq).unsqueeze(0).to(device)
        src_len = torch.from_numpy(np.array([len(clean_seq)])).to(device)

        with torch.no_grad():
            output = model(
                speakers=torch.tensor([0]).to(device),
                texts=tokens,
                src_lens=src_len,
                max_src_len=len(clean_seq)
            )
            
            mel_pred = output[1]
            pitch_pred = output[2]
            energy_pred = output[3]
            log_dur_pred = output[4]
            
            # Vocoder audio generation
            mel_target = mel_pred.transpose(1, 2)
            wav = vocoder(mel_target).squeeze().cpu().numpy()
            
            wav_path = os.path.join(aud_out, f"{tag}.wav")
            wavfile.write(wav_path, preprocess_config["preprocessing"]["audio"]["sampling_rate"], (wav * 32767).astype(np.int16))

            phones = phones_str.split()
            n_phones = len(phones)
            
            durations = np.exp(log_dur_pred[0, :n_phones].cpu().numpy()) - 1
            durations = np.clip(np.round(durations), 1, None).astype(int)
            
            pitch = pitch_pred[0, :n_phones].cpu().numpy() * p_std + p_mean
            energy = energy_pred[0, :n_phones].cpu().numpy() * e_std + e_mean

            # 1. Duration Plot
            plt.figure(figsize=(12, 4))
            plt.bar(range(n_phones), durations, color='skyblue', edgecolor='black')
            plt.xticks(range(n_phones), phones, rotation=45, fontsize=9)
            plt.title(f"Duration Adaptor Plot - {tag}", fontsize=12)
            plt.ylabel("Frames")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(dur_out, f"{tag}_duration.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Pitch Plot
            plt.figure(figsize=(12, 4))
            plt.plot(range(n_phones), pitch, marker='o', color='green', linewidth=2)
            plt.xticks(range(n_phones), phones, rotation=45, fontsize=9)
            plt.title(f"Pitch Adaptor Plot - {tag}", fontsize=12)
            plt.ylabel("Hz")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(pitch_out, f"{tag}_pitch.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Energy Plot
            plt.figure(figsize=(12, 4))
            plt.plot(range(n_phones), energy, marker='s', color='purple', linewidth=2)
            plt.xticks(range(n_phones), phones, rotation=45, fontsize=9)
            plt.title(f"Energy Adaptor Plot - {tag}", fontsize=12)
            plt.ylabel("RMS Energy")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(energy_out, f"{tag}_energy.png"), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("-p", "--preprocess_config", type=str, default="config/preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, default="config/config.yaml")
    parser.add_argument("-t", "--train_config", type=str, default="config/train.yaml")
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, configs=(preprocess_config, model_config, train_config), device=device, train=False)
    vocoder = get_vocoder(model_config, device=device)

    test_sentences = {
        "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
        "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    }

    synthesize_and_plot(model, vocoder, model_config, preprocess_config, test_sentences, args.restore_step)
