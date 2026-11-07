import os
import re
import matplotlib.pyplot as plt

def parse_logs(log_path):
    steps, total_losses, mel_losses, pitch_losses, energy_losses, duration_losses = [], [], [], [], [], []
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Step\s+(\d+).*?Total Loss:\s*([\d\.]+).*?Mel Loss:\s*([\d\.]+).*?Pitch Loss:\s*([\d\.]+).*?Energy Loss:\s*([\d\.]+).*?Duration Loss:\s*([\d\.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                total_losses.append(float(m.group(2)))
                mel_losses.append(float(m.group(3)))
                pitch_losses.append(float(m.group(4)))
                energy_losses.append(float(m.group(5)))
                duration_losses.append(float(m.group(6)))
    return {
        'step': steps, 'total': total_losses, 'mel': mel_losses,
        'pitch': pitch_losses, 'energy': energy_losses, 'duration': duration_losses
    }

out_dir = "../Results/Telugu"
os.makedirs(out_dir, exist_ok=True)

train_data = parse_logs("../Results/Telugu/log/train/log.txt")
if train_data and train_data['step']:
    plt.figure(figsize=(10, 6))
    plt.plot(train_data['step'], train_data['total'], label='Total Loss', color='blue')
    plt.plot(train_data['step'], train_data['mel'], label='Mel Loss', color='orange', alpha=0.7)
    plt.plot(train_data['step'], train_data['pitch'], label='Pitch Loss', color='green', alpha=0.7)
    plt.plot(train_data['step'], train_data['energy'], label='Energy Loss', color='purple', alpha=0.7)
    plt.plot(train_data['step'], train_data['duration'], label='Duration Loss', color='red', alpha=0.7)
    plt.title('Telugu FastSpeech 2 - Training Loss Curves')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated training_loss.png")

val_data = parse_logs("../Results/Telugu/log/val/log.txt")
if val_data and val_data['step']:
    plt.figure(figsize=(10, 6))
    plt.plot(val_data['step'], val_data['total'], label='Validation Total Loss', color='darkblue')
    plt.plot(val_data['step'], val_data['mel'], label='Mel Loss', color='darkorange', alpha=0.7)
    plt.title('Telugu FastSpeech 2 - Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, 'validation_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated validation_loss.png")
