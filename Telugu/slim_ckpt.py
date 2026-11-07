import sys
import os
import torch

def slim_checkpoint(in_ckpt, out_ckpt):
    print(f"Loading {in_ckpt}...")
    ckpt = torch.load(in_ckpt, map_location='cpu')
    model_state = ckpt.get('model', ckpt)
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({'model': model_state}, out_ckpt)
    print(f"Slimmed checkpoint saved to {out_ckpt} ({os.path.getsize(out_ckpt) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python slim_ckpt.py <in_ckpt.pth.tar> <out_ckpt.pth.tar>")
    else:
        slim_checkpoint(sys.argv[1], sys.argv[2])
