# FastSpeech 2 Dual-Language TTS (English & Telugu)

This repository contains the complete implementation, configuration, and results for a dual-language Text-to-Speech system utilizing the **FastSpeech 2** architecture. The models were trained on English (Svarah) and Telugu (IndicTTS) datasets.

## Project Structure

```text
project/
│
├── English/                  # Source code and configs for English
│   ├── checkpoints/          # Weights-only 12000-step checkpoint (Git LFS)
│   ├── config.yaml           # Model configuration
│   └── inference.py          # Synthesis script
│
├── Telugu/                   # Source code and configs for Telugu
│   ├── checkpoints/          # Weights-only 12000-step checkpoint (Git LFS)
│   ├── config.yaml           # Model configuration
│   └── inference.py          # Synthesis script
│
├── Results/                  
│   ├── English/              # Audio, loss plots, and variance adaptor plots
│   └── Telugu/               # Audio, loss plots, and variance adaptor plots
│
├── Report/
│   └── Technical_Report.pdf  # Detailed analysis of methodology and results
│
├── requirements.txt          # Software dependencies
└── README.md                 # Project documentation

Quick Start & Reproducibility
1. Prerequisites
Ensure you have the following installed:

Python 3.8+
Git (Tested on v2.50.0.windows.1)
Git LFS (Tested on v3.6.1)

2. Installation
Clone the repository and install the dependencies:

git clone <your-github-repo-url>
cd project
pip install -r requirements.txt

3. Checkpoints & Git LFS
The 12000-step checkpoints exceed standard GitHub file size limits and are tracked via Git Large File Storage (LFS). Ensure you have Git LFS installed to pull the weights correctly:

git lfs install
git lfs pull

4. Running Inference
The inference scripts are pre-configured to utilize the Griffin-Lim vocoder and will automatically map GPU-trained checkpoints to CPU if necessary.

Synthesize English:

cd English
python inference.py --restore_step 12000

Synthesize Telugu:

cd Telugu
python inference.py --restore_step 12000

The generated .wav audio files and variance adaptor plots will be exported to the respective Results/ directory.

Features & Modifications
PyTorch 2.6 Compatibility: Handled the weights_only=False requirement to allow successful unpickling of checkpoint artifacts in newer PyTorch versions.

Direct Phoneme Synthesis: Configured to accept raw ARPAbet/Indic phoneme strings to guarantee robust processing of complex words and numerics without text-cleaner breakdown.

Author
Sathvik Konduri
