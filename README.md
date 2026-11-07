# FastSpeech 2 Text-to-Speech: English (Svarah) and Telugu (IndicTTS)

Two separately trained FastSpeech 2 models, based on https://github.com/ming024/FastSpeech2.
Both use the HiFi-GAN (universal) vocoder and were trained for 12,000 steps (batch size 16).

## Structure

```text
English/   train.py preprocess.py inference.py config.yaml config/ checkpoints/12000.pth.tar
Telugu/    train.py preprocess.py inference.py config.yaml config/ checkpoints/12000.pth.tar
Results/   English/ and Telugu/: training_loss.png, validation_loss.png,
           test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/
Report/    Technical_Report.pdf
requirements.txt
```

`config.yaml` is the model configuration. `config/` holds the preprocess, train and model
configs that the scripts read.

## Setup

Python 3.11 and a CUDA 12.1 GPU are recommended.

```bash
pip install -r requirements.txt
git lfs install
git lfs pull
```

The weights-only checkpoints and `hifigan/generator_universal.pth.tar` are stored with Git LFS.

## Inference

Run from inside the language folder:

```bash
cd Telugu        # or: cd English
python inference.py --restore_step 12000
```

This writes the five test wavs and the duration, pitch and energy plots to `../Results/<Language>/`.

## Data and alignment

| | English (Svarah) | Telugu (IndicTTS, speaker1) |
|---|---|---|
| Subset | random ~6.2 h, seed 1234 | random, seed 1234, 1-11 s, ~6 h target |
| After alignment | 4,157 utterances | 3,167 utterances (5.02 h) |
| Train | 3,657 utterances (5.05 h) | 2,545 utterances (4.02 h) |
| Validation | 500 utterances (0.69 h) | 622 utterances (1.00 h) |
| Alignment | MMS_FA forced word alignment, phones estimated inside words | proportional character alignment (no forced aligner) |

## Telugu changes

- `text/symbols.py`: character-level Telugu symbols (U+0C00-U+0C7F)
- `text/telugu_text.py`: numbers, times and ordinals to Telugu words; text to characters with `sp` at boundaries
- `gen_telugu_textgrid.py`: subset selection and proportional character alignment
- `audio/stft.py`: device-independent; `preprocessor/preprocessor.py`: scaler fix and duration-based validation split (`val_hours`)

## Reproduce

Telugu (inside `Telugu/`):

```bash
python gen_telugu_textgrid.py --hours 6.0
python preprocess.py config/preprocess.yaml
python train.py -p config/preprocess.yaml -m config/config.yaml -t config/train.yaml
python plot_losses.py
python inference.py --restore_step 12000
```

English (inside `English/`):

```bash
python gen_english_textgrid.py
python preprocess.py config/preprocess.yaml
python train.py -p config/preprocess.yaml -m config/config.yaml -t config/train.yaml
python plot_losses.py
python inference.py --restore_step 12000
```

## Known limitations

Telugu output is only partially intelligible, mainly because of approximate proportional
alignment, character-level symbols and about 4 h of training data. Both models show mild
overfitting after early training. See `Report/Technical_Report.pdf`.

Author: Sathvik Konduri