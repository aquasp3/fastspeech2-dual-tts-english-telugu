import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        pitch_targets = pitch_targets.float()
        energy_targets = energy_targets.float()
        mel_targets = mel_targets.float()
        
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # --- Align Phoneme Level Targets (Duration) ---
        if log_duration_targets.shape[1] != src_masks.shape[1]:
            if log_duration_targets.shape[1] > src_masks.shape[1]:
                log_duration_targets = log_duration_targets[:, :src_masks.shape[1]]
            else:
                log_duration_targets = F.pad(
                    log_duration_targets, (0, src_masks.shape[1] - log_duration_targets.shape[1])
                )

        # --- Align Pitch Targets ---
        target_pitch_mask = src_masks if self.pitch_feature_level == "phoneme_level" else mel_masks
        if pitch_targets.shape[1] != target_pitch_mask.shape[1]:
            if pitch_targets.shape[1] > target_pitch_mask.shape[1]:
                pitch_targets = pitch_targets[:, :target_pitch_mask.shape[1]]
            else:
                pitch_targets = F.pad(
                    pitch_targets, (0, target_pitch_mask.shape[1] - pitch_targets.shape[1])
                )

        # --- Align Energy Targets ---
        target_energy_mask = src_masks if self.energy_feature_level == "phoneme_level" else mel_masks
        if energy_targets.shape[1] != target_energy_mask.shape[1]:
            if energy_targets.shape[1] > target_energy_mask.shape[1]:
                energy_targets = energy_targets[:, :target_energy_mask.shape[1]]
            else:
                energy_targets = F.pad(
                    energy_targets, (0, target_energy_mask.shape[1] - energy_targets.shape[1])
                )

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        elif self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )