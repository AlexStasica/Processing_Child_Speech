"""
Fine-tune MetricGAN+ model on your CHILDES-matched training data
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import numpy as np

# Fix for Windows symlink issues
import speechbrain
from speechbrain.utils.fetching import LocalStrategy

speechbrain.utils.fetching.DEFAULT_LOCAL_STRATEGY = LocalStrategy.COPY


class SpeechEnhancementDataset(Dataset):
    """Dataset for loading clean/noisy speech pairs"""

    def __init__(self, clean_folder, noisy_folder, max_length=64000):
        self.clean_folder = clean_folder
        self.noisy_folder = noisy_folder
        self.max_length = max_length

        # Get matching file pairs
        self.file_pairs = []
        clean_files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]

        for clean_file in clean_files:
            noisy_file = clean_file.replace('.wav', '_noisy.wav')
            if os.path.exists(os.path.join(noisy_folder, noisy_file)):
                self.file_pairs.append((clean_file, noisy_file))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        clean_file, noisy_file = self.file_pairs[idx]

        # Load clean audio
        clean_audio, sr = torchaudio.load(os.path.join(self.clean_folder, clean_file))

        # Load noisy audio
        noisy_audio, sr = torchaudio.load(os.path.join(self.noisy_folder, noisy_file))

        # Ensure single channel
        if clean_audio.shape[0] > 1:
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)

        # Truncate or pad to max_length
        if clean_audio.shape[1] > self.max_length:
            clean_audio = clean_audio[:, :self.max_length]
            noisy_audio = noisy_audio[:, :self.max_length]
        elif clean_audio.shape[1] < self.max_length:
            pad_length = self.max_length - clean_audio.shape[1]
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_length))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_length))

        return {
            'clean': clean_audio.squeeze(),
            'noisy': noisy_audio.squeeze(),
            'filename': clean_file
        }


class MetricGANFineTuner:
    """Fine-tune MetricGAN+ model"""

    def __init__(self, device='cpu'):
        self.device = device
        self.train_losses = []
        self.val_losses = []

        # Load pre-trained model
        print("Loading pre-trained MetricGAN+ model...")
        try:
            from speechbrain.utils.fetching import LocalStrategy
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying fallback approach...")
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": device}
            )

        # Access the underlying model components for training
        print("Model attributes:", dir(self.model))

        # Try different ways to access the model
        if hasattr(self.model, 'enhance_model'):
            self.enhancement_model = self.model.enhance_model
        elif hasattr(self.model, 'models'):
            self.enhancement_model = self.model.models[0]  # First model in the list
        elif hasattr(self.model, 'model'):
            self.enhancement_model = self.model.model
        elif hasattr(self.model, 'mods'):
            # SpeechBrain uses 'mods' for module dictionary
            print("Available modules:", list(self.model.mods.keys()))
            # Common module names for enhancement models
            if 'enhance_model' in self.model.mods:
                self.enhancement_model = self.model.mods['enhance_model']
            elif 'generator' in self.model.mods:
                self.enhancement_model = self.model.mods['generator']
            elif 'model' in self.model.mods:
                self.enhancement_model = self.model.mods['model']
            else:
                # Use the first available model
                model_key = list(self.model.mods.keys())[0]
                self.enhancement_model = self.model.mods[model_key]
                print(f"Using model: {model_key}")
        else:
            raise AttributeError("Cannot find the enhancement model in the loaded model")

        print(f"Enhancement model type: {type(self.enhancement_model)}")

        # Set up optimizer
        self.optimizer = optim.Adam(
            self.enhancement_model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999)
        )

        # Loss function
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.enhancement_model.train()
        total_loss = 0
        valid_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            try:
                # Ensure proper dimensions
                if len(noisy.shape) == 1:
                    noisy = noisy.unsqueeze(0)  # Add batch dimension
                if len(clean.shape) == 1:
                    clean = clean.unsqueeze(0)  # Add batch dimension

                # Enable gradients on input
                noisy.requires_grad_(True)

                # MetricGAN+ expects spectrograms with specific dimensions
                # Convert to spectrogram with the expected parameters
                stft = torch.stft(noisy, n_fft=512, hop_length=256, win_length=512,
                                  window=torch.hann_window(512).to(self.device),
                                  return_complex=True)

                # Get magnitude and phase
                magnitude = torch.abs(stft)  # Shape: [batch, freq_bins, time_frames]
                phase = torch.angle(stft)

                # Transpose to expected format [batch, time_frames, freq_bins]
                magnitude = magnitude.transpose(-1, -2)

                # Calculate lengths as ratio of actual frames to max frames
                lengths = torch.tensor([1.0] * magnitude.shape[0],
                                       dtype=torch.float32, device=self.device)

                try:
                    # Forward pass with proper spectrogram format
                    enhanced_magnitude = self.enhancement_model(magnitude, lengths)

                    # Transpose back to [batch, freq_bins, time_frames] for ISTFT
                    enhanced_magnitude = enhanced_magnitude.transpose(-1, -2)

                    # Reconstruct audio
                    enhanced_complex = enhanced_magnitude * torch.exp(1j * phase)
                    enhanced = torch.istft(enhanced_complex, n_fft=512, hop_length=256,
                                           win_length=512, window=torch.hann_window(512).to(self.device))

                    # Ensure same length as clean
                    if enhanced.shape[-1] != clean.shape[-1]:
                        min_len = min(enhanced.shape[-1], clean.shape[-1])
                        enhanced = enhanced[..., :min_len]
                        clean = clean[..., :min_len]

                except Exception as e1:
                    print(f"Spectrogram forward failed: {e1}")
                    print(f"Magnitude shape: {magnitude.shape}")
                    continue

                # Ensure enhanced and clean have the same shape
                if enhanced.shape != clean.shape:
                    if enhanced.numel() == clean.numel():
                        enhanced = enhanced.view(clean.shape)
                    else:
                        print(f"Shape mismatch: enhanced {enhanced.shape}, clean {clean.shape}")
                        continue

                # Calculate loss
                loss = self.criterion(enhanced, clean)

                # Check if loss requires grad
                if not loss.requires_grad:
                    print(f"Loss doesn't require grad in batch {batch_idx}")
                    continue

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                valid_batches += 1

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        return total_loss / max(valid_batches, 1)

    def validate(self, dataloader):
        """Validate the model"""
        self.enhancement_model.eval()
        total_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)

                try:
                    # Ensure proper dimensions
                    if len(noisy.shape) == 1:
                        noisy = noisy.unsqueeze(0)
                    if len(clean.shape) == 1:
                        clean = clean.unsqueeze(0)

                    # Convert to spectrogram with expected parameters
                    stft = torch.stft(noisy, n_fft=512, hop_length=256, win_length=512,
                                      window=torch.hann_window(512).to(self.device),
                                      return_complex=True)

                    # Get magnitude and phase
                    magnitude = torch.abs(stft)  # Shape: [batch, freq_bins, time_frames]
                    phase = torch.angle(stft)

                    # Transpose to expected format [batch, time_frames, freq_bins]
                    magnitude = magnitude.transpose(-1, -2)

                    # Calculate lengths
                    lengths = torch.tensor([1.0] * magnitude.shape[0],
                                           dtype=torch.float32, device=self.device)

                    # Forward pass with proper spectrogram format
                    enhanced_magnitude = self.enhancement_model(magnitude, lengths)

                    # Transpose back to [batch, freq_bins, time_frames] for ISTFT
                    enhanced_magnitude = enhanced_magnitude.transpose(-1, -2)

                    # Reconstruct audio
                    enhanced_complex = enhanced_magnitude * torch.exp(1j * phase)
                    enhanced = torch.istft(enhanced_complex, n_fft=512, hop_length=256,
                                           win_length=512, window=torch.hann_window(512).to(self.device))

                    # Ensure same length as clean
                    if enhanced.shape[-1] != clean.shape[-1]:
                        min_len = min(enhanced.shape[-1], clean.shape[-1])
                        enhanced = enhanced[..., :min_len]
                        clean = clean[..., :min_len]

                    # Ensure same shape
                    if enhanced.shape != clean.shape:
                        if enhanced.numel() == clean.numel():
                            enhanced = enhanced.view(clean.shape)
                        else:
                            continue

                    loss = self.criterion(enhanced, clean)
                    total_loss += loss.item()
                    valid_batches += 1

                except Exception as e:
                    print(f"Validation error: {e}")
                    continue

        return total_loss / max(valid_batches, 1)

    def fine_tune(self, train_dataloader, val_dataloader, num_epochs=10, patience=5):
        """Fine-tune the model"""

        print(f"Starting fine-tuning for {num_epochs} epochs...")
        patience_counter = 0

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss = self.train_epoch(train_dataloader)
            print(f"Training Loss: {train_loss:.4f}")

            # Validation
            val_loss = self.validate(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Store losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement
            min_improvement = 0.001
            if val_loss < (best_val_loss - min_improvement):
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch}.pth")
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement")
                break

        self.plot_losses()

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, filename):
        """Save the fine-tuned model"""
        torch.save({
            'model_state_dict': self.enhancement_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")


def fine_tune_metricgan(train_clean_folder, train_noisy_folder,
                        val_clean_folder, val_noisy_folder,
                        batch_size=8, num_epochs=10, device='cpu'):
    """
    Fine-tune MetricGAN+

    Args:
        train_clean_folder (str): Training clean audio folder
        train_noisy_folder (str): Training noisy audio folder
        val_clean_folder (str): Validation clean audio folder
        val_noisy_folder (str): Validation noisy audio folder
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        device (str): 'cpu' or 'cuda'
    """

    # Create datasets
    train_dataset = SpeechEnhancementDataset(train_clean_folder, train_noisy_folder)
    val_dataset = SpeechEnhancementDataset(val_clean_folder, val_noisy_folder)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize fine-tuner
    fine_tuner = MetricGANFineTuner(device=device)

    # Start fine-tuning
    fine_tuner.fine_tune(train_loader, val_loader, num_epochs=num_epochs, patience=5)

    return fine_tuner


if __name__ == "__main__":
    fine_tuner = fine_tune_metricgan(
        train_clean_folder="./Dataset_Improved/train/clean",
        train_noisy_folder="./Dataset_Improved/train/noisy",
        val_clean_folder="./Dataset_Improved/val/clean",
        val_noisy_folder="./Dataset_Improved/val/noisy",
        batch_size=4,
        num_epochs=20,
        device='cpu'
    )

    print("Fine-tuning complete!")
