import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.autograd import Function
import warnings
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import glob
import os

warnings.filterwarnings('ignore')


class SpeechDataset(Dataset):
    def __init__(self, audio_paths, labels, domains, sample_rate=16000, max_length=3.0):
        self.audio_paths = audio_paths
        self.labels = labels  # 0: typical, 1: disordered
        self.domains = domains  # 0: CHILDES, 1: Auris
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)

        # Audio preprocessing
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.audio_paths[idx])

        # Resample
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate to fixed length
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            padding = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return {
            'features': mel_spec_db.squeeze(0),  # Remove channel dimension
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'domain': torch.tensor(self.domains[idx], dtype=torch.long)
        }


# DANN Model Architecture
class DANNSpeechClassifier(nn.Module):
    def __init__(self, input_dim=(64, 188), hidden_dim=256, num_classes=2, dropout=0.3):
        super(DANNSpeechClassifier, self).__init__()

        # Feature extractor (CNN + RNN)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Calculate flattened size
        conv_output_size = 128 * 8 * 8

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Label classifier (typical vs disordered)
        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Domain classifier (CHILDES vs Auris)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 2 domains
        )

    def forward(self, x, alpha=1.0):
        # Extract features
        conv_features = self.conv_layers(x.unsqueeze(1))  # Add channel dimension
        features = self.feature_extractor(conv_features)

        # Label prediction
        label_pred = self.label_classifier(features)

        # Domain prediction (with gradient reversal)
        domain_pred = self.domain_classifier(features)

        return label_pred, domain_pred, features


# Training function
def train_dann(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    model = model.to(device)

    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Loss functions
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_label_loss': [], 'train_domain_loss': [], 'train_total_loss': [],
        'val_label_loss': [], 'val_domain_loss': [], 'val_total_loss': [],
        'val_label_acc': [], 'val_domain_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_label_loss = 0
        train_domain_loss = 0
        train_total_loss = 0

        # Calculate alpha for gradient reversal (increases during training)
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            domains = batch['domain'].to(device)

            optimizer.zero_grad()

            # Forward pass
            label_pred, domain_pred, _ = model(features, alpha)

            # Compute losses
            label_loss = label_criterion(label_pred, labels)
            domain_loss = domain_criterion(domain_pred, domains)
            total_loss = label_loss + domain_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            train_label_loss += label_loss.item()
            train_domain_loss += domain_loss.item()
            train_total_loss += total_loss.item()

        # Validation
        model.eval()
        val_label_loss = 0
        val_domain_loss = 0
        val_total_loss = 0
        val_label_preds = []
        val_domain_preds = []
        val_labels = []
        val_domains = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                domains = batch['domain'].to(device)

                label_pred, domain_pred, _ = model(features, alpha)

                label_loss = label_criterion(label_pred, labels)
                domain_loss = domain_criterion(domain_pred, domains)
                total_loss = label_loss + domain_loss

                val_label_loss += label_loss.item()
                val_domain_loss += domain_loss.item()
                val_total_loss += total_loss.item()

                val_label_preds.extend(torch.argmax(label_pred, dim=1).cpu().numpy())
                val_domain_preds.extend(torch.argmax(domain_pred, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_domains.extend(domains.cpu().numpy())

        # Calculate accuracies
        val_label_acc = accuracy_score(val_labels, val_label_preds)
        val_domain_acc = accuracy_score(val_domains, val_domain_preds)

        # Store history
        history['train_label_loss'].append(train_label_loss / len(train_loader))
        history['train_domain_loss'].append(train_domain_loss / len(train_loader))
        history['train_total_loss'].append(train_total_loss / len(train_loader))
        history['val_label_loss'].append(val_label_loss / len(val_loader))
        history['val_domain_loss'].append(val_domain_loss / len(val_loader))
        history['val_total_loss'].append(val_total_loss / len(val_loader))
        history['val_label_acc'].append(val_label_acc)
        history['val_domain_acc'].append(val_domain_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]:')
            print(f'  Train Loss: {train_total_loss / len(train_loader):.4f} '
                  f'(Label: {train_label_loss / len(train_loader):.4f}, '
                  f'Domain: {train_domain_loss / len(train_loader):.4f})')
            print(f'  Val Acc: Label={val_label_acc:.4f}, Domain={val_domain_acc:.4f}')
            print(f'  Alpha (GRL): {alpha:.4f}')
            print()

        scheduler.step()

    return model, history


# Visualization function
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plots
    axes[0, 0].plot(history['train_label_loss'], label='Train Label Loss')
    axes[0, 0].plot(history['val_label_loss'], label='Val Label Loss')
    axes[0, 0].set_title('Label Classification Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_domain_loss'], label='Train Domain Loss')
    axes[0, 1].plot(history['val_domain_loss'], label='Val Domain Loss')
    axes[0, 1].set_title('Domain Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy plots
    axes[1, 0].plot(history['val_label_acc'], label='Label Accuracy')
    axes[1, 0].set_title('Label Classification Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['val_domain_acc'], label='Domain Accuracy')
    axes[1, 1].set_title('Domain Classification Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# Feature visualization function
def visualize_feature_space(model, data_loader, device='cuda'):
    model.eval()
    features_list = []
    labels_list = []
    domains_list = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['features'].to(device)
            labels = batch['label'].cpu().numpy()
            domains = batch['domain'].cpu().numpy()

            _, _, features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels)
            domains_list.extend(domains)

    features_array = np.vstack(features_list)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_array)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot by label (typical vs disordered)
    for label in [0, 1]:
        mask = np.array(labels_list) == label
        label_name = 'Typical' if label == 0 else 'Disordered'
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    label=label_name, alpha=0.6)
    ax1.set_title('Feature Space by Speech Type')
    ax1.legend()
    ax1.grid(True)

    # Plot by domain (CHILDES vs Auris)
    for domain in [0, 1]:
        mask = np.array(domains_list) == domain
        domain_name = 'CHILDES' if domain == 0 else 'Clinical'
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    label=domain_name, alpha=0.6)
    ax2.set_title('Feature Space by Domain')
    ax2.legend()
    ax2.grid(True)

    plt.show()


def prepare_audio_data(base_path="Screener"):
    """
    Load audio data from CHILDES and Auris datasets.

    Args:
        base_path (str): Base path to the Screener directory

    Returns:
        dict: Dictionary containing lists of audio file paths organized by dataset and condition
    """
    print("Loading audio data from CHILDES and clinical datasets...")

    age_groups = ['3yo', '4yo', '5yo', '6yo']

    audio_data = {
        'childes_typical': [],
        'childes_disordered': [],
        'clinical_typical': [],
        'clinical_disordered': []
    }

    # Load CHILDES data
    print("Loading CHILDES data...")
    childes_base = os.path.join(base_path, "Processed_CHILDES")

    for age_group in age_groups:
        # CHILDES TD data
        childes_td_pattern = os.path.join(
            childes_base,
            f"Childes_{age_group}",
            "TD",
            "*",
            "*_chi_chins.wav"
        )
        td_files = glob.glob(childes_td_pattern)
        audio_data['childes_typical'].extend(td_files)

        # CHILDES TOS data
        childes_tos_pattern = os.path.join(
            childes_base,
            f"CHILDES_{age_group}",
            "TOS",
            "*",
            "*_chi_chins.wav"
        )
        tos_files = glob.glob(childes_tos_pattern)
        audio_data['childes_disordered'].extend(tos_files)

    print("Loading Auris data...")

    for age_group in age_groups:
        # Auris TD data
        clinical_td_pattern = os.path.join(
            base_path,
            age_group,
            f"TD-{age_group}",
            "Formatted",
            "*",
            "*_chi_chins.wav"
        )
        td_files = glob.glob(clinical_td_pattern)
        audio_data['clinical_typical'].extend(td_files)

        # Auris TOS data
        clinical_tos_pattern = os.path.join(
            base_path,
            age_group,
            f"TOS-{age_group}",
            "Formatted",
            "*",
            "*_chi_chins.wav"
        )
        tos_files = glob.glob(clinical_tos_pattern)
        audio_data['clinical_disordered'].extend(tos_files)

        # Auris vvTOS data
        clinical_vvtos_pattern = os.path.join(
            base_path,
            age_group,
            f"vvTOS-{age_group}",
            "Formatted",
            "*",
            "*_chi_chins.wav"
        )
        vvtos_files = glob.glob(clinical_vvtos_pattern)
        audio_data['clinical_disordered'].extend(vvtos_files)

    print(f"\nData loading summary:")
    print(f"CHILDES typical files: {len(audio_data['childes_typical'])}")
    print(f"CHILDES disordered files: {len(audio_data['childes_disordered'])}")
    print(f"Clinical typical files: {len(audio_data['clinical_typical'])}")
    print(f"Clinical disordered files: {len(audio_data['clinical_disordered'])}")
    print(f"Total files: {sum(len(files) for files in audio_data.values())}")

    # Verify files exist
    total_missing = 0
    for category, files in audio_data.items():
        missing_files = [f for f in files if not os.path.exists(f)]
        if missing_files:
            print(f"\nWarning: {len(missing_files)} missing files in {category}")
            total_missing += len(missing_files)

    if total_missing == 0:
        print("\nWe're ready to go Dann")

    return audio_data


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading audio data...")
    audio_data = prepare_audio_data(base_path="C:/Users/a.stasica.AURIS.000/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener")

    # God help us it finds the data
    total_files = sum(len(files) for files in audio_data.values())
    if total_files == 0:
        print("No audio files found! Please check your data paths.")
        return

    print(f"\nDataset loading summary:")
    print(f"CHILDES typical files: {len(audio_data['childes_typical'])}")
    print(f"CHILDES disordered files: {len(audio_data['childes_disordered'])}")
    print(f"Clinical typical files: {len(audio_data['clinical_typical'])}")
    print(f"Clinical disordered files: {len(audio_data['clinical_disordered'])}")
    print(f"Total files: {total_files}")

    # Collect all file paths and create labels
    audio_paths = []
    labels = []  # 0: TD, 1: TOS/vvTOS
    domains = []  # 0: CHILDES, 1: Auris

    # Add CHILDES data
    for file_path in audio_data['childes_typical']:
        audio_paths.append(file_path)
        labels.append(0)  # TD
        domains.append(0)  # CHILDES

    for file_path in audio_data['childes_disordered']:
        audio_paths.append(file_path)
        labels.append(1)  # TOS
        domains.append(0)  # CHILDES

    # Add Clinical data
    for file_path in audio_data['clinical_typical']:
        audio_paths.append(file_path)
        labels.append(0)  # TD
        domains.append(1)  # Auris

    for file_path in audio_data['clinical_disordered']:
        audio_paths.append(file_path)
        labels.append(1)  # TOS/vvTOS
        domains.append(1)  # Auris

    print(f"\nFinal dataset composition:")
    print(f"Total files: {len(audio_paths)}")
    print(f"Typical files: {labels.count(0)}")
    print(f"Disordered files: {labels.count(1)}")
    print(f"CHILDES files: {domains.count(0)}")
    print(f"Clinical files: {domains.count(1)}")

    # Check if we have enough data for training
    if len(audio_paths) < 10:
        print("Warning: Very few files found. Training may not be effective.")

    # Check class balance - we don't have it now but it s just a test
    if labels.count(0) == 0 or labels.count(1) == 0:
        print("Error: Need both typical and disordered samples for training.")
        return

    # Split data into train and validation sets
    train_paths, val_paths, train_labels, val_labels, train_domains, val_domains = train_test_split(
        audio_paths, labels, domains, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\nTrain/Validation split:")
    print(f"Training files: {len(train_paths)}")
    print(f"Validation files: {len(val_paths)}")

    # Create datasets and data loaders
    train_dataset = SpeechDataset(train_paths, train_labels, train_domains)
    val_dataset = SpeechDataset(val_paths, val_labels, val_domains)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model and show DANN architecture
    model = DANNSpeechClassifier()
    print("\nNice architecture Dann")
    print(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nStarting training...")
    trained_model, history = train_dann(model, train_loader, val_loader,
                                        num_epochs=50, lr=0.001, device=device)

    # Visualize results
    print("\nGenerating visualizations...")
    plot_training_history(history)
    visualize_feature_space(trained_model, val_loader, device)

    # Save model
    model_path = 'dann_speech_classifier.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
