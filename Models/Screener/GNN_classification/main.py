import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import parselmouth
import pandas as pd
import numpy as np
import torch
import librosa
import os
from torch_geometric.data import Data, DataLoader, Batch
import warnings
from preprocessing import load_dld_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')




class AudioFeatureExtractor:
    """
    extract only audio features without transcription (should implement it later when we will have it)
    """

    def __init__(self, sr=16000):
        self.sr = sr

    def extract_prosodic_features(self, audio):
        """Prosodic features"""
        sound = parselmouth.Sound(audio)  # sampling_frequency=self.sr

        # F0
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values != 0]  # remove non voiced values

        features = {}
        if len(f0_values) > 0:
            features['f0_mean'] = np.mean(f0_values)
            features['f0_std'] = np.std(f0_values)
            features['f0_range'] = np.max(f0_values) - np.min(f0_values)
            features['f0_median'] = np.median(f0_values)
        else:
            features.update({'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'f0_median': 0})

        # Intensity (energy)
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        features['intensity_mean'] = np.mean(intensity_values)
        features['intensity_std'] = np.std(intensity_values)

        return features

    def extract_temporal_features(self, audio):
        """Temporal features"""
        # detect speech vs silence
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)  # 10ms hop

        # Energy per frame
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Threshold to detect speech
        speech_threshold = np.percentile(energy, 30)
        speech_frames = energy > speech_threshold

        # Calculate speech vs silence segments
        speech_segments = []
        pause_segments = []

        current_segment = []
        is_speech = speech_frames[0]

        for i, frame_is_speech in enumerate(speech_frames):
            if frame_is_speech == is_speech:
                current_segment.append(i)
            else:
                if is_speech:
                    speech_segments.append(len(current_segment))
                else:
                    pause_segments.append(len(current_segment))
                current_segment = [i]
                is_speech = frame_is_speech

        # add last segment
        if is_speech:
            speech_segments.append(len(current_segment))
        else:
            pause_segments.append(len(current_segment))

        features = {}
        # mean durations (in seconds)
        hop_duration = hop_length / self.sr
        features['mean_speech_duration'] = np.mean(speech_segments) * hop_duration if speech_segments else 0
        features['mean_pause_duration'] = np.mean(pause_segments) * hop_duration if pause_segments else 0
        features['speech_rate'] = len(speech_segments) / (len(audio) / self.sr)  # segments per second
        features['pause_rate'] = len(pause_segments) / (len(audio) / self.sr)

        # Ratio speech/pause
        total_speech_time = sum(speech_segments) * hop_duration
        total_pause_time = sum(pause_segments) * hop_duration
        features['speech_pause_ratio'] = total_speech_time / (total_pause_time + 1e-6)

        return features

    def extract_spectral_features(self, audio):
        """Spectral features"""
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]

        features = {}

        # MFCC features
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])

        # Other spectral features
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['zcr_mean'] = np.mean(zero_crossing_rate)
        features['zcr_std'] = np.std(zero_crossing_rate)

        return features

    def extract_voice_quality_features(self, audio):
        """Vocal quality features"""
        sound = parselmouth.Sound(audio, sampling_frequency=self.sr)

        features = {}

        # Jitter and Shimmer (measures for vocal stability)
        try:
            pitch = sound.to_pitch()
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, peaks)", 75, 500, True, True)

            # Jitter
            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

            # Shimmer
            shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02,
                                                   1.3, 1.6)
            shimmer_apq3 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3,
                                                  1.6)

            features['jitter_local'] = jitter_local if not np.isnan(jitter_local) else 0
            features['jitter_rap'] = jitter_rap if not np.isnan(jitter_rap) else 0
            features['shimmer_local'] = shimmer_local if not np.isnan(shimmer_local) else 0
            features['shimmer_apq3'] = shimmer_apq3 if not np.isnan(shimmer_apq3) else 0

        except:
            features.update({'jitter_local': 0, 'jitter_rap': 0, 'shimmer_local': 0, 'shimmer_apq3': 0})

        # Harmonics-to-Noise Ratio
        try:
            harmonicity = sound.to_harmonicity()
            hnr_values = harmonicity.values[0]
            hnr_values = hnr_values[~np.isnan(hnr_values)]
            features['hnr_mean'] = np.mean(hnr_values) if len(hnr_values) > 0 else 0
            features['hnr_std'] = np.std(hnr_values) if len(hnr_values) > 0 else 0
        except:
            features.update({'hnr_mean': 0, 'hnr_std': 0})

        return features

    def extract_all_features(self, audio):
        """Combine all audio characteristics"""
        prosodic = self.extract_prosodic_features(audio)
        temporal = self.extract_temporal_features(audio)
        spectral = self.extract_spectral_features(audio)
        voice_quality = self.extract_voice_quality_features(audio)

        # Combine all dictionaries
        all_features = {**prosodic, **temporal, **spectral, **voice_quality}

        # Convert in array numpy
        feature_vector = np.array(list(all_features.values()))
        feature_names = list(all_features.keys())

        return feature_vector, feature_names


class DLDGraphStructure:
    """
    Define graph structure based on the relations between audio features
    """

    def __init__(self):
        # Theoretical relations between audio features for DLD
        self.audio_relationships = {
            # Prosodical relations
            'f0_mean': ['f0_std', 'f0_range', 'intensity_mean'],
            'f0_std': ['f0_range', 'jitter_local'],
            'f0_range': ['intensity_std'],

            # Temporal relations
            'speech_rate': ['mean_speech_duration', 'mean_pause_duration', 'pause_rate'],
            'mean_speech_duration': ['speech_pause_ratio'],
            'mean_pause_duration': ['speech_pause_ratio', 'pause_rate'],

            # Spectral relations
            'spectral_centroid_mean': ['spectral_bandwidth_mean', 'mfcc_0_mean'],
            'spectral_bandwidth_mean': ['spectral_rolloff_mean'],
            'zcr_mean': ['zcr_std', 'spectral_centroid_mean'],

            # Vocal quality relations
            'jitter_local': ['jitter_rap', 'shimmer_local'],
            'shimmer_local': ['shimmer_apq3', 'hnr_mean'],
            'hnr_mean': ['hnr_std'],

            # Cross-domain relations
            'speech_rate': ['spectral_centroid_mean', 'f0_std'],  # flow and stability
            'mean_pause_duration': ['jitter_local', 'shimmer_local'],  # Pauses and vocal quality
        }

    def create_edge_index(self, feature_names):
        """ Create graph adjacency matrix"""
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        edges = []

        for source_feature, target_features in self.audio_relationships.items():
            if source_feature in name_to_idx:
                source_idx = name_to_idx[source_feature]
                for target_feature in target_features:
                    if target_feature in name_to_idx:
                        target_idx = name_to_idx[target_feature]
                        # Add the two directions (non-directed graph)
                        edges.append([source_idx, target_idx])
                        edges.append([target_idx, source_idx])

        # Add self-loops
        for i in range(len(feature_names)):
            edges.append([i, i])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class AudioOnlyGNN(nn.Module):
    """
    GNN for DLD classification based only on audio features
    """

    def __init__(self, num_audio_features, hidden_dim=64, num_classes=2):
        super().__init__()

        self.num_features = num_audio_features

        # Normalisation of input features
        self.feature_norm = nn.BatchNorm1d(num_audio_features)

        # layers GNN
        self.conv1 = GCNConv(num_audio_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)

        # Dropout for regularisation
        self.dropout = nn.Dropout(0.3)

        # classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # features normalisation
        x = self.feature_norm(x)

        # Propagation inside the graph
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = torch.relu(self.conv3(x, edge_index))

        # Aggregation at the graph level (one graph per audio sample)
        x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)


class DLDAudioClassifier:
    """
    Principal classifier combining feature extraction and GNN
    """

    def __init__(self, sr=16000):
        self.feature_extractor = AudioFeatureExtractor(sr=sr)
        self.graph_structure = DLDGraphStructure()
        self.model = None
        self.feature_names = None
        self.edge_index = None

    def prepare_data(self, audio_samples, labels):
        """
        Prepare audio data for training/test
        """

        graphs = []

        for audio, label in zip(audio_samples, labels):
            # Add validation check
            if not isinstance(audio, np.ndarray):
                print(f"Warning: Audio sample is type {type(audio)}, converting to numpy array")
                try:
                    audio = np.array(audio)
                except Exception as e:
                    print(f"Failed to convert audio to numpy array: {e}")
                    continue

        for audio, label in zip(audio_samples, labels):
            # feature extraction
            features, feature_names = self.feature_extractor.extract_all_features(audio)

            if self.feature_names is None:
                self.feature_names = feature_names
                self.edge_index = self.graph_structure.create_edge_index(feature_names)

            # graph creation
            x = torch.tensor(features, dtype=torch.float).unsqueeze(0)  # [1, num_features]

            # replicate the features for each node (each feature becomes a node)
            x = x.repeat(len(feature_names), 1).t()  # [num_features, num_features]
            x = torch.diag(x.squeeze())  # Diagonal matrix -> each node has its own feature
            x = x.unsqueeze(1)  # [num_features, 1]

            # Use the feature values as node features
            x = torch.tensor(features, dtype=torch.float).unsqueeze(1)  # [num_features, 1]

            graph = Data(x=x, edge_index=self.edge_index, y=torch.tensor([label], dtype=torch.long))
            graphs.append(graph)

        return graphs

    def train_model(self, train_graphs, val_graphs=None, epochs=100, lr=0.001):
        """
        training with validation tracking
        """
        if not train_graphs:
            raise ValueError("No training sample")

        num_features = train_graphs[0].x.shape[1]
        self.model = AudioOnlyGNN(num_features)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32) if val_graphs else None

        # Track metrics
        train_losses = []
        val_accuracies = []
        best_val_acc = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            if val_loader:
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        out = self.model(batch.x, batch.edge_index, batch.batch)
                        pred = out.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        total += batch.y.size(0)

                val_acc = correct / total
                val_accuracies.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:3d}/{epochs}: Train Loss: {avg_loss:.4f}', end=' ')
                if val_loader:
                    print(f'- Val Acc: {val_acc:.4f}', end='')
                    if val_acc == best_val_acc:
                        print(' (best)', end='')
                print()

        # Plot training curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        if val_loader:
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')

        plt.tight_layout()
        plt.show()

        # Load best model
        if val_loader and os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))

    def evaluate(self, test_graphs, verbose=True):
        """
        evaluation with multiple metrics
        Returns: dict with metrics and predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained :(")

        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_true = []

        with torch.no_grad():
            for batch in test_loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
                all_true.extend(batch.y.cpu().numpy())

        accuracy = correct / total
        report = classification_report(all_true, all_preds, target_names=['DLD', 'TD'])
        cm = confusion_matrix(all_true, all_preds)
        roc_auc = roc_auc_score(all_true, [p[1] for p in all_probs])

        if verbose:
            print("\n=== Evaluation Metrics ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(report)
            print("\nConfusion Matrix:")
            print(cm)

            # Plot confusion matrix
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['DLD', 'TD'],
                        yticklabels=['DLD', 'TD'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': all_true
        }

    def analyze_features(self, graphs, labels):
        """
        Analyze feature differences between TD and DLD groups
        Returns: DataFrame with feature statistics by group
        """
        # Extract features from all graphs
        features = []
        for graph in graphs:
            features.append(graph.x.squeeze().numpy())

        features = np.array(features)
        feature_names = self.feature_names

        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df['diagnosis'] = ['DLD' if label == 0 else 'TD' for label in labels]

        # Calculate statistics
        stats = df.groupby('diagnosis').agg(['mean', 'std'])

        # Calculate effect sizes (Cohen's d) and add as a simple row
        effect_sizes = []
        for feature in feature_names:
            td_mean = stats.loc['TD', (feature, 'mean')]
            dld_mean = stats.loc['DLD', (feature, 'mean')]
            td_std = stats.loc['TD', (feature, 'std')]
            dld_std = stats.loc['DLD', (feature, 'std')]

            # Get group sizes
            td_n = len(df[df['diagnosis'] == 'TD'])
            dld_n = len(df[df['diagnosis'] == 'DLD'])

            # pooled standard deviation formula
            pooled_std = np.sqrt(((td_n - 1) * td_std ** 2 + (dld_n - 1) * dld_std ** 2) / (td_n + dld_n - 2))

            # handle division by zero
            if pooled_std == 0:
                effect_sizes.append(0.0)
            else:
                effect_sizes.append((td_mean - dld_mean) / pooled_std)

        # Add effect sizes as a new row with single-level column index
        effect_size_df = pd.DataFrame([effect_sizes], columns=feature_names, index=['effect_size'])

        return stats, effect_size_df

    def plot_feature_distributions(self, graphs, labels, top_n=10):
        """
        plot distributions of most discriminative features
        """

        stats, effect_size_df = self.analyze_features(graphs, labels)
        effect_sizes = effect_size_df.loc['effect_size']

        # get top features by absolute effect size
        top_features = effect_sizes.abs().sort_values(ascending=False).head(top_n).index

        # extract features and create DataFrame
        features = []
        for graph in graphs:
            features.append(graph.x.squeeze().numpy())
        features = np.array(features)

        # create DataFrame for plotting
        df = pd.DataFrame(features, columns=self.feature_names)
        df['diagnosis'] = ['DLD' if label == 0 else 'TD' for label in labels]

        # calculate subplot grid
        n_cols = min(5, len(top_features))
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        # Plot
        plt.figure(figsize=(15, 4 * n_rows))
        for i, feature in enumerate(top_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.boxplot(x='diagnosis', y=feature, data=df)
            plt.title(f"{feature}\n(d={effect_sizes[feature]:.3f})")
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def predict(self, audio_sample):
        """
        Predict class for an audio sample
        """
        if self.model is None:
            raise ValueError("Model still couldn't be trained? Wow loser")

        # feature extractions
        features, _ = self.feature_extractor.extract_all_features(audio_sample)

        # graph creation
        x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
        graph = Data(x=x, edge_index=self.edge_index)

        self.model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([graph])
            out = self.model(batch.x, batch.edge_index, batch.batch)
            pred_proba = torch.softmax(out, dim=1)
            pred_class = out.argmax(dim=1).item()

        return pred_class, pred_proba.numpy()[0]


if __name__ == "__main__":
    # initialize classifier
    classifier = DLDAudioClassifier()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_audio, train_labels, val_audio, val_labels, test_audio, test_labels = load_dld_data(
        csv_path="Regression_model.csv",
        test_size=0.10,
        val_size=0.15,
        target_sr=16000,
        random_state=42
    )

    if train_audio is None:
        print("Failed to load data. Exiting.")
        exit()

    # Prepare data
    print("\nPreparing graph data...")
    train_graphs = classifier.prepare_data(train_audio, train_labels)
    val_graphs = classifier.prepare_data(val_audio, val_labels)
    test_graphs = classifier.prepare_data(test_audio, test_labels)

    print(f"\nDataset Summary:")
    print(f"- Training graphs: {len(train_graphs)}")
    print(f"- Validation graphs: {len(val_graphs)}")
    print(f"- Testing graphs: {len(test_graphs)}")
    print(f"- Features extracted: {len(classifier.feature_names)}")

    # Analyze feature differences
    print("\nAnalyzing feature differences between TD and DLD...")
    stats, effect_size_df = classifier.analyze_features(train_graphs, train_labels)
    print(stats.head(10))
    print("Group Statistics:")
    print(stats.head(10))
    print("\nEffect Sizes:")
    print(effect_size_df)

    # Plot feature distributions
    classifier.plot_feature_distributions(train_graphs, train_labels)

    # Train model
    print("\nTraining model...")
    classifier.train_model(train_graphs, val_graphs)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = classifier.evaluate(test_graphs)

    # save model and results
    torch.save(classifier.model.state_dict(), 'dld_classifier.pth')
    stats.to_csv('feature_statistics.csv')


    # Feature importance analysis (using model gradients)
    # print("\nAnalyzing feature importance...")
    # This requires additional implementation to compute gradient-based importance -> to be implemented

