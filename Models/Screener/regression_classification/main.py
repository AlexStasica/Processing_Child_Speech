import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torchaudio.transforms import MFCC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import json
from datetime import datetime

# Create output directory
output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperModel.from_pretrained("openai/whisper-small")
model.eval()

# Base directory
base_dir = r"C:\Users\a.stasica.AURIS.000\OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO\Desktop\Python\Screener\Processed_CHILDES"


class FeatureExtractor:
    def __init__(self, feature_type="whisper"):
        self.feature_type = feature_type
        if feature_type == "whisper":
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large")
            self.model = WhisperModel.from_pretrained("openai/whisper-large")
            self.model.eval()
        elif feature_type == "mfcc":
            # MFCC parameters
            self.n_mfcc = 40
            self.sample_rate = 16000
            self.n_fft = 400
            self.hop_length = 160
            self.mfcc_transform = MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': 128
                }
            )

    def extract(self, audio_tensor):
        if self.feature_type == "whisper":
            return self._extract_whisper_features(audio_tensor)
        elif self.feature_type == "mfcc":
            return self._extract_mfcc_features(audio_tensor)

    def _extract_whisper_features(self, audio_tensor):
        inputs = self.processor(audio_tensor, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            encoder_out = self.model.encoder(inputs.input_features).last_hidden_state
        return encoder_out.mean(dim=1).squeeze().numpy()

    def _extract_mfcc_features(self, audio_tensor):
        # Ensure audio is 2D (batch, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Compute MFCCs
        mfcc = self.mfcc_transform(audio_tensor)

        mean = mfcc.mean(dim=-1).squeeze().numpy()
        std = mfcc.std(dim=-1).squeeze().numpy()
        return np.concatenate([mean, std])


def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return torch.tensor(audio)


def clean_pp_name(pp_name):
    """Clean the PP name by removing .wav if present and any whitespace"""
    if pd.isna(pp_name):
        return None
    pp_name = str(pp_name).strip()
    if pp_name.lower().endswith('.wav'):
        pp_name = pp_name[:-4]  # Remove .wav
    return pp_name


def evaluate_and_plot(test_ids, X_test, y_test, diagnoses_test, regressor, model_name, feature_type):
    # Convert months back to years for better interpretation
    y_test_years = y_test / 12
    y_pred = regressor.predict(X_test)
    y_pred_years = y_pred / 12

    # Create results DataFrame
    results = pd.DataFrame({
        'Actual Age (months)': y_test,
        'Predicted Age (months)': y_pred,
        'Actual Age (years)': y_test_years,
        'Predicted Age (years)': y_pred_years,
        'Diagnosis': diagnoses_test,
    })

    results_df = pd.DataFrame({
        'PP': test_ids,
        'Actual_Age': y_test,
        'Predicted_Age': y_pred,
        'Diagnosis': diagnoses_test,
        'Model': model_name,
        'Feature_Type': feature_type
    })

    # Calculate metrics
    metrics = {
        'model': model_name,
        'feature_type': feature_type,
        'overall': {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'r2': regressor.score(X_test, y_test)
        },
        'TD': {},
        'DLD': {}
    }

    for diag in ['TD', 'DLD']:
        diag_mask = results['Diagnosis'] == diag
        if sum(diag_mask) > 0:
            y_true = results[diag_mask]['Actual Age (months)']
            y_pred = results[diag_mask]['Predicted Age (months)']

            metrics[diag] = {
                'count': sum(diag_mask),
                'mae': np.mean(np.abs(y_true - y_pred)),
                'mse': mean_squared_error(y_true, y_pred),
                'r2': regressor.score(X_test[diag_mask], y_test[diag_mask]),
                'pearson_r': stats.pearsonr(y_true, y_pred)[0],
                'pearson_p': stats.pearsonr(y_true, y_pred)[1],
                'age_diff': np.mean(y_true - y_pred) if diag == 'DLD' else None
            }

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{model_name} ({feature_type} features)", y=1.02)

    # 1. Actual vs Predicted Age Scatter Plot
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=results, x='Actual Age (years)', y='Predicted Age (years)', hue='Diagnosis')
    plt.plot([3, 8], [3, 8], 'k--')  # Perfect prediction line
    plt.title('Actual vs Predicted Age')

    # 2. Error Distribution
    plt.subplot(2, 2, 2)
    results['Error (months)'] = results['Actual Age (months)'] - results['Predicted Age (months)']
    sns.boxplot(data=results, x='Diagnosis', y='Error (months)')
    plt.title('Prediction Error Distribution')

    # 3. Age Difference (DLD only)
    if 'DLD' in results['Diagnosis'].unique():
        plt.subplot(2, 2, 3)
        dld_results = results[results['Diagnosis'] == 'DLD']
        sns.histplot(dld_results['Error (months)'] / 12, bins=15, kde=True)
        plt.title('DLD Children: Actual - Predicted Age (years)')

    # 4. Age Distribution Comparison
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=results, x='Predicted Age (years)', hue='Diagnosis', fill=True)
    plt.title('Predicted Age Distribution')

    plt.tight_layout()

    # Save plot
    plot_filename = f"{output_dir}/{feature_type}_{model_name.replace(' ', '_')}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    return metrics, results_df


def train_and_evaluate_models(test_ids, X_train, y_train, X_test, y_test, X_dld, y_dld, diagnoses_test, feature_type):
    models = {
        "Ridge Regression": Ridge(),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), early_stopping=True),
        "Bayesian Ridge": BayesianRidge()
    }

    all_metrics = []
    all_predictions = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        metrics, results = evaluate_and_plot(
            test_ids,
            np.vstack([X_test, X_dld]),
            np.concatenate([y_test, y_dld]),
            np.concatenate([['TD'] * len(y_test), ['DLD'] * len(y_dld)]),
            model, name, feature_type
        )
        all_metrics.append(metrics)

        all_predictions.append(results)

        # Print summary
        print(f"\n{name} Results ({feature_type} features):")
        print(f"Overall MSE: {metrics['overall']['mse']:.2f}")
        print(f"DLD Underestimation: {metrics['DLD']['age_diff']:.2f} months")



    # Save all metrics
    with open(f"{output_dir}/{feature_type}_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics, all_predictions


def get_age_prefix(leeftijd):
    """Extract the first digit from age string (e.g., '3;11' -> '3')"""
    if pd.isna(leeftijd):
        return None
    return str(leeftijd).split(';')[0][0]


def convert_age_to_months(age_str):
    """Convert age string 'year;months' to total months"""
    if pd.isna(age_str):
        return None
    years, months = map(int, str(age_str).split(';'))
    return years * 12 + months


def convert_months_to_age(months):
    """Convert total months back to 'year;months' format"""
    if isinstance(months, (np.ndarray, list)):
        return [convert_months_to_age(m) for m in months]

    if pd.isna(months):
        return "NA"

    months_int = int(round(float(months)))
    years = months_int // 12
    remaining_months = months_int % 12
    return f"{years};{remaining_months:02d}"


def extract_features(audio_tensor):
    inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        encoder_out = model.encoder(inputs.input_features).last_hidden_state
    features = encoder_out.mean(dim=1).squeeze().numpy()
    return features


def find_audio_file(pp_name, full_path):
    """Find audio file using the provided full path"""
    if pd.isna(pp_name) or pd.isna(full_path):
        return None

    # Two filename patterns
    filename_patterns = [
        f"{pp_name}_chi_chins.wav",  # Original pattern
        f"{pp_name}_audio_chi_chins.wav"  # Alternative pattern
    ]

    for pattern in filename_patterns:
        audio_path = os.path.join(full_path, pattern)
        if os.path.exists(audio_path):
            return audio_path

    # Debug output if file not found
    print(f"Could not find audio file for {pp_name} in {full_path}")
    print("Tried patterns:")
    for pattern in filename_patterns:
        print(f"- {os.path.join(full_path, pattern)}")

    return None


def load_dataset(csv_path):
    """Load dataset from CSV using provided full paths"""
    df = pd.read_csv(csv_path)

    # Clean data and verify required columns
    required_columns = {'PP', 'Leeftijd', 'Diagnose', 'Full Path'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df['PP'] = df['PP'].apply(clean_pp_name)
    df = df.dropna(subset=['PP', 'Leeftijd', 'Diagnose', 'Full Path'])

    audio_files = []
    targets = []
    pp_ids = []
    diagnoses = []
    missing_files = set()

    for idx, row in df.iterrows():
        pp_name = row['PP']
        audio_path = find_audio_file(pp_name, row['Full Path'])

        if audio_path:
            audio_files.append(audio_path)
            targets.append(convert_age_to_months(row['Leeftijd']))
            diagnoses.append(row['Diagnose'])
            pp_ids.append(pp_name)
        else:
            missing_files.add(pp_name)

    # Reporting
    print(f"\nDataset summary:")
    print(f"- Total valid entries in CSV: {len(df)}")
    print(f"- Successfully found audio files: {len(audio_files)}")
    print(f"- Missing audio files: {len(missing_files)}")
    if missing_files:
        print("Sample missing PP names:", list(missing_files))

    return audio_files, np.array(targets), np.array(diagnoses), np.array(pp_ids)


def main(feature_type):
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(feature_type=feature_type)

    # Load dataset
    csv_path = "Regression_model.csv"
    audio_files, targets_months, diagnoses, pp_ids = load_dataset(csv_path)

    if len(audio_files) == 0:
        print("Error: No audio files found.")
        return

    print(f"Successfully loaded {len(audio_files)} audio files")

    # Extract features
    X = []
    successful_indices = []  # Track which files were processed successfully
    print(f"Extracting {feature_type} features...")

    for i, file in enumerate(audio_files):
        try:
            audio = load_audio(file)
            feats = feature_extractor.extract(audio)
            X.append(feats)
            successful_indices.append(i)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(audio_files)} files")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # Filter targets and diagnoses to only include successfully processed files
    targets_months = targets_months[successful_indices]
    diagnoses = diagnoses[successful_indices]
    pp_ids = pp_ids[successful_indices]
    X = np.array(X)

    print(f"\nFinal dataset sizes:")
    print(f"- Features (X): {len(X)}")
    print(f"- Targets: {len(targets_months)}")
    print(f"- Diagnoses: {len(diagnoses)}")

    # Split into TD (train+test) and DLD (test only)
    td_mask = (diagnoses == 'TD')
    dld_mask = (diagnoses == 'DLD')

    X_td = X[td_mask]
    y_td = targets_months[td_mask]

    X_dld = X[dld_mask]
    y_dld = targets_months[dld_mask]

    # Split TD data into train and test
    X_train, X_test_td, y_train, y_test_td = train_test_split(
        X_td, y_td, test_size=0.2, random_state=42
    )

    pp_ids_td = pp_ids[td_mask]
    pp_ids_dld = pp_ids[dld_mask]

    train_ids, test_ids = train_test_split(
        pp_ids_td,
        test_size=0.2,
        random_state=42
    )

    print(f"\n=== Evaluating Multiple Models with {feature_type} features ===")
    metrics, predictions = train_and_evaluate_models(
        np.concatenate([test_ids, pp_ids_dld]),
        X_train, y_train,
        X_test_td, y_test_td,
        X_dld, y_dld,
        np.concatenate([['TD'] * len(y_test_td), ['DLD'] * len(y_dld)]),
        feature_type
    )

    final_predictions = pd.concat(predictions)
    final_predictions.to_csv('predictions.csv', index=False)

    return X, targets_months, diagnoses, metrics


if __name__ == "__main__":
    all_results = {}

    for feature_type in ["mfcc", "whisper"]:
        print(f"\n=== {feature_type.upper()} Features ===")
        _, _, _, metrics = main(feature_type)
        all_results[feature_type] = metrics

    # Generate comparison report
    comparison_report = []
    for feature_type, metrics_list in all_results.items():
        for model_metrics in metrics_list:
            comparison_report.append({
                'Feature Type': feature_type,
                'Model': model_metrics['model'],
                'Overall MSE': model_metrics['overall']['mse'],
                'TD MAE': model_metrics['TD']['mae'],
                'DLD MAE': model_metrics['DLD']['mae'],
                'DLD Underestimation': model_metrics['DLD']['age_diff']
            })

    pd.DataFrame(comparison_report).to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\nAll results saved to: {output_dir}")