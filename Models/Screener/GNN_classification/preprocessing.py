import pandas as pd
import numpy as np
import torch
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DLDDataPreprocessor:
    """
    Data preprocessing class for DLD audio classification
    """

    def __init__(self, target_sr=16000, test_size=0.2, random_state=42):
        self.target_sr = target_sr
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def clean_pp_name(self, pp_name):
        """Clean the PP name by removing .wav if present and any whitespace"""
        if pd.isna(pp_name):
            return None
        pp_name = str(pp_name).strip()
        if pp_name.lower().endswith('.wav'):
            pp_name = pp_name[:-4]
        return pp_name

    def convert_age_to_months(self, age_str):
        """Convert age string 'year;months' or 'year:months' to total months"""
        if pd.isna(age_str):
            return None
        try:
            # Handle both ';' and ':' separators
            if ';' in str(age_str):
                years, months = map(int, str(age_str).split(';'))
            elif ':' in str(age_str):
                years, months = map(int, str(age_str).split(':'))
            else:
                # If no separator, assume it's just years
                years = int(float(age_str))
                months = 0
            return years * 12 + months
        except (ValueError, IndexError):
            print(f"Warning: Could not parse age '{age_str}'")
            return None

    def find_audio_file(self, pp_name, full_path):
        """Find audio file using the provided full path"""
        if pd.isna(pp_name) or pd.isna(full_path):
            return None

        # Multiple filename patterns to try
        filename_patterns = [
            f"{pp_name}_chi_chins.wav",
            f"{pp_name}_audio_chi_chins.wav",
            f"{pp_name}.wav",
            f"{pp_name}_processed.wav"
        ]

        for pattern in filename_patterns:
            audio_path = os.path.join(full_path, pattern)
            if os.path.exists(audio_path):
                return audio_path

        return None

    def load_audio(self, file_path):
        try:
            # Load audio file and convert to mono if needed
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_and_process_csv(self, csv_path):
        """Load and validate dataset from CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_columns = {'PP', 'Leeftijd', 'Diagnose', 'Full Path'}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"CSV is missing required columns: {missing_cols}")

        # Clean and validate data
        df['PP'] = df['PP'].apply(self.clean_pp_name)
        df = df.dropna(subset=['PP', 'Leeftijd', 'Diagnose', 'Full Path'])

        # Convert ages to months
        df['Age_Months'] = df['Leeftijd'].apply(self.convert_age_to_months)
        df = df.dropna(subset=['Age_Months'])

        valid_data = []
        missing_files = []

        print("Processing audio files...")
        for idx, row in df.iterrows():
            pp_name = row['PP']
            audio_path = self.find_audio_file(pp_name, row['Full Path'])

            if audio_path and os.path.exists(audio_path):
                valid_data.append({
                    'pp_id': pp_name,
                    'audio_path': audio_path,
                    'age_months': row['Age_Months'],
                    'diagnosis': row['Diagnose'],
                    'original_age': row['Leeftijd']
                })
            else:
                missing_files.append(pp_name)

        # Create final DataFrame
        processed_df = pd.DataFrame(valid_data)

        print(f"\nDataset Summary:")
        print(f"- Total entries in CSV: {len(df)}")
        print(f"- Valid audio files found: {len(processed_df)}")
        print(f"- Missing audio files: {len(missing_files)}")
        if len(processed_df) > 0:
            print(f"- Diagnosis distribution: {processed_df['diagnosis'].value_counts().to_dict()}")
            print(
                f"- Age range: {processed_df['age_months'].min():.1f} - {processed_df['age_months'].max():.1f} months")

        if missing_files:
            print(f"- Sample missing files: {missing_files[:5]}")

        return processed_df

    def create_train_val_test_split(self, processed_df, test_size=0.2, val_size=0.2):
        """Create stratified train/validation/test split"""
        if len(processed_df) == 0:
            raise ValueError("No data to split")

        # Create age groups for better stratification
        age_groups = pd.cut(processed_df['age_months'],
                            bins=min(4, len(processed_df)),
                            labels=False)

        # Combine diagnosis and age group for stratification
        stratify_labels = [f"{diag}_{age}" for diag, age in zip(processed_df['diagnosis'], age_groups)]

        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            range(len(processed_df)),
            test_size=test_size,
            stratify=stratify_labels,
            random_state=self.random_state
        )

        # Second split: separate validation from train
        # Adjust val_size to be relative to train+val portion
        relative_val_size = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            stratify=[stratify_labels[i] for i in train_val_idx],
            random_state=self.random_state
        )

        train_df = processed_df.iloc[train_idx].reset_index(drop=True)
        val_df = processed_df.iloc[val_idx].reset_index(drop=True)
        test_df = processed_df.iloc[test_idx].reset_index(drop=True)

        print(f"\nTrain/Validation/Test Split:")
        print(f"- Training samples: {len(train_df)}")
        print(f"- Validation samples: {len(val_df)}")
        print(f"- Test samples: {len(test_df)}")
        print(f"- Train diagnosis distribution: {train_df['diagnosis'].value_counts().to_dict()}")
        print(f"- Val diagnosis distribution: {val_df['diagnosis'].value_counts().to_dict()}")
        print(f"- Test diagnosis distribution: {test_df['diagnosis'].value_counts().to_dict()}")

        return train_df, val_df, test_df

    def prepare_audio_and_labels(self, df, fit_encoder=True):
        """
        Prepare audio samples and labels for the classifier
        Returns: (audio_samples, labels)
        """
        audio_samples = []
        labels = []

        print(f"Loading {len(df)} audio files...")

        for idx, row in df.iterrows():
            # Load audio
            audio = self.load_audio(row['audio_path'])

            if audio is not None:
                audio_samples.append(audio)

                # Encode diagnosis labels
                if fit_encoder:
                    # Fit the encoder on first use
                    if not hasattr(self.label_encoder, 'classes_'):
                        self.label_encoder.fit(df['diagnosis'].unique())

                labels.append(self.label_encoder.transform([row['diagnosis']])[0])
            else:
                print(f"Failed to load audio for {row['pp_id']}")

        print(f"Successfully loaded {len(audio_samples)} audio samples")

        return audio_samples, labels


def load_dld_data(csv_path, test_size=0.2, val_size=0.2, target_sr=16000, random_state=42):
    """
    Main function to load and preprocess data
    Returns: train_audio, train_labels, val_audio, val_labels, test_audio, test_labels
    """
    # Initialize preprocessor
    preprocessor = DLDDataPreprocessor(
        target_sr=target_sr,
        test_size=test_size,
        random_state=random_state
    )

    try:
        # Load and process dataset
        print("Loading dataset...")
        processed_df = preprocessor.load_and_process_csv(csv_path)

        if len(processed_df) == 0:
            print("Error: No valid data found.")
            return None, None, None, None, None, None

        # Create train/val/test split
        print("\nCreating train/validation/test split...")
        train_df, val_df, test_df = preprocessor.create_train_val_test_split(
            processed_df,
            test_size=test_size,
            val_size=val_size
        )

        # Prepare data for classifier
        print("\nPreparing training data...")
        train_audio, train_labels = preprocessor.prepare_audio_and_labels(train_df, fit_encoder=True)

        print("Preparing validation data...")
        val_audio, val_labels = preprocessor.prepare_audio_and_labels(val_df, fit_encoder=False)

        print("Preparing test data...")
        test_audio, test_labels = preprocessor.prepare_audio_and_labels(test_df, fit_encoder=False)

        print(f"\nData preprocessing completed!")
        print(f"- Training: {len(train_audio)} samples")
        print(f"- Validation: {len(val_audio)} samples")
        print(f"- Testing: {len(test_audio)} samples")
        print(f"- Label mapping: {dict(zip(preprocessor.label_encoder.classes_, range(len(preprocessor.label_encoder.classes_))))}")

        return train_audio, train_labels, val_audio, val_labels, test_audio, test_labels

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None, None, None, None, None