from datasets import load_from_disk
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification
import torch
import argparse
from sklearn.metrics import classification_report
from collections import defaultdict
import numpy as np
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import glob
import tkinter


def pad_mfcc(mfcc_list):
    """Pads all MFCC sequences to the length of the longest one, ensuring proper dimensionality."""
    mfcc_list = [np.atleast_2d(np.array(mfcc)) for mfcc in mfcc_list]  # Ensure 2D

    # Determine the longest MFCC sequence
    target_length = max(mfcc.shape[1] for mfcc in mfcc_list)

    padded_mfccs = []
    for mfcc in mfcc_list:
        num_mfcc, current_length = mfcc.shape  # Extract dimensions

        pad_width = target_length - current_length
        padded_mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')  # Pad only in time axis
        padded_mfccs.append(padded_mfcc)

    return np.array(padded_mfccs)


# Function to extract MFCCs
def extract_mfcc(file, label, sampling_rate=16000, n_mfcc=13, melkwargs=None):
    if label == 1:
        folder = os.path.join(args.audio_dir, 'td')
    elif label == 0:
        folder = os.path.join(args.audio_dir, 'dld')

    original_filename = file  # This is the filename stored in your dataset (without _partX)

    # Find all matching segmented files
    search_pattern = os.path.join(folder, f"{original_filename}_part*.wav")
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"Error: No matching files found for {original_filename} in {folder}")
        return np.zeros(n_mfcc)

    mfcc_list = []

    for filename in matching_files:
        try:
            waveform, sample_rate = torchaudio.load(filename)

            if waveform.numel() == 0:
                print(f"Warning: Empty waveform for {filename}. Skipping.")
                continue

            if waveform.shape[0] > 1:  # Convert stereo to mono
                waveform = waveform.mean(dim=0, keepdim=True)

            if melkwargs is None:
                melkwargs = {'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sampling_rate,
                n_mfcc=n_mfcc,
                melkwargs=melkwargs
            )

            mfcc = mfcc_transform(waveform)
            mfcc = mfcc.mean(dim=-1).squeeze(0).numpy()  # Average over time
            mfcc_list.append(mfcc)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if not mfcc_list:
        print(f"Error: All matching files for {original_filename} failed.")
        return np.zeros(n_mfcc)

    # Concatenate all parts' MFCCs into one vector
    mfcc_concat = np.concatenate(mfcc_list, axis=-1)
    return mfcc_concat


def preprocess_function(examples):
    # Extract the audio data from the examples
    filenames = examples['original_filename']
    labels = examples["label"]

    # Initialize the list to store the MFCC features
    mfcc_features = []

    # Process each audio example
    for i, audio in enumerate(filenames):
        label = labels[i]
        # Assuming extract_mfcc is the function to extract MFCC features
        mfcc = extract_mfcc(audio, label)

        # Check for NaN or Inf values
        if np.isnan(mfcc).any() or np.isinf(mfcc).any():
            print(f"Warning: NaN or Inf found in MFCC for file {audio}. Replacing with zeros.")
            mfcc = np.zeros_like(mfcc)  # Replace with zeros or handle accordingly

        mfcc_features.append(mfcc)

    # Convert the list of MFCCs into a tensor
    mfccs = {
        "mfcc": mfcc_features
    }
    print(f"MFCC shape: {mfcc.shape}")
    print(f"MFCC values: {mfcc}")

    return mfccs


def main(args):
    # Load the pre-split DatasetDict from disk
    split = load_from_disk(args.split_dataset)

    split['train'] = split['train'].map(preprocess_function, batched=True)
    split['test'] = split['test'].map(preprocess_function, batched=True)


    # Verify the distribution in the splits
    print(f"Train set distribution: {split['train']['label'].count(0)}")
    print(f"Train set distribution: {split['train']['label'].count(1)}")
    print(f"Test set distribution: {split['test']['label'].count(0)}")
    print(f"Test set distribution: {split['test']['label'].count(1)}")

    # Extract MFCC features
    mfcc_train = [example['mfcc'] for example in split['train']]
    mfcc_test = [example['mfcc'] for example in split['test']]

    X_train = pad_mfcc(mfcc_train)
    X_test = pad_mfcc(mfcc_test)

    # Extract labels
    y_train = np.array([example['label'] for example in split['train']])
    y_test = np.array([example['label'] for example in split['test']])

    # Check the shape of MFCC features
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Apply t-SNE for dimensionality reduction (e.g., reduce to 2D)
    tsne = TSNE(n_components=2, random_state=42)

    # Flatten MFCCs before applying t-SNE
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)  # Convert to (num_samples, features)
    X_test_flattened += np.random.normal(0, 1e-6, X_test_flattened.shape)  # Tiny noise

    # Apply t-SNE
    X_test_tsne = tsne.fit_transform(X_test_flattened)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_tsne[y_test == 0][:, 0], X_test_tsne[y_test == 0][:, 1], label="DLD", alpha=0.7, color='red')
    plt.scatter(X_test_tsne[y_test == 1][:, 0], X_test_tsne[y_test == 1][:, 1], label="TD", alpha=0.7, color='blue')

    # Add labels and legend
    plt.title("t-SNE Visualization of MFCC Features (Test Data)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend()
    plt.savefig('tnse-test.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Flatten MFCCs before applying t-SNE
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)  # Convert to (num_samples, features)
    X_train_flattened += np.random.normal(0, 1e-6, X_train_flattened.shape)  # Tiny noise

    # Apply t-SNE
    X_train_tsne = tsne.fit_transform(X_train_flattened)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_tsne[y_train == 0][:, 0], X_train_tsne[y_train == 0][:, 1], label="DLD", alpha=0.7, color='red')
    plt.scatter(X_train_tsne[y_train == 1][:, 0], X_train_tsne[y_train == 1][:, 1], label="TD", alpha=0.7, color='blue')

    # Add labels and legend
    plt.title("t-SNE Visualization of MFCC Features (Train Data)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend()
    plt.savefig('tnse-train.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Variables for chunk-level metrics
    chunk_preds, chunk_labels = [], []

    # Variables for original file-level metrics
    original_file_probs = defaultdict(lambda: {"DLD": [], "TD": []})
    original_file_labels = {}  # Store the true label for each original file
    original_file_final_preds = {}  # Store the final predicted label for each original file

    # Load the feature extractor and model
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.finetuned_model_id)
    model = WhisperForAudioClassification.from_pretrained(args.finetuned_model_id)
    # Check the available features in the split

    # Iterate over items in the test set
    for i, item in enumerate(split['test']):
        filename = None  # Initialize filename to a default value
        try:
            # Skip if required keys are missing
            if "input_features" not in item or "label" not in item:
                print(f"Skipping item {i} because it is missing required keys.")
                continue

            input_features = item["input_features"]  # This is the preprocessed audio feature sequence
            label = item["label"]
            filename = item.get("original_filename", "unknown")  # Use a default value if filename is missing
            original_filename = item.get("original_filename", "unknown")  # Get the original filename

            # Debug: print filename
            print(f"Processing file: {filename}")

            # Convert input features to tensor (assuming it's a list of lists or array-like structure)
            input_tensor = torch.tensor(input_features)

            # If the model expects a specific shape (e.g., [batch_size, num_features]),
            # ensure the tensor has the correct shape.
            # Example: If input_features is a sequence of features, you might need to add a batch dimension:
            input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension if needed

            # Run inference
            with torch.no_grad():
                logits = model(input_tensor).logits  # Pass the processed input through the model

            # Get predicted label
            predicted_class_ids = torch.argmax(logits).item()
            predicted_label = model.config.id2label[predicted_class_ids]
            true_label = model.config.id2label[label]
            print(f'File name: {filename}')
            print(f'Predicted label for item {i}: {predicted_label}')
            print(f'True label for item {i}: {true_label}')
            print()

            # Store chunk-level predictions and labels
            chunk_preds.append(predicted_label)
            chunk_labels.append(true_label)

            # Get probabilities for each label
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()  # Convert to probabilities
            for label_idx, label_name in model.config.id2label.items():
                original_file_probs[original_filename][label_name].append(probs[label_idx])

            # Store the true label for the original file (use the first chunk's label)
            if original_filename not in original_file_labels:
                original_file_labels[original_filename] = true_label

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    # Calculate mean probabilities and final labels for each original file
    original_file_preds, original_file_true_labels = [], []
    print("\nMean probabilities and final labels by original filename:")
    for original_filename, probs_dict in original_file_probs.items():
        mean_probs = {
            label: np.mean(probs) for label, probs in probs_dict.items()
        }
        final_label = max(mean_probs, key=mean_probs.get)  # Assign the label with the highest mean probability
        true_label = original_file_labels[original_filename]  # Get the true label for the original file

        print(f"Original file: {original_filename}")
        print(f"Mean probabilities: {mean_probs}")
        print(f"Final label: {final_label}")
        print(f"True label: {true_label}")
        print()

        # Store original file-level predictions and labels
        original_file_preds.append(final_label)
        original_file_true_labels.append(true_label)

    # Handle empty labels or preds
    if not chunk_labels or not chunk_preds:
        print("No files were successfully processed. Cannot generate classification report.")
    else:
        # Print chunk-level metrics
        print("Chunk-level metrics:")
        print(classification_report(chunk_labels, chunk_preds))

    # Print original file-level metrics
    if original_file_true_labels and original_file_preds:
        print("Original file-level metrics:")
        print(classification_report(original_file_true_labels, original_file_preds))
    else:
        print("No original files were processed. Cannot generate classification report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dataset", type=str, default="finetuning_data/split_dataset")
    parser.add_argument("--audio_dir", type=str,
                        default="C:/Users/a.stasica/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener/MODEL/Segmented_4_chins_whisper")
    parser.add_argument("--finetuned_model_id", type=str, default="finetuned_model/4yo_childes_auris_whisper")
    args = parser.parse_args()
    main(args)


