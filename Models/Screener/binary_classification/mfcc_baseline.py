import torch
import torchaudio
import joblib
import numpy as np
from datasets import load_dataset, load_from_disk, Audio, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os
from finetune_wav2vec2 import balance_dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Function to extract MFCCs
def extract_mfcc(file, label, sampling_rate=16000, n_mfcc=13, melkwargs=None):
    if label == 1:
        filename = args.audio_dir + 'td/' + file
    elif label == 0:
        filename = args.audio_dir + 'dld/' + file
    
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except Exception as e:
        print(f"Error opening file: {e}")
        mfcc = torch.zeros(13)
        print(mfcc.shape)
        return mfcc

    if waveform.shape[1] != 0:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)  # Averaging the channels to get a mono signal

        # Extract MFCC
        if melkwargs is None:
            melkwargs = {'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sampling_rate, 
            n_mfcc=n_mfcc,
            melkwargs=melkwargs
        )(waveform)

        # Average MFCC features across time steps
        mfcc = mfcc.mean(dim=-1).squeeze(0).numpy()
    else:
        mfcc = torch.zeros(13)
        print(mfcc.shape)
    return mfcc

def preprocess_function(examples):
    # Extract the audio data from the examples
    filenames = examples['filename']
    labels = examples["label"]

    # Initialize the list to store the MFCC features
    mfcc_features = []
    
    # Process each audio example
    for i, audio in enumerate(filenames):
        label = labels[i]
        # Assuming extract_mfcc is the function to extract MFCC features
        mfcc = extract_mfcc(audio, label)
        mfcc_features.append(mfcc) 
    
    # Convert the list of MFCCs into a tensor
    mfccs = {
        "mfcc": mfcc_features
    }
    
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

    # Extract input values (MFCC features) and labels
    X_train = np.array([example['mfcc'] for example in split['train']])
    y_train = np.array([example['label'] for example in split['train']])
    X_test = np.array([example['mfcc'] for example in split['test']])
    y_test = np.array([example['label'] for example in split['test']])

    # Apply t-SNE for dimensionality reduction (e.g., reduce to 2D)
    tsne = TSNE(n_components=2, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test)
    
    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_tsne[y_test == 0][:, 0], X_test_tsne[y_test == 0][:, 1], label="DLD", alpha=0.7, color='red')
    plt.scatter(X_test_tsne[y_test == 1][:, 0], X_test_tsne[y_test == 1][:, 1], label="TD", alpha=0.7, color='blue')
    
    # Add labels and legend
    plt.title("t-SNE Visualization of MFCC Features (Test Data)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend()
    plt.savefig('tnse.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Train a classifier (Random Forest in this case)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # # Save the model
    # if not os.path.exists(args.model_output_dir):
    #     os.makedirs(args.model_output_dir)
    
    # model_output_path = os.path.join(args.model_output_dir, "random_forest_model.pkl")
    # joblib.dump(clf, model_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_id", type=str, default="finetuning_data/4yo_childes_auris_UPDATED_large")
    parser.add_argument("--model_output_dir", type=str, default="mfcc_baseline_model")
    parser.add_argument("--audio_dir", type=str, default="")
    parser.add_argument("--split_dataset", type=str, default="finetuning_data/split_dataset")
    args = parser.parse_args()
    main(args)
