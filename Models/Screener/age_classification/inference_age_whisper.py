import pandas as pd
from transformers import WhisperForAudioClassification, WhisperFeatureExtractor
import torch
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict
import os
import argparse
import glob
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D


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


def extract_mfcc(file, label, chronological_age, is_dld, args, sampling_rate=16000, n_mfcc=13, melkwargs=None):
    # Define base folder for td or dld
    base_folder_td = 'td'
    base_folder_dld = 'dld'

    # Correct mapping: Folder names -> Labels
    age_group_mapping = {
        '3_0_3_6': '3_1',
        '3_7_3_12': '3_2',
        '4_0_4_6': '4_1',
        '4_7_4_12': '4_2',
        '5_0_5_6': '5_1',
        '5_7_5_12': '5_2'
    }

    # Reverse mapping: Labels -> Folder names
    reversed_mapping = {v: k for k, v in age_group_mapping.items()}

    # Find the correct folder name for this label
    age_group_folder = reversed_mapping.get(chronological_age, None)

    if age_group_folder is None:
        print(f" Warning: Unknown age group {chronological_age} for file {file}. Skipping.")
        return np.zeros(n_mfcc)

    # Construct path for both td and dld folders within the correct age group
    folder_td = os.path.join(args.audio_dir, base_folder_td, age_group_folder)
    folder_dld = os.path.join(args.audio_dir, base_folder_dld, age_group_folder)

    original_filename = os.path.splitext(os.path.basename(file))[0]  # Remove extension if present

    # Debugging: Print the expected search pattern
    search_pattern_td = os.path.join(folder_td, f"{original_filename}_part*.wav")
    search_pattern_dld = os.path.join(folder_dld, f"{original_filename}_part*.wav")
    print(f" Looking for files in td: {file}")
    print(f" Looking for files in dld: {file}")

    # Look for files first in td folder
    matching_files_td = glob.glob(search_pattern_td)
    if not matching_files_td:
        # If not found in td, look in dld folder
        matching_files_dld = glob.glob(search_pattern_dld)
        if not matching_files_dld:
            print(f" Error: No matching files found for {original_filename} in {folder_td} or {folder_dld}")
            return np.zeros(n_mfcc)
        else:
            matching_files = matching_files_dld
            print(f" Found files in dld: {matching_files}")
    else:
        matching_files = matching_files_td
        print(f" Found files in td: {matching_files}")

    mfcc_list = []
    for filename in matching_files:
        try:
            waveform, sample_rate = torchaudio.load(filename)
            if waveform.numel() == 0:
                print(f" Warning: Empty waveform for {filename}. Skipping.")
                continue

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if melkwargs is None:
                melkwargs = {'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sampling_rate, n_mfcc=n_mfcc, melkwargs=melkwargs
            )

            mfcc = mfcc_transform(waveform)
            mfcc = mfcc.mean(dim=-1).squeeze(0).numpy()
            mfcc_list.append(mfcc)

        except Exception as e:
            print(f" Error processing file {filename}: {e}")

    if not mfcc_list:
        print(f" Error: All matching files for {original_filename} failed.")
        return np.zeros(n_mfcc)

    return np.concatenate(mfcc_list, axis=-1)


def preprocess_function(examples, args):

    filenames = examples['original_filename']
    labels = examples["label"]
    is_dld = examples['is_dld']
    ages = examples['chronological_age']
    for filename, label, dld, age in zip(filenames, labels, is_dld, ages):
        print(f"Processing Example: {filename}, Label: {label}, DLD: {dld}, Age: {age}")
    mfcc_features = []

    for i, audio in enumerate(filenames):
        label = labels[i]
        age = ages[i]
        mfcc = extract_mfcc(audio, label, age, is_dld, args)

        if np.isnan(mfcc).any() or np.isinf(mfcc).any():
            print(f"Warning: NaN or Inf found in MFCC for file {audio}. Replacing with zeros.")
            mfcc = np.zeros_like(mfcc)

        mfcc_features.append(mfcc)

    return {"mfcc": mfcc_features}


def visualize_mfcc(split, args):
    age_group_mapping = {
        '3_0_3_6': '3_1',
        '3_7_3_12': '3_2',
        '4_0_4_6': '4_1',
        '4_7_4_12': '4_2',
        '5_0_5_6': '5_1',
        '5_7_5_12': '5_2'
    }

    # Reverse mapping from chronological age to folder range
    reverse_age_group_mapping = {
        '3_1': '3_0_3_6',
        '3_2': '3_7_3_12',
        '4_1': '4_0_4_6',
        '4_2': '4_7_4_12',
        '5_1': '5_0_5_6',
        '5_2': '5_7_5_12'
    }

    age_colors = {
        '3_1': '#1f77b4',
        '3_2': '#aec7e8',
        '4_1': '#ff7f0e',
        '4_2': '#ffbb78',
        '5_1': '#2ca02c',
        '5_2': '#98df8a'
    }

    dld_colors = {
        '4_1': '#d62728',
        '4_2': '#ff9896',
        '5_1': '#9467bd',
        '5_2': '#c5b0d5'
    }

    split['train'] = split['train'].map(lambda x: preprocess_function(x, args), batched=True)
    split['test'] = split['test'].map(lambda x: preprocess_function(x, args), batched=True)

    mfcc_train = [example['mfcc'] for example in split['train']]
    mfcc_test = [example['mfcc'] for example in split['test']]

    X_train = pad_mfcc(mfcc_train)
    X_test = pad_mfcc(mfcc_test)

    # Get age labels and DLD status
    y_train_age = [reverse_age_group_mapping[example['chronological_age']]
                   for example in split['train']]
    print("Train Age Groups:", y_train_age)
    y_test_age = [reverse_age_group_mapping[example['chronological_age']]
                  if example['is_dld'] == 0
                  else example['chronological_age']
                  for example in split['test']]

    for example in split['train'][:5]:  # Check first 5 examples
        print("Chronological Age:", example['chronological_age'], "Mapped Age:",
              reverse_age_group_mapping.get(example['chronological_age'], "Not Found"))
    is_dld_train = [example['is_dld'] for example in split['train']]
    print("Train DLD Distribution:", sum(is_dld_train), "DLD children in train set")
    is_dld_test = [example['is_dld'] for example in split['test']]

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)

    # ===== TRAIN SET VISUALIZATION (TD only) =====
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_train_tsne = tsne.fit_transform(X_train_flattened)

    plt.figure(figsize=(12, 8))
    for age_group in age_group_mapping.values():
        mask = np.array(y_train_age) == age_group
        print(f"Plotting Age Group: {age_group}, Mask: {mask.sum()} entries")

        plt.scatter(X_train_tsne[mask, 0], X_train_tsne[mask, 1],
                    label=age_group, alpha=0.7, color=age_colors[age_group])

    plt.title("t-SNE of TD Children by Age Group (Train Set)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Age Groups")
    plt.savefig('tsne_train_td_by_age.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== TEST SET VISUALIZATION (TD + DLD) =====
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)
    X_test_tsne = tsne.fit_transform(X_test_flattened)

    plt.figure(figsize=(14, 10))

    # Plot TD children (with age groups)
    for age_group in age_group_mapping.values():
        mask = (np.array(y_test_age) == age_group) & (np.array(is_dld_test) == 0)
        if any(mask):
            plt.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1],
                        label=f"TD {age_group}", alpha=0.7, color=age_colors[age_group],
                        edgecolors='k', linewidths=0.5)

    # Plot DLD children (with their true age)
    for age_group in ['4_1', '4_2', '5_1', '5_2']:
        mask = (np.array(y_test_age) == age_group) & (np.array(is_dld_test) == 1)
        if any(mask):
            plt.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1],
                        label=f"DLD ({age_group})", alpha=0.8, color=dld_colors[age_group],
                        marker='X', s=100, edgecolors='k', linewidths=0.8)

    # Add arrows showing DLD children's age vs position
    for i, (is_dld, true_age) in enumerate(zip(is_dld_test, y_test_age)):
        if is_dld:
            # Find centroid of TD children of same age
            td_mask = (np.array(y_test_age) == true_age) & (np.array(is_dld_test) == 0)
            if any(td_mask):
                td_center = np.mean(X_test_tsne[td_mask], axis=0)
                plt.arrow(td_center[0], td_center[1],
                          X_test_tsne[i, 0] - td_center[0], X_test_tsne[i, 1] - td_center[1],
                          color='gray', alpha=0.4, width=0.01, head_width=0.2)

    plt.title("t-SNE of Test Set (TD by Age vs DLD with True Age Markers)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='TD Children',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='DLD Children',
               markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements + [
        Line2D([0], [0], marker='o', color='w', label=age,
               markerfacecolor=age_colors[age], markersize=10)
        for age in age_group_mapping.values()
    ], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('tsne_test_td_dld_by_age.png', dpi=300, bbox_inches='tight')
    plt.show()


def main(args):
    split = load_from_disk(args.split_dataset)
    print(split.column_names)
    # print(split['train'][0]['input_features'])
    visualize_mfcc(split, args)
    #{'train': ['label', 'is_dld', 'chronological_age', 'original_filename', 'input_features'],
     # 'test': ['label', 'is_dld', 'chronological_age', 'original_filename', 'input_features']}

    # Load model
    model = WhisperForAudioClassification.from_pretrained(args.finetuned_model_id).eval()

    # Age group mappings
    age_groups = ['3_1', '3_2', '4_1', '4_2', '5_1', '5_2']
    id2label = {i: age for i, age in enumerate(age_groups)}
    label2id = {age: i for i, age in enumerate(age_groups)}  # This was missing

    # Load test dataset
    test_dataset = load_from_disk(args.split_dataset)['test']

    # Prepare results storage
    results = {
        'filename': [],
        'true_age': [],
        'predicted_age': [],
        'is_dld': [],
        'is_predicted_younger': [],
        **{f'prob_{age}': [] for age in age_groups}
    }

    # Process all samples
    for item in test_dataset:
        # Convert input features to tensor
        inputs = {
            "input_features": torch.tensor(item["input_features"]).unsqueeze(0)
        }

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            pred_age = id2label[torch.argmax(logits).item()]
            true_age = id2label[item["label"]]

        # Check if DLD is predicted younger
        is_younger = None
        if item["is_dld"]:
            pred_idx = age_groups.index(pred_age)
            true_idx = age_groups.index(true_age)
            is_younger = pred_idx < true_idx

        # Store results
        results['filename'].append(item["original_filename"])
        results['true_age'].append(true_age)
        results['predicted_age'].append(pred_age)
        results['is_dld'].append(item["is_dld"])
        results['is_predicted_younger'].append(is_younger)
        for age in age_groups:
            results[f'prob_{age}'].append(probs[label2id[age]])  # Now using defined label2id

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Generate reports
    print("\nClassification Report (All Samples):")
    print(classification_report(df['true_age'], df['predicted_age'], target_names=age_groups))

    # DLD-specific analysis
    if df['is_dld'].any():
        print("\nDLD Children Analysis:")
        dld_df = df[df['is_dld'] == 1]
        for age in ['4_1', '4_2', '5_1', '5_2']:
            age_df = dld_df[dld_df['true_age'] == age]
            if len(age_df) > 0:
                younger_pct = age_df['is_predicted_younger'].mean() * 100
                print(f"{age}: {younger_pct:.1f}% predicted younger")

    # Save to Excel
    output_path = os.path.join(args.finetuned_model_id, "predictions.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dataset", type=str, default="finetuning_data/split_dataset_age")
    parser.add_argument("--audio_dir", type=str,
                        default="")
    parser.add_argument("--finetuned_model_id", type=str, default="finetuned_model/age_classification_childes_whisper")
    args = parser.parse_args()
    main(args)
