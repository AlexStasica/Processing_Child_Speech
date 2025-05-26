from datasets import Dataset, Audio
from huggingface_hub import login
import glob
import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from reduce_noise import reduce_noise


def split_audio(file_list, segmented_audio_dir, sub_dir, n_seconds=30):
    '''
    Given a list of wav files, split each file into segments of a pre-defined length,
    skipping files that have already been segmented.
    '''
    split_duration = n_seconds * 1000  # in milliseconds

    # Ensure the subdirectory exists
    sub_dir_path = os.path.join(segmented_audio_dir, sub_dir)
    os.makedirs(sub_dir_path, exist_ok=True)

    # Get already segmented files
    existing_segments = set(f for f in os.listdir(sub_dir_path) if f.endswith(".wav"))

    file_list = reduce_noise(file_list)

    for file in file_list:
        base_filename = os.path.basename(file)  # Original file name

        # Check if this file has already been segmented
        if any(f.startswith(base_filename.replace('.wav', '_part_')) for f in existing_segments):
            print(f"Skipping already segmented file: {base_filename}")
            continue

        # Load the audio file
        audio = AudioSegment.from_wav(file)

        # Calculate the number of segments
        num_segments = len(audio) // split_duration + (1 if len(audio) % split_duration != 0 else 0)

        # Split and export the audio
        for i in range(num_segments):
            start_time = i * split_duration
            end_time = (i + 1) * split_duration
            segment = audio[start_time:end_time]
            new_filename = f"{base_filename.replace('.wav', '')}_part_{i + 1}.wav"
            segment.export(os.path.join(sub_dir_path, new_filename), format="wav")


def main(data_dir, age_group):
    segmented_audio_dir = f"{data_dir}/MODEL/Segmented_4_chins_whisper_alldenoised"
    os.makedirs(segmented_audio_dir, exist_ok=True)

    # Ensure TD and DLD subdirectories exist
    os.makedirs(f'{segmented_audio_dir}/td', exist_ok=True)
    os.makedirs(f'{segmented_audio_dir}/dld', exist_ok=True)

    # Load Auris data
    files_td_auris = glob.glob(f'{data_dir}/{age_group}/TD*/Formatted/*/*_chi_chins.wav')
    files_tos_auris = glob.glob(f'{data_dir}/{age_group}/TOS*/Formatted/*/*_chi_chins.wav')
    files_vvtos_auris = glob.glob(f'{data_dir}/{age_group}/vvTOS*/Formatted/*/*_chi_chins.wav')
    files_dld_auris = files_tos_auris + files_vvtos_auris

    if age_group == '3yo':
        files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/*/*_chi_chins.wav')
        files_td = files_td_auris + files_td_childes
    else:
        files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TD/*/*/*_chi_chins.wav')
        files_dld_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TOS/*/*/*_chi_chins.wav')
        files_td = files_td_auris + files_td_childes
        files_dld = files_dld_auris + files_dld_childes
        print("TD CHILDES:", len(files_td_childes), "DLD CHILDES:", len(files_dld_childes), "TD AURIS:",
              len(files_td_auris), "DLD AURIS:", len(files_dld_auris))

    # Perform segmentation, skipping already segmented files
    split_audio(files_td, segmented_audio_dir, sub_dir='td', n_seconds=30)
    split_audio(files_dld, segmented_audio_dir, sub_dir='dld', n_seconds=30)


    # Load the segmented files
    files_td_segmented = glob.glob(f'{segmented_audio_dir}/td/*.wav')
    files_dld_segmented = glob.glob(f'{segmented_audio_dir}/dld/*.wav')

    # Print the number of files for each class
    print(f"Number of TD files: {len(files_td_segmented)}")
    print(f"Number of DLD files: {len(files_dld_segmented)}")

    # Get label list for each class
    labels_td = len(files_td_segmented) * ['TD']
    labels_dld = len(files_dld_segmented) * ['DLD']

    # Create a single list with all files/labels from the two classes
    files = files_td_segmented + files_dld_segmented
    labels = labels_td + labels_dld

    print(f"Total number of files: {len(files)}")
    print(f"Total number of labels: {len(labels)}")

    # Create a dictionary with the audio file paths and corresponding labels
    data = {
        "audio": files,
        "label": labels
    }

    # Create the dataset object
    dataset = Dataset.from_dict(data).cast_column("audio", Audio())

    return dataset


if __name__ == "__main__":
    data_dir = 'C:/Users/a.stasica/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener'
    age_group = '4yo'
    dataset = main(data_dir, age_group)

    print(dataset[0])
    
    # Save to disk
    dataset.save_to_disk(f"finetuning_data/{age_group}_childes_auris_alldenoised_whisper")
