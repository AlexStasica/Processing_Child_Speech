from datasets import Dataset, Audio
from huggingface_hub import login
import glob
import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr


def split_audio(file_list, segmented_audio_dir, sub_dir, n_seconds=30):
    '''
    Given a list of wav files, split each file into segments of a pre-defined length
    '''
    # Split files into segments
    split_duration = n_seconds * 1000  # in milliseconds

    # Iterate through file list
    for file in file_list:
        # Load the audio file
        audio = AudioSegment.from_wav(file)

        # Calculate how many segments are needed
        num_segments = len(audio) // split_duration + (1 if len(audio) % split_duration != 0 else 0)

        # Split and export the audio
        for i in range(num_segments):
            start_time = i * split_duration
            end_time = (i + 1) * split_duration

            # Extract the segment
            segment = audio[start_time:end_time]

            # Create new filename
            new_filename = f'{file.split("\\")[-2]}_part_{i + 1}.wav'

            # Save the segment to a new file
            segment.export(f"{segmented_audio_dir}/{sub_dir}/{new_filename}", format="wav")


def main(data_dir):
    # Check if the 'segmented_audio' directory exists
    segmented_audio_dir = f"{data_dir}/Segmented_age_whisper"
    if not os.path.exists(segmented_audio_dir):
        # Folder doesn't exist, so create the folder + split the audio files
        print(f"The Segmented_age_whisper folder does not exist yet. Splitting files...")
        os.makedirs(segmented_audio_dir)

        # Create sub dirs for TD (per age) and DLD
        td_subdirs = {
            '3_0_3_6': '3_0_3_6',
            '3_7_3_12': '3_7_3_12',
            '4_0_4_6': '4_0_4_6',
            '4_7_4_12': '4_7_4_12',
            '5_0_5_6': '5_0_5_6',
            '5_7_5_12': '5_7_5_12'
        }

        for subdir in td_subdirs.values():
            os.makedirs(f'{segmented_audio_dir}/td/{subdir}', exist_ok=True)

        dld_subdirs = {
            '4_0_4_6': '4_0_4_6',
            '4_7_4_12': '4_7_4_12',
            '5_0_5_6': '5_0_5_6',
            '5_7_5_12': '5_7_5_12'
        }

        for subdir in dld_subdirs.values():
            os.makedirs(f'{segmented_audio_dir}/dld/{subdir}', exist_ok=True)

        # Get file paths for each age group
        files_td = {
            '3_0_3_6': glob.glob(f'{data_dir}/Childes_3yo/TD/3_0_3_6/*/*_chi_chins.wav'),
            '3_7_3_12': glob.glob(f'{data_dir}/Childes_3yo/TD/3_7_3_12/*/*_chi_chins.wav'),
            '4_0_4_6': glob.glob(f'{data_dir}/Childes_4yo/TD/4_0_4_6/*/*_chi_chins.wav'),
            '4_7_4_12': glob.glob(f'{data_dir}/Childes_4yo/TD/4_7_4_12/*/*_chi_chins.wav'),
            '5_0_5_6': glob.glob(f'{data_dir}/Childes_5yo/TD/5_0_5_6/*/*_chi_chins.wav'),
            '5_7_5_12': glob.glob(f'{data_dir}/Childes_5yo/TD/5_7_5_12/*/*_chi_chins.wav')
        }

        files_dld = {
            '4_0_4_6': glob.glob(f'{data_dir}/Childes_4yo/TOS/4_0_4_6/*/*_chi_chins.wav'),
            '4_7_4_12': glob.glob(f'{data_dir}/Childes_4yo/TOS/4_7_4_12/*/*_chi_chins.wav'),
            '5_0_5_6': glob.glob(f'{data_dir}/Childes_5yo/TOS/5_0_5_6/*/*_chi_chins.wav'),
            '5_7_5_12': glob.glob(f'{data_dir}/Childes_5yo/TOS/5_7_5_12/*/*_chi_chins.wav')
        }


        # Print file counts
        for age_group, files in files_td.items():
            print(f"TD {age_group} CHILDES:", len(files))

        for age_group, files in files_dld.items():
            print(f"DLD {age_group} CHILDES:", len(files))

        # Split files into segments of 30 seconds and save in appropriate subfolders
        for age_group, files in files_td.items():
            split_audio(files, segmented_audio_dir, sub_dir=f'td/{age_group}', n_seconds=30)

        for age_group, files in files_dld.items():
            split_audio(files, segmented_audio_dir, sub_dir=f'dld/{age_group}', n_seconds=30)
    else:
        print("The 'Segmented' folder already exists. Skipping the splitting process.")

    # First define a mapping from folder names to simplified labels
    age_group_mapping = {
        '3_0_3_6': '3_1',
        '3_7_3_12': '3_2',
        '4_0_4_6': '4_1',
        '4_7_4_12': '4_2',
        '5_0_5_6': '5_1',
        '5_7_5_12': '5_2'
    }

    # Get segmented files with proper labels
    files_td_segmented = []
    td_labels = []
    for age_folder, label in age_group_mapping.items():
        age_files = glob.glob(f'{segmented_audio_dir}/td/{age_folder}/*.wav')
        files_td_segmented.extend(age_files)
        td_labels.extend([label] * len(age_files))

    files_dld_segmented = []
    dld_labels = []

    # DLD only has 4 and 5 year olds, but we'll use the same age mapping
    for age_folder, label in age_group_mapping.items():
        if not age_folder.startswith('3_'):  # Skip 3yo folders for DLD
            age_files = glob.glob(f'{segmented_audio_dir}/dld/{age_folder}/*.wav')
            files_dld_segmented.extend(age_files)
            dld_labels.extend([label] * len(age_files))

    # Print detailed counts
    print("\nFile counts by age group:")
    print("TD children:")
    for label in set(td_labels):
        print(f"{label}: {td_labels.count(label)}")

    print("\nDLD children (chronological age):")
    for label in set(dld_labels):
        print(f"{label}: {dld_labels.count(label)}")

    # Combine all files and labels
    files = files_td_segmented + files_dld_segmented
    labels = td_labels + dld_labels

    # Add source information (TD or DLD)
    is_dld = [0] * len(files_td_segmented) + [1] * len(files_dld_segmented)

    print(f"\nTotal number of files: {len(files)}")
    print(f"Total number of labels: {len(labels)}")

    # Create the dataset dictionary with additional metadata
    data = {
        "audio": files,
        "label": labels,
        "is_dld": is_dld,
        "chronological_age": labels
    }

    # Create the dataset object
    dataset = Dataset.from_dict(data).cast_column("audio", Audio())

    return dataset


if __name__ == "__main__":
    data_dir = 'C:/Users/a.stasica/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener/Processed_CHILDES'
    dataset = main(data_dir)


    # Save to disk
    dataset.save_to_disk(f"finetuning_data/age_classification_childes_whisper")
