from datasets import Dataset, Audio
from huggingface_hub import login
import glob
import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr

def split_audio(file_list, segmented_audio_dir, sub_dir, n_seconds=10):
    '''
    Given a list of wav files, split each file into segments of a pre-defined length.
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
            new_filename = f'{file.split('\\')[-2]}_part_{i+1}.wav'

            # Save the segment to a new file
            segment.export(f"{segmented_audio_dir}/{sub_dir}/{new_filename}", format="wav")


def main(data_dir, age_group):

    # Check if the 'segmented_audio' directory exists
    segmented_audio_dir = f"{data_dir}/{age_group}/Segmented"
    if not os.path.exists(segmented_audio_dir):

        # Folder doesn't exist, so create the folder + split the audio files
        print("The 'Segmented' folder does not exist yet. Splitting files...")
        os.makedirs(segmented_audio_dir)

        # Also create sub dirs for TD and DLD
        os.makedirs(f'{segmented_audio_dir}/td')
        os.makedirs(f'{segmented_audio_dir}/dld')

        # Load Auris data
        files_td_auris = glob.glob(f'{data_dir}/{age_group}/TD*/Formatted/*/*chi.wav')
        files_tos_auris = glob.glob(f'{data_dir}/{age_group}/TOS*/Formatted/*/*chi.wav')
        files_vvtos_auris = glob.glob(f'{data_dir}/{age_group}/vvTOS*/Formatted/*/*chi.wav')
        files_dld_auris = files_tos_auris + files_vvtos_auris

        # For Childes, we use our 'reduced noise' files
        if age_group == '3yo':
            files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/*/*chi_reduced_noise.wav')
            # Combine data from Childes and Auris (only for TD)
            files_td = files_td_auris + files_td_childes

        elif age_group == '4yo':
            files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TD/*/*chi_reduced_noise.wav')
            files_dld_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/DLD/*/*chi_reduced_noise.wav')
            # Combine data from Childes and Auris
            files_td = files_td_auris + files_td_childes
            files_dld = files_dld_auris + files_dld_childes
            print(len(files_td_childes), len(files_dld_childes), len(files_td_auris), len(files_dld_auris))

        # Split files into segments of 10 seconds
        split_audio(files_td, segmented_audio_dir, sub_dir='td', n_seconds=10)
        split_audio(files_dld, segmented_audio_dir, sub_dir='dld', n_seconds=10)
    else:
        print("The 'Segmented' folder already exists. Skipping the splitting process.")

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
    data_dir = 'C:/Users/c.pouw/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Screener'
    age_group = '4yo'
    dataset = main(data_dir, age_group)

    print(dataset[0])
    
    # Save to disk
    dataset.save_to_disk(f"finetuning_data/{age_group}_childes_auris_denoised")
