from datasets import Dataset, Audio
import glob
import os
from pydub import AudioSegment
import noisereduce as nr
import statistics
from textgrid import TextGrid


def get_adjusted_boundaries(textgrid_file, split_duration=10.0):
    """
    Reads a TextGrid file and returns a list of adjusted segment boundaries
    that avoid splitting in the middle of intervals with 'CHI' in the 'Child Speech' tier.
    Ensures each chunk is at least 10 seconds long.
    """
    tg = TextGrid()
    tg.read(textgrid_file)

    # Find the 'Child Speech' tier
    child_speech_tier = next((tier for tier in tg.tiers if tier.name == "Child Speech"), None)

    if not child_speech_tier:
        print(f"No 'Child Speech' tier found in {textgrid_file}")
        return []

    boundaries = []
    start_time = 0.0
    total_duration = child_speech_tier.maxTime

    while start_time < total_duration:
        # Calculate the default end time (10 seconds after start_time)
        end_time = start_time + split_duration

        # Ensure the end_time does not exceed the total duration
        if end_time > total_duration:
            end_time = total_duration

        # Check if the end_time falls within a 'CHI' interval
        for interval in child_speech_tier.intervals:
            if interval.mark == "CHI" and interval.minTime <= end_time <= interval.maxTime:
                # Extend the boundary to the end of the 'CHI' interval
                end_time = interval.maxTime
                break

        # Ensure the chunk is at least 10 seconds long
        if end_time - start_time < split_duration:
            # Skip this chunk and move to the next 10-second boundary
            start_time = end_time
            continue

        # Add the adjusted boundary
        boundaries.append((start_time, end_time))

        # Move to the next chunk
        start_time = end_time

    print(f"Adjusted boundaries for {textgrid_file}: {boundaries}")
    return boundaries


def split_audio(file_list, segmented_audio_dir, sub_dir, n_seconds=10):
    split_duration = n_seconds  # in seconds
    segment_lengths = []

    for file in file_list:
        textgrid_file = file.replace(".wav", ".TextGrid")

        if not os.path.exists(textgrid_file):
            print(f"No TextGrid file found for {file}, skipping...")
            continue

        audio = AudioSegment.from_wav(file)
        boundaries = get_adjusted_boundaries(textgrid_file, split_duration)

        if not boundaries:
            print(f"No boundaries found for {file}, skipping...")
            continue

        # Split and export the audio
        for i, (start, end) in enumerate(boundaries):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # Extract the segment
            segment = audio[start_ms:end_ms]
            segment_length = (end - start)
            segment_lengths.append(segment_length)

            # Create new filename
            new_filename = f'{os.path.splitext(os.path.basename(file))[0]}_part_{i + 1}.wav'
            output_path = os.path.join(segmented_audio_dir, sub_dir, new_filename)

            # Save the segment to a new file
            segment.export(output_path, format="wav")
            print(f"Exported segment {i + 1} for {file} to {output_path}")

    # Compute mean and median segment lengths
    if segment_lengths:
        mean_length = statistics.mean(segment_lengths)
        median_length = statistics.median(segment_lengths)
        print(f"Mean segment length: {mean_length:.2f} sec")
        print(f"Median segment length: {median_length:.2f} sec")
    else:
        print("No segments were created.")


def main(data_dir, age_group):

    # Check if the 'segmented_audio' directory exists
    segmented_audio_dir = f"{data_dir}/MODEL/Segmented_test_4yo_chi"
    if not os.path.exists(segmented_audio_dir):

        # Folder doesn't exist, so create the folder + split the audio files
        print(f"The {segmented_audio_dir} folder does not exist yet. Splitting files...")
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
            files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/*/*chi.wav')
            # Combine data from Childes and Auris (only for TD)
            files_td = files_td_auris + files_td_childes

        elif age_group == '4yo':
            files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TD/*/*chi.wav')
            files_dld_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TOS/*/*chi.wav')
            # Combine data from Childes and Auris
            files_td = files_td_auris + files_td_childes
            files_dld = files_dld_auris + files_dld_childes
            print("TD CHILDES:", len(files_td_childes), "DLD CHILDES:", len(files_dld_childes), "TD AURIS:", len(files_td_auris), "DLD AURIS:", len(files_dld_auris))

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
    data_dir = ''
    age_group = '4yo'
    dataset = main(data_dir, age_group)

    print(dataset[0])
    
    # Save to disk
    dataset.save_to_disk(f"finetuning_data/{age_group}_childes_auris")
