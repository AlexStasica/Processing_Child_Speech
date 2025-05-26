import os
import pandas as pd
from pydub import AudioSegment


def get_audio_duration_in_minutes(file_path):
    sound = AudioSegment.from_file(file_path)
    #duration calculation function
    sound.duration_seconds == (len(sound) / 1000.0)
    #seconds to minutes conversion
    minutes_duartion = int(sound.duration_seconds // 60)
    seconds_duration = round((sound.duration_seconds % 60),3)
    duration = f"{minutes_duartion}:{seconds_duration}"
    return duration


def get_audio_files_with_durations(folder_path):
    audio_data = []  # List to store file names and durations
    # Supported audio formats
    audio_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    # Loop through files and gather durations for valid audio formats
    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in audio_formats):
            file_path = os.path.join(folder_path, file_name)
            duration = get_audio_duration_in_minutes(file_path)
            audio_data.append((file_name, duration))
    return audio_data


def save_to_excel(audio_data, output_file):
    # Convert the data to a DataFrame
    df = pd.DataFrame(audio_data, columns=['File Name', 'Duration (minutes)'])
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)


def main():
    folder_path = ''
    output_folder = ''

    if os.path.isdir(folder_path):
        audio_data = get_audio_files_with_durations(folder_path)
        folder_name = os.path.basename(folder_path)
        output_file = os.path.join(output_folder, f"{folder_name}_audio_durations.xlsx")
        save_to_excel(audio_data, output_file)
        print(f"Excel file saved: {output_file}")
    else:
        print("The provided path is not a valid folder.")


if __name__ == '__main__':
    main()
