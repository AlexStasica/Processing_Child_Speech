'''
SCRIPT REQUIREMENTS:
  - pyannote.audio > "pip install pyannote.audio"
  - praat textgrid > "pip install textgrid"
  - A Huggingface account to accept pyannote's user conditions:
    - Accept https://hf.co/pyannote/segmentation-3.0 user conditions (click on link to do so)
    - Accept https://hf.co/pyannote/speaker-diarization-3.1 conditions (click on link to do so)
    - Create access token at https://hf.co/settings/tokens.
'''

from pyannote.audio import Pipeline
import wave
from textgrid import TextGrid, IntervalTier
import argparse
import glob
import torch

def pipeline(file):
    '''
    Calls the pyannote diarization pipeline to automatically detect speakers, turns, and timestamps
    ARGS:
      - file : path to the audio file (/!\ .wav FORMAT necessary /!\)
    '''

    if str(file).endswith(".wav") == False:
        print(file)
        print("Audio file is not in .wav format")
        return

    # Calling the pipeline and saving the result, /!\ IN THE "use_auth_token" argument, copy-paste your own Huggingface access token /!\
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="ADD PERSONAL ACCESS TOKEN HERE")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarization = pipeline(file)

    # Automatically getting the filename
    filename = str(file).split("/")[-1]

    # Calculation of the audio file's duration
    with wave.open(file, 'r') as audio_file:
        # Get the total number of frames in the file
        num_frames = audio_file.getnframes()
        # Get the sample rate (frames per second)
        sample_rate = audio_file.getframerate()
        # Calculate the duration in seconds
        duration = num_frames / float(sample_rate)

    tg = diarization_formatting_to_textgrid(str(diarization), filename, duration)

    return tg

def diarization_formatting_to_textgrid(input, filename, duration):
    '''
    Formatting the pipeline's results into a textgrid file, this is in a separated function because calling the pipeline requires
    a lot of computational capacity, and may take a long time, so you can call this function to reformat a pre-saved pyannote output

    Args:
     - input : output of the pipeline function
     - filename : name of the file, taken directly from the pipeline function
     - duration : duration of the file, taken directly from the pipeline function
    '''

    # Creating empty dict to store the data
    turns = {}

    # Going through each turn (in pyannote's output, one line = one turn)
    for line in input.split("\n"):

        elements = line.split()

        # For each turn, we first get the value of the speaker (the last element of the line)
        speaker = elements[-1]

        # When encountering a speaker for the first time: add it to the list
        if speaker not in turns.keys():
            turns[speaker] = []

        # The format of pyannote's output is "HH:MM:SS.MMMM", so we convert it all into seconds
        turn_start = hours_start, minutes_start, seconds_start = map(float, elements[1].split(":"))
        turn_start = round((hours_start * 3600 + minutes_start * 60 + seconds_start), 3)
        turn_end = hours_end, minutes_end, seconds_end = map(float, elements[3][:-1].split(":"))
        turn_end = round((hours_end * 3600 + minutes_end * 60 + seconds_end), 3)

        # We add this line's turn to the dict
        turns[speaker].append([turn_start, turn_end])

    # Creating the textgrid
    tg = TextGrid(minTime=0, maxTime=duration)

    # We go through all speakers
    for speaker in turns.keys():

        # We create a new tier for each speaker
        tier = IntervalTier(name=speaker, minTime=0, maxTime=duration)

        # We create a new interval for each turn (each value of the speaker's key)
        for interval in turns[speaker]:

            # Due to rounding?, the last turn's end time can be greater than the file's calculated duration
            if interval[1] > duration:
                tier.add(interval[0], duration, str(speaker))
            else:
                tier.add(interval[0], interval[1], str(speaker))

        # Once we created all the intervals for the speaker's tier, we append that tier to the textgrid
        tg.append(tier)

    return tg, filename

def main(data_dir, output_folder):
    for audio_file in glob.glob(data_dir + "/*.wav"):
        tg, filename = pipeline(audio_file)
        # Writing the result in a .TextGrid file
        tg.write(f"{output_folder}/{filename}_diarized.TextGrid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pyannote on audio files, return textgrids with diarizations")
    parser.add_argument('data_dir', type=str, help="Directory containing audio files")
    parser.add_argument('output_folder', type=str, help="Path to the folder where textgrids will be stored")
    args = parser.parse_args()
    main(args.data_dir, args.output_folder)