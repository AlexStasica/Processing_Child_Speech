from textgrid import TextGrid, IntervalTier
import re
import random
from pydub import AudioSegment
import time
import os
from openpyxl import Workbook

"""
TO DO:
    - Add support for specific annotations: several noises? | chi_un | adu_un
    - Find a way to get all segment with a potential typo in the annotation and return it for manual verification
    - Add support for several children or adults
"""


AUDIO_FOLDER = "Audio"
TEXTGRID_FOLDER = "TextGrid"
OUTPUT_FOLDER = "Formatted"
#/!\Make sure to follow the naming convention: /!\
#   Audio files: xxxx.wav
#   Textfiles:   xxxx_diarized_clean.TextGrid

segment_options = {
    "chi" : True,
    "chi_ns" : False,
    "adu" : True
}
#Options : which kind of segments to keep in the final audio

def processing(textgrid_file,audio_file,segment_options):
    
    """
    This function takes the cleaned textgrid files, finds all the intervals of child speech and creates a new textgrid with only those intervals, 
    cuts the audio file to keep only the child speech (with silence in between utterances), and synchronizes the new textgrid and the new audio.

    Args:
        - textgrid_file : Path to the cleaned TextGrid file
        - audio_file : Path to the corresponding audio file in a .wav
        -segment_options : Dict listing the different kind of segment to add to the output audio
    """

    #Error messages
    if audio_file.endswith(".wav") is False:
        return None, None, None, None, f"Error: {audio_file} is not in .wav format !"
    if textgrid_file.endswith(".TextGrid") is False:
        return None, None, None, None, f"Error: {textgrid_file} is not in .TextGrid format !"
    
    print(f"Processing {audio_file}")
    start_time = time.time()

    list_intervals = []
    durations_dict = {}
    noise_interval = None
    if segment_options["chi"] == True:
        durations_dict["chi"] = 0
    if segment_options["chi_ns"] == True:
        durations_dict["chins"] = 0
    if segment_options["adu"] == True:
        durations_dict["adu"] = 0

    #Read the textgrid to extract all child speech intervals
    try:
        with open(textgrid_file,"r",encoding="utf-8") as input_textgrid: #Depending on the file, may need to read with "encoding='utf-16'"
            lines = input_textgrid.readlines()
    except:
        with open(textgrid_file,"r",encoding="utf-16") as input_textgrid:
            lines = input_textgrid.readlines()

    for x,line in enumerate(lines):

        if re.match('.*text',line):
            interval_start = float(lines[x-2].split()[-1])
            interval_end = float(lines[x-1].split()[-1])
        else:
            continue

        if segment_options["chi_ns"] is True:
            if re.match('.*text = ".*[Cc][Hh][Ii]_[Nn][Ss].*"',line):
                list_intervals.append([interval_start,interval_end,'CHI_NS'])
                durations_dict["chins"] += interval_end - interval_start
                continue

        if segment_options["chi"] is True:
            if re.match('.*text = ".*[Cc][Hh][Ii].*"',line):
                list_intervals.append([interval_start,interval_end,'CHI'])
                durations_dict["chi"] += interval_end - interval_start
                continue

        if segment_options["adu"] is True:
            if re.match('.*text = ".*[Aa][Dd][Uu].*"',line):
                list_intervals.append([interval_start,interval_end,'ADU'])
                durations_dict["adu"] += interval_end - interval_start
                continue

        if re.match(".*noise",line):
            noise_interval = [interval_start,interval_end]
            continue

    if noise_interval == None or noise_interval[1]-noise_interval[0] < 1.5:
         return None,None,None,None, f"Error: 'Noise' segment not found or less than 1.5sec"
    
    #Open the audio file
    audio = AudioSegment.from_file(audio_file)
    duration_of_audio = audio.duration_seconds
    noise_audio = audio[noise_interval[0]*1000:noise_interval[1]*1000]

    """
    Concatenating adjacent intervals
    """
    #Sort all the intervals, as they are not all in the same tier, they are not ordered when reading the textgrid line by line
    list_intervals.sort(key=lambda x: x[0])
    concatenated_intervals = [list_intervals[0]]  # Start with the first interval
    
    for current in list_intervals[1:]:
        previous = concatenated_intervals[-1]
        # Check if intervals touch
        if current[0] == previous[1] and current[2] == previous[2]:  # If they touch AND they are in the same tier
            concatenated_intervals[-1] = [previous[0], current[1],current[2]]  # Merge intervals
        else:
            concatenated_intervals.append(current)  # No touching: add to list without modification

    """
    Concatenating all intervals with noise in between
    +
    Creating the processed audio file
    """
    final_intervals =[concatenated_intervals[0]] #Start with the first interval
    new_audio = audio[concatenated_intervals[0][0]*1000:concatenated_intervals[0][1]*1000] #Creating the audio output, starting with the first interval


    for current in concatenated_intervals[1:]:

        previous = final_intervals[concatenated_intervals.index(current)-1]

        duration = current[1] - current[0] #Calculating the duration of the interval
        wait_time = random.uniform(0.2,1.5) #Defining a random wait time

        if current[0] != previous[1]: #If the current interval is not touching the previous one
            interval_start = previous[1] + wait_time #The interval starts a random amount of time after the end of the previous one
            new_audio += noise_audio[0:wait_time*1000] #Adding the noise in the audio before the new interval
        
        else: #If the two intervals touch, we do not add noise
            interval_start = previous[1]


        interval_end = interval_start + duration
        new_audio += audio[current[0]*1000:current[1]*1000] #adding the interval in the audio

        final_intervals.append([round(interval_start,3),round(interval_end,3),current[2]]) #Adding the interval to the list
    
    
    """
    Exporting the processed textgrid
    """
    #Creating a tier for each option
    tg = TextGrid()
    if segment_options["chi"] == True:
        chi_tier = IntervalTier(name="Child Speech",minTime=0,maxTime=final_intervals[-1][1])
    if segment_options["chi_ns"] == True:
        chi_ns_tier = IntervalTier(name="Child Non Speech",minTime=0,maxTime=final_intervals[-1][1])
    if segment_options["adu"] == True :
        adu_tier = IntervalTier(name="Adult Speech",minTime=0,maxTime=final_intervals[-1][1])

    offset = final_intervals[0][0] #As the first interval doesn't start a 0.00sec, but the audio does, we offset all intervals by
                                   #the start time of the first one, so it all starts at 0.

    #For each interval, add it to the relevant tier, with the offset
    for interval in final_intervals:
        if interval[2] == 'CHI' and segment_options["chi"] == True:
            chi_tier.add(interval[0]-offset,interval[1]-offset,"CHI")
        if interval[2] == 'CHI_NS' and segment_options["chi_ns"] == True:
            chi_ns_tier.add(interval[0]-offset,interval[1]-offset,"CHI_NS")
        if interval[2] == 'ADU' and segment_options["adu"] == True :
            adu_tier.add(interval[0]-offset,interval[1]-offset,"ADU")

    #Adding the tiers to the textgrid
    if segment_options["chi"] == True:
        tg.append(chi_tier)
    if segment_options["chi_ns"] == True:
        tg.append(chi_ns_tier)
    if segment_options["adu"] == True:
        tg.append(adu_tier)
    time_taken = time.time() - start_time
    
    print(f"Done processing {audio_file} in {time_taken:.3f} seconds !\n")
    return new_audio, tg, duration_of_audio, durations_dict, None


def input(audio_folder, textgrid_folder, output_folder, segment_options):
    """
    This function goes through all the files in the folders, call the processing function to get the new textgrid time stamps
    and the new audio, then export those into a new textgrid file and a new wav file, it also creates a spreadsheet to contain
    duration info for all the files
    """

    #Creating the spreadsheet and its headers
    workbook = Workbook()
    sheet = workbook.active
    row1 = ["File:","Duration of file"]
    for key in segment_options:
        if segment_options[key] == True:
            row1.append(key)
    sheet.append(row1)

    #For each of the audio files in the folders
    for file_name in os.listdir(audio_folder):
        
        #get the audio file and the corresponding textgrid file
        audio_file_path = os.path.join(audio_folder, file_name)
        text_grid_file_path = os.path.join(textgrid_folder, f'{file_name[:-4]}_diarized_clean.TextGrid')
        
        #run the processing function to get the new audio, new textgrids, the durations
        if os.path.isfile(audio_file_path) and os.path.isfile(text_grid_file_path):
            audio_output, textgrid_output, total_duration, duration_dict ,error = processing(text_grid_file_path,audio_file_path,segment_options)
            if error != None:
                print(error)
                continue
        else:
            print(f"{file_name} was not processed : It is a wav file with a corresponding textgrid ?")
            continue
        
        #Create the output subfolder for the audio file
        if os.path.isdir(f'{output_folder}/{file_name[:-4]}') is False:
            os.mkdir(f"{output_folder}/{file_name[:-4]}")

        #Export the new audio and the new textgrid
        audio_output.export(f"{output_folder}/{file_name[:-4]}/{file_name[:-4]}_audio_{'_'.join(duration_dict.keys())}.wav",format="wav") #Export the finalized audio
        textgrid_output.write(f"{output_folder}/{file_name[:-4]}/{file_name[:-4]}_wav_textgrid_{'_'.join(duration_dict.keys())}.TextGrid")

        #Save the duration information of the file into the spreadsheet
        row = [file_name[:-4] , total_duration]
        for key in duration_dict:
            formatted_seconds = str(duration_dict[key]).replace(".",",")
            row.append(formatted_seconds)

        sheet.append(row)
    
    #Export the spreadsheet
    folder_name = str(audio_folder).split("/")[-2]
    workbook.save(f'{output_folder}/{folder_name}durations.xlsx')
    print("Spreadsheet created.")

input(AUDIO_FOLDER,TEXTGRID_FOLDER,OUTPUT_FOLDER,segment_options)
