import os
import glob


def rename_files(data_dir, age_group):
    # Load the audio files
    files_td_auris = glob.glob(f'{data_dir}/{age_group}/TD*/Formatted/*/*chi.wav')
    files_tos_auris = glob.glob(f'{data_dir}/{age_group}/TOS*/Formatted/*/*chi.wav')
    files_vvtos_auris = glob.glob(f'{data_dir}/{age_group}/vvTOS*/Formatted/*/*chi.wav')
    files_auris = files_tos_auris + files_vvtos_auris + files_td_auris

    files_td_auris_txt = glob.glob(f'{data_dir}/{age_group}/TD*/Formatted/*/*chi.TextGrid')
    files_tos_auris_txt = glob.glob(f'{data_dir}/{age_group}/TOS*/Formatted/*/*chi.TextGrid')
    files_vvtos_auris_txt = glob.glob(f'{data_dir}/{age_group}/vvTOS*/Formatted/*/*chi.TextGrid')
    files_auris_txt = files_tos_auris_txt + files_vvtos_auris_txt + files_td_auris_txt

    if age_group == '3yo':
        files_td_3_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/*/*chi.wav')
        files_td_3_childes_txt = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/*/*chi.TextGrid')

    elif age_group == '4yo':
        files_td_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TD/*/*chi.wav')
        files_dld_childes = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TOS/*/*chi.wav')

        files_td_childes_txt = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TD/*/*chi.TextGrid')
        files_dld_childes_txt = glob.glob(f'{data_dir}/Processed_CHILDES/Childes_{age_group}/TOS/*/*chi.TextGrid')

    if age_group == '3yo':
        all_files = files_auris + files_auris_txt + files_td_3_childes + files_td_3_childes_txt

    else:
        all_files = files_td_childes + files_dld_childes + files_td_childes_txt + files_dld_childes_txt

    # Iterate over the files to rename them
    for file in all_files:
        # Rename the .wav files by removing '_audio'
        if file.endswith('.wav'):
            new_name = file.replace('_audio', '')
            os.rename(file, new_name)
            print(f'Renamed {file} to {new_name}')

        # Rename the .textGrid files by removing '_wav_textgrid'
        if file.endswith('.TextGrid'):
            new_name = file.replace('_wav_textgrid', '')
            os.rename(file, new_name)
            print(f'Renamed {file} to {new_name}')


rename_files('C:/Users/a.stasica/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener', '4yo')
