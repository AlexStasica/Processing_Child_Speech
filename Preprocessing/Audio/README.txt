# Audio Preprocessing

This folder contains scripts for preprocessing audio data used in the annotation and analysis of Dutch child speech. These scripts help standardize the audio files and improve their quality prior to annotation or further analysis.

## Contents

### Current Scripts

- **`convert_to_wav_and_mono.py`**  
  Converts audio files to `.wav` format and ensures they are single-channel (mono).  
  *Author: Xin Wan (X.Wan-2@student.tudelft.nl)*

- **`check_audio_quality.py`**  
  Performs basic checks on the audio files to assess their overall quality (e.g., silence, clipping, length consistency).

- **`get_wav_durations.py`**  
  Computes the duration (in seconds) of each `.wav` file in a directory, which can be useful for quality control and alignment.

---

### Upcoming Scripts

Scripts for denoising and enhancing audio—particularly for the CHILDES data—will be added soon. These scripts will help address:

- **Additive noise**
- **Low bandwidth artifacts**
- **Reverberation**

They are being developed by Jorge Martinez and will be added to this folder once shared.

---

## Purpose

The goal of this preprocessing pipeline is to standardize and enhance the quality of audio recordings from different sources (Auris, CHILDES) before annotation or machine learning-based analysis.

---

## Contact

Alex Elio Stasica (eliostasica@gmail.com)
Charlotte Pouw (c.m.pouw@uva.nl)

