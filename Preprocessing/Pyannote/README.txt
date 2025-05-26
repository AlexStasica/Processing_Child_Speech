# PyAnnote-Based Preprocessing

This folder contains scripts used for preprocessing audio data using [PyAnnote](https://github.com/pyannote/pyannote-audio), specifically for speaker diarization and formatting. 
These scripts form part of the pipeline used to create annotated audio segments focused on child and/or adult speech.

## Content

### `pyannote_to_textgrid.py`
This script performs **speaker diarization** on raw `.wav` audio files using PyAnnote and outputs corresponding `.TextGrid` files.  
These initial TextGrids serve as the base for manual correction. Each speaker detected by PyAnnote is labeled (e.g., `SPEAKER00`, `SPEAKER01`).

### `cleaned_textgrid_to_formatted_wav.py`
This script processes manually corrected `.TextGrid` files and the original `.wav` audio to produce:
- A new `.TextGrid` that only includes segments annotated as relevant (e.g., `chi`, `adu`, etc.).
- A new `.wav` audio file that contains only the selected annotated segments (e.g., only child speech).
- Speech segments are separated by stretches of noise (e.g., background noise of ~2 seconds), based on the annotations.

### `change_filename.py`
Utility script to **standardize or correct filenames** of manually corrected `.TextGrid` files before feeding them into `cleaned_textgrid_to_formatted_wav.py`. 
This ensures consistent file naming across the pipeline.


## Contact

Alex Elio Stasica (eliostasica@gmail.com)
Charlotte Pouw (c.m.pouw@uva.nl)
Louis Berard (berardlouis@gmail.com)
