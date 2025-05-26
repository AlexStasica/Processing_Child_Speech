# SLP Diarization System

## Overview

This project addresses a significant challenge in automatic speech recognition (ASR) for child speech diarization,
where current models often struggle with accuracy. Rather than tackling the broad problem of differentiating between all adults
and children, we take a targeted approach by fine-tuning the pyannote speaker diarization model to specifically recognize
Speech-Language Therapists (SLTs) in clinical recordings.
The system focuses on recordings containing one SLT and one child, with a finite set of known SLTs. By teaching the model
to identify specific SLTs' speech patterns, we can improve diarization in clinical environments where a therapist interacts
with a child. This approach circumvents the challenge of limited child speech training data by instead focusing on recognizing
the known SLT speakers.

## Goal

The primary goal is to train the model to recognize the characteristic features of SLT speech, allowing it to generalize
to other SLTs in similar clinical settings. This improved diarization will facilitate more accurate transcription of therapy
sessions and assessments, which is crucial for clinical documentation and research.
In future iterations, incorporating transcription data could further enhance the model's performance by allowing it to
use linguistic patterns (word choices and phrasing) typical of therapist speech.

## Technical Requirements

- Python Version: Python 3.9 is required (PyTorch compatibility issues exist with Python 3.12/3.13)
- Dependencies: All required libraries are imported in the script
- Pyannote Access: Users must accept the terms and conditions on Hugging Face and generate an access token
- Token Configuration: Insert your personal Hugging Face token in the code where indicated

## Data Requirements

- Audio recordings containing one SLT and one child
- TextGrid files with diarization (ADU vs CHI) for the training
- Excel mapping file connecting audio filenames to SLT identification codes

## How to Use

Setup
- Install Python 3.9 and all required dependencies
- Obtain a Hugging Face access token for pyannote
- Update the token in the code (use_auth_token="YOUR_TOKEN_HERE")
- Prepare your data directory structure and use the Excel mapping file (present in the repository)

## Running the System

The script offers two operational modes controlled by the TRAIN_MODE flag in the main section:

Training Mode (TRAIN_MODE = True)
Sets up the system to train on your dataset
Configure the following parameters:

ROOT_DIR: Path to your audio and TextGrid files
EXCEL_PATH: Path to your SLT mapping Excel file
TEST_SIZE: Proportion of data to use for testing (default: 0.2)

Outputs trained models to the "models" directory
Provides performance metrics (precision, recall, F1-score)

Inference Mode (TRAIN_MODE = False)
Applies the trained model to new recordings
Configure the following parameters:

AUDIO_FILE: Path to the new recording for analysis
OUTPUT_DIR: Directory where resulting TextGrids will be saved

Creates TextGrid files with speaker diarization results

## System Workflow
- Data Processing: Locates audio-TextGrid pairs and extracts speaker segments
- Embedding Extraction: Uses pyannote to extract voice embeddings for SLTs and children
- Model Training: Develops two classifiers:

Speaker type classifier (SLT vs. child)
- SLT identifier (distinguishes between different SLTs)
- Evaluation: Assesses model performance with standard metrics
- Inference: Applies the trained model to new recordings
- Output Generation: Creates TextGrid files with diarization results

## Current performance 

=== SLP vs Child Classifier ===
              precision    recall  f1-score   support

       Child       0.89      0.90      0.90      1504
         SLP       0.92      0.91      0.92      1850

    accuracy                           0.91      3354
   macro avg       0.91      0.91      0.91      3354
weighted avg       0.91      0.91      0.91      3354


=== SLP Identifier ===
              precision    recall  f1-score   support

           1       0.88      0.62      0.72       141
           2       0.84      0.75      0.79       417
           3       0.89      0.42      0.57       149
           4       0.92      0.72      0.81        79
           5       0.79      0.75      0.77       411
           6       0.65      0.85      0.74       654

    accuracy                           0.75      1851
   macro avg       0.83      0.68      0.73      1851
weighted avg       0.77      0.75      0.75      1851


## Future Improvements

- Add correct identifier for all the SLTs
- Use the DER as a metric for the model
- Integration with transcription data to use linguistic patterns for the classification
- Extension to multi-speaker scenarios
- Enhanced embedding techniques for better speaker separation

## Contact

Alex Elio Stasica (eliostasica@gmail.com)
