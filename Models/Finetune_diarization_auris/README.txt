### Two different approaches to fine tune speaker diarization in clinical settings

## 'main.py'
First test for speaker diarization 

##Running the System

The script offers two operational modes controlled by the TRAIN_MODE flag in the main section:

#Training Mode (TRAIN_MODE = True)

Sets up the system to train on your dataset
Configure the following parameters:

ROOT_DIR: Path to your audio and TextGrid files
EXCEL_PATH: Path to your SLT mapping Excel file
TEST_SIZE: Proportion of data to use for testing (default: 0.2)


Outputs trained models to the "models" directory
Provides performance metrics (precision, recall, F1-score)

#Inference Mode (TRAIN_MODE = False)

Applies the trained model to new recordings
Configure the following parameters:

AUDIO_FILE: Path to the new recording for analysis
OUTPUT_DIR: Directory where resulting TextGrids will be saved


Creates TextGrid files with speaker diarization results

#System Workflow

Data Processing: Locates audio-TextGrid pairs and extracts speaker segments
Embedding Extraction: Uses pyannote to extract voice embeddings for SLTs and children
Model Training: Develops two classifiers:

- Speaker type classifier (SLT vs. child)
- SLT identifier (distinguishes between different SLTs)


Evaluation: Assesses model performance with standard metrics
Inference: Applies the trained model to new recordings
Output Generation: Creates TextGrid files with diarization results

## 'main_v3.py'
Model Architecture
Base Models

Diarization Pipeline: pyannote/speaker-diarization-3.1
Embedding Model: pyannote/wespeaker-voxceleb-resnet34-LM (ResNet-34 based speaker embeddings)

#Fine-tuning Approach
The system employs a transfer learning approach with the following components:

Selective Parameter Unfreezing: Only the final layers (FC, linear, head layers) and last ResNet blocks (layer3, layer4) are fine-tuned
Custom Loss Function: Contrastive loss designed for clinical speaker identification
Clinical-Aware Training: Specialized for 2-speaker scenarios (child + SLP)

#Loss Function
Custom ClinicalSpeakerLoss combining:

Contrastive Loss: Pulls similar speakers together, pushes different speakers apart
SLP Identification Loss: Specialized loss for distinguishing between different SLPs
Weighted Combination: Balances speaker type classification vs. SLP identification

#Input Data Requirements
Audio Files

Format: WAV files
Naming Convention: *audio_chi_chins_adu.wav
Sample Rate: 16 kHz (automatically resampled if different)
Duration: Variable (processed in 3-second chunks during training)

Annotation Files

Format: Praat TextGrid files (.TextGrid)
Naming Convention: wav_textgrid_*.TextGrid (corresponding to audio files)
Tiers:

Child tier (containing 'child' or 'chi' in name)
Adult/SLP tier (containing 'adult' or 'adu' in name)

#Training Process
Cross-Validation Setup

5-fold cross-validation based on SLP grouping
Ensures no SLP appears in both training and test sets
Maintains speaker identity integrity across folds

#Training Parameters

Batch Size: 16
Learning Rate: 1e-4
Epochs: 10
Chunk Duration: 3.0 seconds
Optimizer: Adam with weight decay (1e-5)
Gradient Clipping: Max norm = 1.0

#Data Augmentation

Chunk Segmentation: Long audio segments split into 3-second chunks
Zero Padding: Short segments padded to maintain consistent input size
Speaker Balancing: Equal representation of child and SLP segments

#Output and Results
Performance Metrics

Diarization Error Rate (DER): Primary metric for speaker diarization quality
Speaker Mapping Accuracy: Measures correct speaker role assignment
SLP Recognition Accuracy: Specific metric for SLP identification

#Key Features
Clinical Specialization

2-Speaker Focus: Optimized for child-therapist interactions
SLP Recognition: Distinguishes between different Speech-Language Pathologists
Age Group Support: Handles recordings from children aged 3-6 years
Robust Preprocessing: Handles various audio qualities and durations

#Evaluation Framework

Cross-Validation: Rigorous evaluation with SLP-based fold splitting
Multiple Metrics: Comprehensive performance assessment
Statistical Analysis: Mean and standard deviation reporting
Failure Tracking: Monitors processing errors and success rates

#Technical Robustness

Memory Efficient: Processes audio in manageable chunks
Error Handling: Comprehensive exception handling throughout pipeline
Flexible Input: Adapts to various audio formats and qualities
Reproducible: Fixed random seeds for consistent results
