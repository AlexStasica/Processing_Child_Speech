# Screener: Classification and Regression Models for Dutch Child Speech

This folder contains all scripts used for training, evaluating, and benchmarking machine learning models designed to screen for developmental language disorders (DLD) and estimate age from child speech. 
  The models are based on **Whisper**, **Wav2Vec2**, and a **baseline MFCC-based classifier**. 
The goal is to support the development of an interpretable and accurate speech screener for Dutch-speaking children.

---

## Folder Structure

### `age_classification/`

This folder contains scripts to **fine-tune OpenAI's Whisper** model for **multi-class age classification**. Children are divided into **6-month age bins**, 
and the model is trained to predict the correct bin from raw audio.

- `finetune_age_whisper.py`: Fine-tunes Whisper on the age classification task.
- `inference_age_whisper.py`: Performs inference on test data to predict the age bin.

---

### `binary_classification/`

Scripts for **binary classification** to distinguish between:
- **TD**: Typically Developing children
- **DLD**: Children with Developmental Language Disorder

Includes models based on:
- **Whisper**
- **Whisper + LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Wav2Vec2**

Each model has:
- A fine-tuning script (e.g., `finetune_whisper.py`, `finetune_wav2vec.py`)
- A matching inference script for evaluation
- A baseline model using **MFCC** features as comparison

---

### `regression_classification/`

This folder contains scripts for **age regression** using Whisper:
- **Task**: Predict the **exact age** (year and month) of the child from speech.
- **Training Data**: Only **typically developing children**.
- **Objective**: Check whether children with DLD are consistently predicted to be younger than their actual age (hypothesis: their speech resembles that of younger children).

---

### `dataset_creation/`

This folder includes all scripts used to create Hugging Face–style datasets for training the models in the other folders. 
Each script is tailored to the specific task (e.g., classification vs regression, Whisper vs Wav2Vec2 format, etc.).

Scripts ensure:
- Consistent preprocessing across datasets
- Proper formatting for tokenization, audio loading, and metadata association
- Easy integration with Hugging Face `datasets` and `transformers` libraries

---

## Notes

- All models are trained on **Dutch child speech data**, including children with and without DLD.
- Audio files have been manually annotated for speaker diarization and cleaned using Pyannote-based pipelines.
- Whisper and Wav2Vec2 are used as **end-to-end models** using raw audio for robust performance across tasks.
- LoRA is used to reduce computational cost and memory footprint in some fine-tuning scenarios.
- Baseline comparisons using MFCC features are included to assess the benefits of deep learning models.

---

## Dependencies

Make sure to install the following libraries:
- `transformers`
- `datasets`
- `torchaudio`
- `scikit-learn`
- `peft` (for LoRA)
- `evaluate`
- `numpy`
- `pandas`
- `huggingface_hub`

---

## Project Goal

The broader aim of this folder and its scripts is to contribute to a fully interpretable, automatic screener for speech and language disorders, which can aid early diagnosis and intervention.

## Contact

Alex Elio Stasica (eliostasica@gmail.com)
Charlotte Pouw (c.m.pouw@uva.nl)
