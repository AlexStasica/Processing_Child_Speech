# DANN Speech Classification for Child Language Disorders

## Overview

This repository contains an initial implementation of a **Domain-Adversarial Neural Network (DANN)** for classifying speech patterns in children with and without language disorders. 
The project aims to create a robust classifier that can work across different datasets (domains) by learning domain-invariant features using a very clean dataset (Auris) and a noisy one (CHILDES).

## Background

The main challenge in speech classification for clinical applications is domain shift - when models trained on one dataset (e.g., CHILDES) perform poorly on another dataset (e.g., clinical recordings from Auris). 
DANN addresses this by:

1. **Feature Extraction**: Learning representations from speech spectrograms
2. **Label Classification**: Distinguishing between typical development (TD) and language disorders (TOS/vvTOS)
3. **Domain Adaptation**: Using gradient reversal to make features domain-invariant

## Repository Structure

- `main.py` - Main DANN implementation 
- `simple_classifier.py` - Standard CNN+RNN classifier without domain adaptation (for comparison)
- Both scripts work with the same data structure and provide comparable results

## Key Features

### DANN Architecture
- **CNN Feature Extractor**: Processes mel-spectrograms with 2D convolutions
- **RNN Components**: Captures temporal patterns in speech
- **Gradient Reversal Layer**: Core DANN component that confuses the domain classifier
- **Dual Classification Heads**: 
  - Label classifier (TD vs TOS/vvTOS)
  - Domain classifier (CHILDES vs Clinical)

### Data Processing
- Converts audio files to mel-spectrograms (64 mel bins, 512 FFT)
- Handles variable-length audio (max 3 seconds)
- Normalizes features for stable training
- Supports multiple age groups (3yo, 4yo, 5yo, 6yo)

### Training Features
- Adversarial training with gradient reversal
- Dynamic lambda scheduling for domain adaptation strength
- Cross-entropy loss for both classification tasks
- Adam optimizer with learning rate scheduling
- Validation monitoring with accuracy metrics

## Data Structure Expected

```
Screener/
├── Processed_CHILDES/
│   ├── Childes_3yo/TD/*/*/*_chi_chins.wav
│   ├── Childes_3yo/TOS/*/*/*_chi_chins.wav
│   └── ... (other age groups)
├── 3yo/
│   ├── TD-3yo/Formatted/*/*/*_chi_chins.wav
│   ├── TOS-3yo/Formatted/*/*/*_chi_chins.wav
│   └── vvTOS-3yo/Formatted/*/*/*_chi_chins.wav
└── ... (other age groups: 4yo, 5yo, 6yo)
```

## Usage

### Requirements
```bash
pip install torch torchaudio numpy matplotlib scikit-learn
```

### Running the Script
```python
python dann_speech_classifier.py
```

**Important**: Update the `base_path` in the `main()` function to match your data location:
```python
audio_data = prepare_audio_data(base_path="YOUR_DATA_PATH_HERE")
```

### Expected Output
1. Data loading summary with file counts
2. Model architecture details
3. Training progress every 5 epochs
4. Validation accuracy for both tasks
5. Training history plots
6. Feature space visualizations
7. Saved model (`dann_speech_classifier.pth`)

## Model Performance Interpretation

### Key Metrics
- **Label Accuracy**: How well the model distinguishes TD from TOS/vvTOS
- **Domain Accuracy**: How well the model can identify the source dataset
  - **Lower domain accuracy is better** - indicates successful domain adaptation
  - High domain accuracy means the model is overfitting to dataset-specific features

### Visualizations
1. **Loss Curves**: Monitor convergence and overfitting
2. **Accuracy Plots**: Track performance on both tasks
3. **Feature Space Visualization**: PCA projection showing feature distribution
   - Ideally: Clear separation by speech type, mixed distribution by domain

## Known Limitations & Future Work

### Current Limitations
- **Initial Implementation**: This is a proof-of-concept, not production-ready
- **Limited Hyperparameter Tuning**: Default values may not be optimal
- **Small Dataset Warning**: Performance depends heavily on data availability
- **Class Imbalance**: No handling of unequal class distributions

### Suggested Improvements
1. **Hyperparameter Optimization**: Grid search for learning rates, lambda values, architecture sizes
2. **Data Augmentation**: Implement audio augmentation techniques
3. **Advanced Architectures**: Try transformer-based models or more sophisticated CNNs
4. **Cross-Validation**: Implement k-fold CV for more robust evaluation
5. **Class Balancing**: Add weighted losses or sampling strategies
6. **Feature Engineering**: Explore additional acoustic features beyond mel-spectrograms

## Technical Details

### Gradient Reversal Layer
The core DANN component that reverses gradients during backpropagation:
- Forward pass: Identity function
- Backward pass: Multiplies gradients by -λ
- Lambda (λ) increases during training: `λ = 2/(1 + exp(-10*p)) - 1`

### Model Architecture
- Input: Mel-spectrogram (64 x 188 time frames)
- CNN: 3 conv layers with increasing channels (32→64→128)
- Feature vector: 256-dimensional representation
- Label classifier: 256 → 128 → 2 classes
- Domain classifier: 256 → 128 → 2 domains

## Comparison with Baseline

The companion `baseline_classifier.py` script implements the same architecture **without** the domain adaptation components. 
Compare both models to evaluate DANN's effectiveness:

- Baseline: Standard supervised learning
- DANN: Domain-adversarial training

Expected outcome: DANN should show better generalization across domains (CHILDES ↔ Clinical).

## Contact & Handover Notes

This implementation serves as a starting point for domain adaptation in speech classification. 
The code is well-commented and modular for easy extension. Key areas for immediate attention:

1. **Data Path Configuration**: Ensure correct paths in `prepare_audio_data()`
2. **Computational Resources**: Training requires GPU for reasonable speed
3. **Baseline Comparison**: Always run both scripts to validate DANN benefits
4. **Result Documentation**: Save training plots and model checkpoints


## Contact

For questions, suggestions, or collaboration, please contact one of the developer:

Alex Elio Stasica (eliostasica@gmail.com)
