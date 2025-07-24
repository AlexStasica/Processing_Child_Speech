# MetricGAN+ Fine-tuning for CHILDES Speech Enhancement

A PyTorch implementation for fine-tuning the pre-trained MetricGAN+ model on custom speech enhancement datasets, 
specifically designed for CHILDES-matched training data.

## Features

- Fine-tune pre-trained MetricGAN+ model from SpeechBrain
- Custom dataset loader for clean/noisy speech pairs
- Automatic model saving with early stopping
- Training/validation loss visualization
- Spectrogram-based processing with STFT/ISTFT reconstruction


## Parameters

- `batch_size`: Training batch size (default: 8)
- `num_epochs`: Maximum training epochs (default: 10)
- `device`: 'cpu' or 'cuda' for GPU acceleration
- `max_length`: Maximum audio length in samples (default: 64000)
- `patience`: Early stopping patience (default: 5)

## Output

- **Model checkpoints**: Saved as `best_model_epoch_X.pth`
- **Loss plots**: Training and validation loss visualization
- **Console logs**: Real-time training progress and metrics

## Key Components

### SpeechEnhancementDataset
- Loads clean/noisy audio pairs
- Handles audio preprocessing and padding
- Supports .wav format audio files

### MetricGANFineTuner
- Loads pre-trained MetricGAN+ from SpeechBrain
- Implements spectrogram-based training loop
- Includes validation and early stopping
- Automatic model saving and loss visualization

## Model Architecture

The script fine-tunes the pre-trained MetricGAN+ model using:
- STFT with 512 FFT size, 256 hop length
- Spectrogram magnitude processing
- Phase reconstruction for audio generation
- MSE loss for training objective

##Inference and Evaluation
The inference_evaluation.py script provides comprehensive evaluation and comparison capabilities
for both pre-trained and fine-tuned models.


# Run evaluation
python inference_evaluation.py
Evaluation Metrics
The script calculates multiple audio quality metrics:

SNR Improvement: Signal-to-noise ratio enhancement
PESQ Score: Perceptual evaluation of speech quality (simplified)
STOI Score: Short-time objective intelligibility (simplified)
Spectral Distortion: Frequency domain distortion measure
MSE: Mean squared error between clean and enhanced audio
Correlation: Temporal correlation coefficient
Segmental SNR: Frame-wise SNR analysis

##Key Components
#ModelInference

Loads pre-trained or fine-tuned models
Handles audio preprocessing and chunking for large files
Memory-optimized inference with automatic cleanup
Supports both CPU and GPU processing

#SpeechEnhancementEvaluator

Comprehensive metric calculation
Batch evaluation of test sets
Model comparison with improvement analysis
Automatic result visualization and CSV export

#Output Files

Enhanced Audio: ./enhanced_output/pretrained/ and ./enhanced_output/finetuned/
Evaluation Results: pretrained_results.csv, finetuned_results.csv
Model Comparison: model_comparison.csv
Visualization: model_comparison_plots.png

#Features

Memory-efficient processing for large datasets
Automatic audio format handling (mono conversion, resampling)
Chunk-based processing for long audio files
Comprehensive error handling and logging
Statistical analysis and visualization of results

##Contact

Alex Elio Stasica (eliostasica@gmail.com)
