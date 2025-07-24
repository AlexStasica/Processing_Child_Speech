### Objective

## 'create_auris_noise_dataset.py'
CHILDES Noise Simulation and Dataset Creation
A first step to fine tune the MetricGan model to learn how to remove noise
present on the CHILDES data and learn what the clean gold standard is (Auris clinical data)
Here, we create synthetic noisy datasets that match authentic recording conditions for speech enhancement research.

#Overview
This project provides tools to:

Analyze acoustic characteristics of real CHILDES recordings
Extract noise profiles from naturalistic child-adult interactions
Simulate realistic recording conditions on clean speech
Generate matched training datasets for speech enhancement models

#Key Features
Acoustic Analysis

Reverberation Analysis: RT60 estimation using energy decay methods
Noise Characterization: Multi-dimensional noise profiling
Spectral Analysis: MFCC, spectral centroid, rolloff, and flatness
Signal Quality: SNR estimation and temporal features
Memory Optimization: Efficient processing of large audio corpora

Noise Simulation

Physics-Based Degradation: Realistic room acoustics and reverberation
Spectral Shaping: Frequency-dependent noise characteristics
Dynamic Processing: Compression and AGC simulation
Recording Artifacts: Electrical interference and microphone effects
Intensity Scaling: Variable degradation levels

##Architecture
Core Components
#1. CHILDESNoiseAnalyzer
Purpose: Extracts acoustic characteristics from real CHILDES recordings
Key Methods:

analyze_childes_noise(): Main analysis pipeline
_estimate_rt60_improved(): Reverberation time estimation
_estimate_noise_level_improved(): Background noise quantification
_extract_spectral_profile(): Frequency domain analysis

Features:

Memory-efficient processing of large files
Robust error handling for corrupted audio
Statistical aggregation across multiple recordings

#2. CHILDESNoiseSimulator
Purpose: Applies realistic degradations to clean speech
Degradation Pipeline:

Reverberation: Room impulse response simulation
Spectral Coloration: Microphone/preamp response
Recording Artifacts: Electrical noise (60Hz hum, etc.)
Background Noise: Shaped noise addition with target SNR
Dynamic Compression: AGC and limiting effects
Frequency Response: EQ-like variations

#Parameters:

Intensity factor: Controls degradation severity (0.1-3.0)
Target SNR: Signal-to-noise ratio range (0-30 dB)
RT60 range: Reverberation time (0.1-2.0 seconds)

#Input Requirements
CHILDES Audio Files

Format: WAV files (any sample rate, auto-resampled to 16kHz)
Structure: Nested directories with audio files
Naming: Files ending with _reduced_noise.wav are automatically excluded
Size Limits: Files >100MB or >300 seconds are skipped
Quality: Minimum 1-second duration, basic silence detection

Clean Speech Files

Format: WAV files
Quality: Any quality (will be normalized)
Organization: Can be in nested folder structure
Naming: Files starting with 'P' treated as clean speech

##Advanced Usage
python# Manual noise analysis and simulation
from dataset_creator import CHILDESNoiseAnalyzer, CHILDESNoiseSimulator

# Analyze noise characteristics
analyzer = CHILDESNoiseAnalyzer()
noise_profile = analyzer.analyze_childes_noise("childes_folder", n_samples=2000)

# Create simulator
simulator = CHILDESNoiseSimulator(noise_profile)

# Apply degradation with custom intensity
import torchaudio
clean_audio, sr = torchaudio.load("clean_speech.wav")
degraded_audio = simulator.simulate_childes_conditions(clean_audio, intensity_factor=1.5)

# Generate multiple versions
versions = simulator.generate_multiple_versions(clean_audio, n_versions=5)

Analysis Parameters
pythonanalyzer_config = {
    'sample_rate': 16000,
    'max_file_size_mb': 100,      # Skip files larger than this
    'max_duration_seconds': 300,   # Skip files longer than this
}
Simulation Parameters
pythonsimulator_config = {
    'target_rms': 0.03,           # Target RMS level
    'max_amplitude': 0.8,         # Prevent clipping
    'min_snr': 0.0,              # Minimum SNR in dB
    'max_snr': 25.0,             # Maximum SNR in dB
}

#Quality Control Features
Robust Error Handling

File Validation: Size, duration, and format checks
Audio Validation: NaN/Inf detection, silence detection
Memory Management: Chunked processing, garbage collection
Graceful Degradation: Continues processing despite individual file errors

#Processing Statistics

Files successfully processed vs. failed
Detailed error logging
Progress tracking with tqdm
Memory usage optimization

#Audio Quality Assurance

RMS normalization for consistent levels
Clipping prevention
SNR validation
Dynamic range preservation

#Technical Details
Memory Optimization

Streaming Processing: Files processed in chunks
Efficient Resampling: Minimal memory footprint
Garbage Collection: Explicit cleanup of large objects
Progress Monitoring: Real-time memory usage tracking

#Signal Processing

RT60 Estimation: Energy decay curve fitting
Noise Floor Detection: Percentile-based estimation
Spectral Shaping: FIR filter design from spectral profiles
Room Simulation: Physics-based impulse response generation

#Validation and Testing

Statistical Validation: Mean, std, median reporting
Audio Integrity: NaN/Inf checking at each stage
File System Safety: Robust path handling
Cross-platform: Windows/Linux/Mac compatibility

## Contact

Alex Elio Stasica (eliostasica@gmail.com)
