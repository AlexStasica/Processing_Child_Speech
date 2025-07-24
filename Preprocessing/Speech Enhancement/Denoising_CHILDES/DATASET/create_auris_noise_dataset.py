import os
import random
import numpy as np
import torch
import torchaudio
from scipy import signal
from sklearn.model_selection import train_test_split
import json
from typing import Dict, List
import librosa
import gc
import os
import shutil
import pandas as pd
from tqdm import tqdm


class CHILDESNoiseAnalyzer:
    """Analyze CHILDES dataset to extract noise characteristics"""

    def __init__(self, sample_rate=16000, max_file_size_mb=100, max_duration_seconds=300):
        self.sample_rate = sample_rate
        self.max_file_size_mb = max_file_size_mb
        self.max_duration_seconds = max_duration_seconds
        self.noise_profile = {}
        torch.set_num_threads(1)

    def get_file_info(self, filepath):

        try:
            # Check file size first
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return {
                    'valid': False,
                    'reason': f'File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB',
                    'size_mb': file_size_mb,
                    'duration': None
                }

            # Get duration without loading full audio
            try:
                duration = librosa.get_duration(path=filepath)
            except:
                # Fallback to torchaudio
                info = torchaudio.info(filepath)
                duration = info.num_frames / info.sample_rate

            if duration > self.max_duration_seconds:
                return {
                    'valid': False,
                    'reason': f'Duration too long: {duration:.1f}s > {self.max_duration_seconds}s',
                    'size_mb': file_size_mb,
                    'duration': duration
                }

            # Additional checks for minimum duration
            if duration < 1.0:
                return {
                    'valid': False,
                    'reason': f'Duration too short: {duration:.1f}s < 1.0s',
                    'size_mb': file_size_mb,
                    'duration': duration
                }
            return {
                'valid': True,
                'reason': 'Valid file',
                'size_mb': file_size_mb,
                'duration': duration
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f'Error reading file: {str(e)}',
                'size_mb': None,
                'duration': None
            }

    def find_all_wav_files(self, root_folder):

        wav_files = []

        print(f"Searching for wav files in: {root_folder}")

        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith('.wav') and not file.endswith('_reduced_noise.wav'):
                    full_path = os.path.join(root, file)
                    wav_files.append(full_path)

        print(f"Found {len(wav_files)} wav files")
        return wav_files

    def find_audio_folders(self, root_folder):
        wav_files = []

        print(f"Searching for 'Audio' folders in: {root_folder}")

        for root, dirs, files in os.walk(root_folder):
            # Check if current folder is named 'Audio' (case insensitive)
            if os.path.basename(root).lower() == 'audio':
                print(f"Found Audio folder: {root}")
                for file in files:
                    if file.lower().endswith(('.wav')):
                        full_path = os.path.join(root, file)
                        wav_files.append(full_path)

        print(f"Found {len(wav_files)} wav files in Audio folders")
        return wav_files

    def select_valid_files(self, all_files, target_count=1500):

        print(f"Selecting {target_count} valid files from {len(all_files)} candidates...")

        valid_files = []
        checked_files = 0

        # Shuffle the files to get random selection
        random.shuffle(all_files)

        for filepath in all_files:
            if len(valid_files) >= target_count:
                break

            checked_files += 1
            if checked_files % 50 == 0:
                print(f"Checked {checked_files} files, found {len(valid_files)} valid files")

            file_info = self.get_file_info(filepath)

            if file_info['valid']:
                valid_files.append(filepath)
                print(f"✓ Valid file {len(valid_files)}: {os.path.basename(filepath)} "
                      f"({file_info['size_mb']:.1f}MB, {file_info['duration']:.1f}s)")
            else:
                print(f" Skipped: {os.path.basename(filepath)} - {file_info['reason']}")

        print(f"\nSelected {len(valid_files)} valid files out of {checked_files} checked")
        return valid_files

    def load_audio_efficiently(self, filepath, max_length_seconds=None):

        # Load only the needed portion
        if max_length_seconds:
            # Load only first part of file
            audio, sr = torchaudio.load(filepath, frame_offset=0,
                                        num_frames=int(max_length_seconds * self.sample_rate))
        else:
            audio, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            del resampler  # Free memory

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Convert to numpy and normalize
        audio_np = audio.squeeze().numpy()

        # Normalize to prevent overflow
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val

        # Clear torch tensors
        del audio
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return audio_np, True

    def analyze_childes_noise(self, childes_folder, n_samples=1200):
        print("Analyzing CHILDES noise characteristics...")

        # Find all wav files recursively
        all_wav_files = self.find_all_wav_files(childes_folder)

        if not all_wav_files:
            raise ValueError(f"No wav files found in {childes_folder}")

        # Select valid files efficiently
        valid_files = self.select_valid_files(all_wav_files, n_samples)

        if len(valid_files) < min(10, n_samples // 2):
            raise ValueError(f"Only found {len(valid_files)} valid files, need at least {min(10, n_samples // 2)}")

        sampled_files = valid_files[:n_samples]

        # Initialize lists to store all metrics
        reverb_times = []
        noise_levels = []
        spectral_profiles = []
        spectral_centroids = []
        spectral_rolloffs = []
        spectral_flatness = []
        zero_crossing_rates = []
        mfccs = []
        snr_estimates = []

        successful_files = 0

        for i, filepath in enumerate(sampled_files):
            print(f"Processing file {i + 1}/{len(sampled_files)}: {os.path.basename(filepath)}")

            # Load audio efficiently (limit to first 60 seconds for analysis)
            audio_np, success = self.load_audio_efficiently(filepath, max_length_seconds=60)

            if not success or audio_np is None:
                print(f"  Failed to load {filepath}")
                continue

            # Skip if audio is too short or contains only silence
            if len(audio_np) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                print(f"  Skipping {filepath}: too short after loading")
                continue

            if np.max(np.abs(audio_np)) < 1e-6:  # Nearly silent
                print(f"  Skipping {filepath}: nearly silent")
                continue

            # Process in chunks to avoid memory issues
            chunk_size = self.sample_rate * 30  # 30 second chunks

            if len(audio_np) > chunk_size:
                # Use first chunk for analysis
                audio_chunk = audio_np[:chunk_size]
            else:
                audio_chunk = audio_np

            # Estimate reverberation time (RT60)
            rt60 = self._estimate_rt60_improved(audio_chunk)
            reverb_times.append(rt60)

            # Estimate noise level
            noise_level = self._estimate_noise_level_improved(audio_chunk)
            noise_levels.append(noise_level)

            # Extract spectral profile (use smaller chunk for memory efficiency)
            small_chunk = audio_chunk[:self.sample_rate * 10]  # 10 seconds max
            spectral_profile = self._extract_spectral_profile(small_chunk)
            spectral_profiles.append(spectral_profile)

            # Additional spectral features
            spectral_centroid = self._compute_spectral_centroid(small_chunk)
            spectral_centroids.append(spectral_centroid)

            spectral_rolloff = self._compute_spectral_rolloff(small_chunk)
            spectral_rolloffs.append(spectral_rolloff)

            spectral_flat = self._compute_spectral_flatness(small_chunk)
            spectral_flatness.append(spectral_flat)

            zcr = self._compute_zero_crossing_rate(small_chunk)
            zero_crossing_rates.append(zcr)

            # MFCC features
            mfcc = self._compute_mfcc(small_chunk)
            mfccs.append(mfcc)

            # SNR estimation
            snr = self._estimate_snr(small_chunk)
            snr_estimates.append(snr)

            successful_files += 1

            # Force garbage collection every 10 files
            if i % 10 == 0:
                gc.collect()
        print(f"Successfully analyzed {successful_files} files")

        if successful_files == 0:
            raise ValueError("No files could be analyzed successfully")

        # Store characteristics
        self.noise_profile = {
            'rt60_mean': np.mean(reverb_times),
            'rt60_std': np.std(reverb_times),
            'rt60_median': np.median(reverb_times),
            'noise_level_mean': np.mean(noise_levels),
            'noise_level_std': np.std(noise_levels),
            'noise_level_median': np.median(noise_levels),
            'spectral_profile_mean': np.mean(spectral_profiles, axis=0),
            'spectral_profile_std': np.std(spectral_profiles, axis=0),
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloffs),
            'spectral_rolloff_std': np.std(spectral_rolloffs),
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'spectral_flatness_std': np.std(spectral_flatness),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rates),
            'zero_crossing_rate_std': np.std(zero_crossing_rates),
            'mfcc_mean': np.mean(mfccs, axis=0),
            'mfcc_std': np.std(mfccs, axis=0),
            'snr_mean': np.mean(snr_estimates),
            'snr_std': np.std(snr_estimates),
            'snr_median': np.median(snr_estimates),
            'successful_files': successful_files,
            'total_files': len(sampled_files)
        }

        self._print_analysis_results()
        return self.noise_profile

    def _estimate_rt60_improved(self, audio):
        """Improved RT60 estimation using multiple methods - Memory optimized"""
        try:
            # Limit audio length for RT60 estimation
            max_samples = self.sample_rate * 20  # 20 seconds max
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Method 1: Energy decay method
            energy = audio ** 2

            # Apply smoothing with reasonable window size
            window_size = min(1024, len(energy) // 20, 2048)  # Limit window size
            if window_size < 3:
                window_size = 3
            if window_size % 2 == 0:
                window_size += 1

            try:
                energy_smooth = signal.savgol_filter(energy, window_size, 3)
            except:
                # Fallback to simple moving average
                energy_smooth = np.convolve(energy, np.ones(window_size) / window_size, mode='same')

            # Convert to dB with proper handling of zeros
            energy_smooth = np.maximum(energy_smooth, np.max(energy_smooth) * 1e-10)
            energy_db = 10 * np.log10(energy_smooth)

            # Find the peak and estimate decay
            peak_idx = np.argmax(energy_db)

            # Look for decay after peak
            decay_portion = energy_db[peak_idx:]

            if len(decay_portion) < 100:
                return 0.3  # Default reasonable value

            # Fit linear decay with limited samples
            sample_step = max(1, len(decay_portion) // 1000)  # Limit to 1000 points max
            time_samples = np.arange(0, len(decay_portion), sample_step)
            decay_samples = decay_portion[::sample_step]

            if len(time_samples) > 10:
                coeffs = np.polyfit(time_samples, decay_samples, 1)
                slope = coeffs[0]

                if slope < 0:
                    rt60 = -60 / (slope * self.sample_rate)
                    return np.clip(rt60, 0.1, 2.0)

        except Exception as e:
            print(f"RT60 estimation error: {e}")

        return 0.3

    def _estimate_noise_level_improved(self, audio):
        """Improved noise level estimation - Memory optimized"""
        try:
            # Limit audio length
            max_samples = self.sample_rate * 30  # 30 seconds max
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Method 1: Percentile-based approach
            energy = audio ** 2
            noise_floor_percentile = np.percentile(energy, 5)

            # Method 2: Frame-based approach with limited frames
            frame_length = 1024
            hop_length = 512
            max_frames = 1000  # Limit number of frames

            frames = []
            frame_count = 0

            for i in range(0, len(audio) - frame_length, hop_length):
                if frame_count >= max_frames:
                    break
                frame = audio[i:i + frame_length]
                frames.append(np.mean(frame ** 2))
                frame_count += 1

            if len(frames) > 0:
                frames = np.array(frames)
                noise_frames = frames[frames < np.percentile(frames, 20)]
                if len(noise_frames) > 0:
                    noise_level_vad = np.sqrt(np.mean(noise_frames))
                else:
                    noise_level_vad = np.sqrt(noise_floor_percentile)
            else:
                noise_level_vad = np.sqrt(noise_floor_percentile)

            noise_level = (np.sqrt(noise_floor_percentile) + noise_level_vad) / 2
            return noise_level

        except Exception as e:
            print(f"Noise level estimation error: {e}")
            return 0.001

    def _extract_spectral_profile(self, audio):
        """Extract spectral characteristics"""
        try:
            # Limit audio length
            max_samples = self.sample_rate * 10  # 10 seconds max
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Use smaller nperseg for memory efficiency
            nperseg = min(1024, len(audio) // 4)
            f, psd = signal.welch(audio, fs=self.sample_rate, nperseg=nperseg)

            # Focus on speech frequency range (0-8kHz)
            speech_idx = f <= 8000
            psd_speech = psd[speech_idx]

            # Downsample if too many points
            if len(psd_speech) > 100:
                indices = np.linspace(0, len(psd_speech) - 1, 100).astype(int)
                psd_speech = psd_speech[indices]

            return psd_speech

        except Exception as e:
            print(f"Spectral profile error: {e}")
            return np.zeros(50)

    def _compute_spectral_centroid(self, audio):
        """Compute spectral centroid"""
        try:
            # Limit audio length
            max_samples = self.sample_rate * 5  # 5 seconds max
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            nperseg = min(1024, len(audio) // 4)
            f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=nperseg)
            magnitude = np.abs(Zxx)

            # Compute centroid for limited number of frames
            max_frames = min(100, magnitude.shape[1])
            centroids = []

            for i in range(0, max_frames, max(1, max_frames // 50)):
                spectrum = magnitude[:, i]
                if np.sum(spectrum) > 0:
                    centroid = np.sum(f * spectrum) / np.sum(spectrum)
                    centroids.append(centroid)

            return np.mean(centroids) if centroids else 0

        except Exception as e:
            print(f"Spectral centroid error: {e}")
            return 0

    def _compute_spectral_rolloff(self, audio):
        """Compute spectral rolloff"""
        try:
            max_samples = self.sample_rate * 5
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            nperseg = min(1024, len(audio) // 4)
            f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=nperseg)
            magnitude = np.abs(Zxx) ** 2

            rolloffs = []
            max_frames = min(50, magnitude.shape[1])

            for i in range(0, max_frames, max(1, max_frames // 25)):
                spectrum = magnitude[:, i]
                if np.sum(spectrum) > 0:
                    cumsum = np.cumsum(spectrum)
                    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
                    if len(rolloff_idx) > 0:
                        rolloffs.append(f[rolloff_idx[0]])

            return np.mean(rolloffs) if rolloffs else 0

        except Exception as e:
            print(f"Spectral rolloff error: {e}")
            return 0

    def _compute_spectral_flatness(self, audio):
        """Compute spectral flatness"""
        try:
            max_samples = self.sample_rate * 5
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            nperseg = min(1024, len(audio) // 4)
            f, psd = signal.welch(audio, fs=self.sample_rate, nperseg=nperseg)

            psd = np.maximum(psd, 1e-10)
            geometric_mean = np.exp(np.mean(np.log(psd)))
            arithmetic_mean = np.mean(psd)

            return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0

        except Exception as e:
            print(f"Spectral flatness error: {e}")
            return 0

    def _compute_zero_crossing_rate(self, audio):
        """Compute zero crossing rate"""
        try:
            max_samples = self.sample_rate * 10
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            signs = np.sign(audio)
            zero_crossings = np.sum(np.abs(np.diff(signs))) / 2
            return zero_crossings / len(audio)

        except Exception as e:
            print(f"ZCR error: {e}")
            return 0

    def _compute_mfcc(self, audio):
        """Compute MFCC features"""
        try:
            max_samples = self.sample_rate * 10
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            return np.mean(mfccs, axis=1)

        except Exception as e:
            print(f"MFCC error: {e}")
            return np.zeros(13)

    def _estimate_snr(self, audio):
        """Estimate Signal-to-Noise Ratio"""
        try:
            max_samples = self.sample_rate * 10
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            frame_length = 1024
            hop_length = 512
            max_frames = 200

            frame_energies = []
            frame_count = 0

            for i in range(0, len(audio) - frame_length, hop_length):
                if frame_count >= max_frames:
                    break
                frame = audio[i:i + frame_length]
                frame_energies.append(np.mean(frame ** 2))
                frame_count += 1

            if len(frame_energies) == 0:
                return 10

            frame_energies = np.array(frame_energies)
            signal_energy = np.mean(frame_energies[frame_energies > np.percentile(frame_energies, 50)])
            noise_energy = np.mean(frame_energies[frame_energies < np.percentile(frame_energies, 20)])

            if noise_energy > 0:
                snr_linear = signal_energy / noise_energy
                snr_db = 10 * np.log10(snr_linear)
                return np.clip(snr_db, -10, 50)
            else:
                return 30

        except Exception as e:
            print(f"SNR estimation error: {e}")
            return 10

    def _print_analysis_results(self):
        """Print detailed analysis results"""
        print(f"\n=== CHILDES Noise Analysis Results ===")
        print(
            f"Successfully analyzed: {self.noise_profile['successful_files']}/{self.noise_profile['total_files']} files")
        print(f"\nReverberation (RT60):")
        print(f"  Mean: {self.noise_profile['rt60_mean']:.3f} seconds")
        print(f"  Std:  {self.noise_profile['rt60_std']:.3f} seconds")
        print(f"  Median: {self.noise_profile['rt60_median']:.3f} seconds")

        print(f"\nNoise Level:")
        print(f"  Mean: {self.noise_profile['noise_level_mean']:.6f}")
        print(f"  Std:  {self.noise_profile['noise_level_std']:.6f}")
        print(f"  Median: {self.noise_profile['noise_level_median']:.6f}")

        print(f"\nSpectral Features:")
        print(
            f"  Centroid: {self.noise_profile['spectral_centroid_mean']:.1f} ± {self.noise_profile['spectral_centroid_std']:.1f} Hz")
        print(
            f"  Rolloff: {self.noise_profile['spectral_rolloff_mean']:.1f} ± {self.noise_profile['spectral_rolloff_std']:.1f} Hz")
        print(
            f"  Flatness: {self.noise_profile['spectral_flatness_mean']:.4f} ± {self.noise_profile['spectral_flatness_std']:.4f}")

        print(f"\nTemporal Features:")
        print(
            f"  Zero Crossing Rate: {self.noise_profile['zero_crossing_rate_mean']:.4f} ± {self.noise_profile['zero_crossing_rate_std']:.4f}")

        print(f"\nSignal Quality:")
        print(f"  SNR: {self.noise_profile['snr_mean']:.1f} ± {self.noise_profile['snr_std']:.1f} dB")
        print(f"  SNR Median: {self.noise_profile['snr_median']:.1f} dB")

    def save_noise_profile(self, filename):
        """Save noise profile to file"""
        np.savez(filename, **self.noise_profile)
        print(f"Noise profile saved to {filename}")

    def load_noise_profile(self, filename):
        """Load noise profile from file"""
        data = np.load(filename)
        self.noise_profile = {key: data[key] for key in data.files}
        print(f"Noise profile loaded from {filename}")


class CHILDESNoiseSimulator:
    """Simulate CHILDES-like noise conditions on clean recordings"""

    def __init__(self, noise_profile: Dict, sample_rate: int = 16000):
        self.noise_profile = noise_profile
        self.sample_rate = sample_rate
        self._prepare_noise_shaping_filter()

        # Better normalization targets for speech enhancement
        self.target_rms = 0.03  # Lower target RMS for better dynamic range
        self.max_amplitude = 0.8  # More conservative max to prevent clipping
        self.min_snr = 0.0  # Minimum SNR in dB
        self.max_snr = 25.0  # Maximum SNR in dB

    def _prepare_noise_shaping_filter(self):
        """Prepare filter for spectral shaping of noise"""
        try:
            spectral_profile = self.noise_profile.get('spectral_profile_mean', None)
            if spectral_profile is None or len(spectral_profile) == 0:
                self.noise_shaping_filter = None
                return

            # Normalize and smooth the spectral profile
            spectral_profile = np.array(spectral_profile)
            max_val = np.max(spectral_profile)
            if max_val > 0:
                spectral_profile = spectral_profile / max_val
            else:
                self.noise_shaping_filter = None
                return

            # Create filter coefficients (simple FIR filter)
            filter_order = min(512, len(spectral_profile) * 4)
            self.noise_shaping_filter = signal.firwin(filter_order, 0.5, window='hamming')

            # Modify filter based on spectral profile
            if len(spectral_profile) > 10:
                resampled_profile = signal.resample(spectral_profile, filter_order)
                self.noise_shaping_filter = self.noise_shaping_filter * resampled_profile

        except Exception as e:
            print(f"Warning: Could not create noise shaping filter: {e}")
            self.noise_shaping_filter = None

    def simulate_childes_conditions(self, clean_audio: torch.Tensor,
                                    intensity_factor: float = 1.0) -> torch.Tensor:

        if len(clean_audio.shape) == 1:
            clean_audio = clean_audio.unsqueeze(0)

        # Validate input
        if clean_audio.shape[1] == 0:
            return clean_audio

        # Normalize input audio to consistent level
        audio_rms = torch.sqrt(torch.mean(clean_audio ** 2))
        if audio_rms < 1e-8:
            clean_audio = clean_audio + 1e-6 * torch.randn_like(clean_audio)
            audio_rms = torch.sqrt(torch.mean(clean_audio ** 2))

        # Normalize to target RMS
        clean_audio = clean_audio * (self.target_rms / audio_rms)

        degraded_audio = clean_audio.clone()

        # Apply degradations in realistic order
        try:
            # 1. First add reverberation (room acoustics)
            rt60 = self._sample_rt60() * np.clip(intensity_factor, 0.1, 2.0)
            degraded_audio = self._add_improved_reverberation(degraded_audio, rt60)
        except Exception as e:
            print(f"Warning: Reverberation failed: {e}")

        try:
            # 2. Add spectral coloration (microphone/preamp response)
            degraded_audio = self._add_spectral_coloration(degraded_audio)
        except Exception as e:
            print(f"Warning: Spectral coloration failed: {e}")

        try:
            # 3. Add recording artifacts (electrical noise, etc.)
            degraded_audio = self._add_realistic_recording_artifacts(degraded_audio, intensity_factor)
        except Exception as e:
            print(f"Warning: Recording artifacts failed: {e}")

        try:
            # 4. Add background noise (environmental)
            snr_target = self._sample_snr() + (5 * (1 - np.clip(intensity_factor, 0.1, 1.5)))
            degraded_audio = self._add_shaped_background_noise(degraded_audio, snr_target)
        except Exception as e:
            print(f"Warning: Background noise failed: {e}")

        try:
            # 5. Apply dynamic compression (AGC, limiting)
            degraded_audio = self._add_dynamic_compression(degraded_audio)
        except Exception as e:
            print(f"Warning: Dynamic compression failed: {e}")

        try:
            # 6. Final frequency response variation
            degraded_audio = self._add_frequency_response_variation(degraded_audio)
        except Exception as e:
            print(f"Warning: Frequency response variation failed: {e}")

        # Final normalization to prevent clipping
        max_val = torch.max(torch.abs(degraded_audio))
        if max_val > self.max_amplitude:
            degraded_audio = degraded_audio * (self.max_amplitude / max_val)

        return degraded_audio

    def _sample_rt60(self) -> float:
        """Sample RT60 from profile distribution with fallback"""
        try:
            rt60_mean = self.noise_profile.get('rt60_mean', 0.5)
            rt60_std = self.noise_profile.get('rt60_std', 0.2)

            # Handle invalid statistics
            if rt60_std <= 0 or not np.isfinite(rt60_std):
                rt60_std = 0.2
            if not np.isfinite(rt60_mean):
                rt60_mean = 0.5

            rt60 = np.random.normal(rt60_mean, rt60_std)
            return np.clip(rt60, 0.1, 2.0)
        except:
            return np.random.uniform(0.2, 0.8)

    def _sample_snr(self) -> float:
        """Sample SNR from profile distribution with fallback"""
        try:
            snr_mean = self.noise_profile.get('snr_mean', 15.0)
            snr_std = self.noise_profile.get('snr_std', 5.0)

            # Handle invalid statistics
            if snr_std <= 0 or not np.isfinite(snr_std):
                snr_std = 5.0
            if not np.isfinite(snr_mean):
                snr_mean = 15.0

            snr = np.random.normal(snr_mean, snr_std)
            return np.clip(snr, 0, 30)
        except:
            return np.random.uniform(5, 20)

    def _add_shaped_background_noise(self, audio: torch.Tensor, snr_target_db: float) -> torch.Tensor:
        """Fixed background noise addition with proper SNR calculation"""
        try:
            # Ensure audio is properly shaped
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)

            # Calculate RMS power (not mean squared)
            signal_rms = torch.sqrt(torch.mean(audio ** 2))

            # Skip if signal is too quiet
            if signal_rms < 1e-6:
                return audio

            # Generate white noise
            noise = torch.randn_like(audio)

            # Validate and clip SNR
            snr_target_db = np.clip(snr_target_db, self.min_snr, self.max_snr)
            snr_linear = 10 ** (snr_target_db / 10)

            # Calculate noise RMS needed to achieve target SNR
            # SNR = signal_rms / noise_rms, so noise_rms = signal_rms / SNR
            target_noise_rms = signal_rms / np.sqrt(snr_linear)

            # Scale noise to achieve target RMS
            noise_rms = torch.sqrt(torch.mean(noise ** 2))
            noise_scaled = noise * (target_noise_rms / noise_rms)

            # Mix signal and noise
            noisy_audio = audio + noise_scaled

            # Verify achieved SNR (for debugging)
            final_noise_rms = torch.sqrt(torch.mean(noise_scaled ** 2))
            achieved_snr = 20 * torch.log10(signal_rms / final_noise_rms)

            return noisy_audio

        except Exception as e:
            print(f"Error in shaped background noise: {e}")
            return audio + 0.001 * torch.randn_like(audio)

    def _add_dynamic_compression(self, audio: torch.Tensor) -> torch.Tensor:
        """Fixed dynamic compression with proper audio processing"""
        try:
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)

            audio_np = audio.squeeze().numpy()

            if len(audio_np) == 0:
                return audio

            # Use RMS-based compression instead of peak-based
            frame_size = 1024
            hop_size = 512

            # Calculate RMS for each frame
            rms_values = []
            for i in range(0, len(audio_np) - frame_size, hop_size):
                frame = audio_np[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)

            if not rms_values:
                return audio

            # Adaptive threshold based on signal statistics
            rms_values = np.array(rms_values)
            threshold = np.percentile(rms_values, 75)  # 75th percentile as threshold

            # Gentle compression
            ratio = 2.0  # Less aggressive compression

            compressed = audio_np.copy()

            # Apply frame-by-frame compression
            frame_idx = 0
            for i in range(0, len(audio_np) - frame_size, hop_size):
                if frame_idx < len(rms_values):
                    frame_rms = rms_values[frame_idx]

                    if frame_rms > threshold:
                        # Calculate compression gain
                        excess = frame_rms - threshold
                        compressed_excess = excess / ratio
                        gain = (threshold + compressed_excess) / frame_rms

                        # Apply gain to frame
                        compressed[i:i + frame_size] *= gain

                    frame_idx += 1

            return torch.tensor(compressed, dtype=audio.dtype).unsqueeze(0)

        except Exception as e:
            print(f"Error in dynamic compression: {e}")
            return audio

    def _add_improved_reverberation(self, audio: torch.Tensor, rt60: float) -> torch.Tensor:
        """Fixed reverberation with proper room simulation"""
        try:
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)

            audio_np = audio.squeeze().numpy()

            if len(audio_np) == 0:
                return audio

            # Validate RT60
            rt60 = np.clip(rt60, 0.1, 1.5)  # Reasonable range for room acoustics

            # Calculate RIR length (should be about 3-4 times RT60)
            rir_length = int(rt60 * self.sample_rate * 3)
            rir_length = min(rir_length, self.sample_rate * 2)  # Max 2 seconds
            rir_length = max(rir_length, 256)  # Minimum length

            # Create more realistic room impulse response
            t = np.arange(rir_length) / self.sample_rate

            # Exponential decay with some randomness
            decay = np.exp(-3 * np.log(10) * t / rt60)

            # Add some early reflections
            rir = np.zeros(rir_length)
            rir[0] = 1.0  # Direct path

            # Add early reflections (first 50ms)
            early_samples = int(0.05 * self.sample_rate)
            for i in range(1, min(early_samples, rir_length)):
                if np.random.random() < 0.1:  # 10% chance of reflection
                    rir[i] = np.random.uniform(-0.3, 0.3) * decay[i]

            # Add late reverberation (random noise shaped by decay)
            late_start = early_samples
            if late_start < rir_length:
                late_reverb = np.random.randn(rir_length - late_start) * decay[late_start:]
                rir[late_start:] = late_reverb

            # Normalize RIR
            rir = rir / np.sqrt(np.sum(rir ** 2))

            # Apply convolution
            reverb_audio = signal.convolve(audio_np, rir, mode='same')

            # Mix with dry signal (not fully wet)
            wet_dry_ratio = 0.3  # 30% wet, 70% dry
            mixed_audio = (1 - wet_dry_ratio) * audio_np + wet_dry_ratio * reverb_audio

            return torch.tensor(mixed_audio, dtype=audio.dtype).unsqueeze(0)

        except Exception as e:
            print(f"Error in reverberation: {e}")
            return audio

    def _add_spectral_coloration(self, audio: torch.Tensor) -> torch.Tensor:
        """Add spectral coloration with safe operations"""
        try:
            audio_np = audio.squeeze().numpy()

            if len(audio_np) == 0:
                return audio

            # Check signal level
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms < 1e-8:
                return audio

            # Apply band-pass filter
            nyquist = self.sample_rate / 2
            low_freq = max(200, nyquist * 0.01)
            high_freq = min(6000, nyquist * 0.9)

            if low_freq >= high_freq:
                return audio

            sos = signal.butter(2, [low_freq, high_freq], btype='band',
                                fs=self.sample_rate, output='sos')
            filtered_audio = signal.sosfilt(sos, audio_np)

            # Mix with original
            mix_ratio = 0.3
            colored_audio = (1 - mix_ratio) * audio_np + mix_ratio * filtered_audio

            return torch.tensor(colored_audio, dtype=audio.dtype).unsqueeze(0)

        except Exception as e:
            print(f"Error in spectral coloration: {e}")
            return audio

    def _add_realistic_recording_artifacts(self, audio: torch.Tensor, intensity_factor: float) -> torch.Tensor:
        """Add recording artifacts with safe operations"""
        try:
            audio_with_artifacts = audio.clone()
            audio_length = len(audio.squeeze())

            if audio_length == 0:
                return audio

            # Validate intensity factor
            if not np.isfinite(intensity_factor):
                intensity_factor = 1.0
            intensity_factor = np.clip(intensity_factor, 0.1, 3.0)

            # Add electrical interference (60Hz hum)
            if random.random() < 0.3 * intensity_factor:
                t = torch.arange(audio_length, dtype=audio.dtype) / self.sample_rate
                hum_60 = 0.02 * torch.sin(2 * np.pi * 60 * t)
                hum_120 = 0.01 * torch.sin(2 * np.pi * 120 * t)
                audio_with_artifacts[0, :] += hum_60 + hum_120

            return audio_with_artifacts

        except Exception as e:
            print(f"Error in recording artifacts: {e}")
            return audio

    def _add_frequency_response_variation(self, audio: torch.Tensor) -> torch.Tensor:
        """Add frequency response variations with safe operations"""
        try:
            audio_np = audio.squeeze().numpy()

            if len(audio_np) == 0:
                return audio

            # Check signal level
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms < 1e-8:
                return audio

            # Random EQ-like adjustments with safe frequency ranges
            nyquist = self.sample_rate / 2
            low_cutoff = min(300, nyquist * 0.1)
            high_cutoff = max(3000, nyquist * 0.5)

            if low_cutoff >= nyquist or high_cutoff >= nyquist:
                return audio

            # Apply filters safely
            try:
                # Low freq component
                low_gain = np.random.uniform(0.8, 1.2)
                sos_low = signal.butter(2, low_cutoff, btype='low',
                                        fs=self.sample_rate, output='sos')
                low_component = signal.sosfilt(sos_low, audio_np)

                # High freq component
                high_gain = np.random.uniform(0.7, 1.3)
                sos_high = signal.butter(2, high_cutoff, btype='high',
                                         fs=self.sample_rate, output='sos')
                high_component = signal.sosfilt(sos_high, audio_np)

                # Mid component
                mid_component = audio_np - low_component - high_component

                # Recombine with gains
                eq_audio = low_gain * low_component + mid_component + high_gain * high_component

                return torch.tensor(eq_audio, dtype=audio.dtype).unsqueeze(0)

            except:
                return audio

        except Exception as e:
            print(f"Error in frequency response variation: {e}")
            return audio

    def generate_multiple_versions(self, clean_audio: torch.Tensor,
                                   n_versions: int = 1) -> List[torch.Tensor]:
        """Generate multiple degraded versions with varying intensity"""
        versions = []

        # Validate input
        if clean_audio.shape[1] == 0:
            print("Warning: Empty audio tensor in generate_multiple_versions")
            return [clean_audio] * n_versions

        for i in range(n_versions):
            try:
                # Vary intensity: light, medium, heavy degradation
                intensity = 0.5 + (i / max(1, n_versions - 1)) * 1.0  # Avoid division by zero
                degraded = self.simulate_childes_conditions(clean_audio, intensity)
                versions.append(degraded)
            except Exception as e:
                print(f"Error generating version {i}: {e}")
                # Return original audio as fallback
                versions.append(clean_audio.clone())

        return versions


def safe_audio_processing(splits, simulator, output_folder, versions_per_clean=1):
    """Fixed audio processing with proper normalization and validation"""
    processed_counts = {}

    for split_name, file_list in splits.items():
        print(f"\nProcessing {split_name} split...")

        n_versions = versions_per_clean if split_name == 'train' else 1
        processed_count = 0

        for file_idx, filepath in enumerate(file_list):
            filename = os.path.basename(filepath)
            print(f"  Processing ({file_idx + 1}/{len(file_list)}): {filename}")

            try:
                # Load and validate audio
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    continue

                try:
                    clean_audio, sr = torchaudio.load(filepath)
                except Exception as e:
                    print(f"    Error loading audio: {e}")
                    continue

                # Basic validation
                if clean_audio.shape[1] == 0:
                    continue

                if torch.isnan(clean_audio).any() or torch.isinf(clean_audio).any():
                    continue

                # Resample if necessary
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    clean_audio = resampler(clean_audio)

                # Convert to mono
                if clean_audio.shape[0] > 1:
                    clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)

                # Length validation
                if clean_audio.shape[1] < 16000:  # Less than 1 second
                    continue

                # Audio level validation
                audio_rms = torch.sqrt(torch.mean(clean_audio ** 2))
                if audio_rms < 1e-6:
                    continue

                # Normalize clean audio to reasonable level
                target_rms = 0.1  # Target RMS for clean audio
                clean_audio = clean_audio * (target_rms / audio_rms)

                # Prevent clipping
                max_val = torch.max(torch.abs(clean_audio))
                if max_val > 0.9:
                    clean_audio = clean_audio * (0.9 / max_val)

                # Generate degraded versions
                try:
                    degraded_versions = simulator.generate_multiple_versions(clean_audio, n_versions)
                except Exception as e:
                    print(f"    Error generating degraded versions: {e}")
                    continue

                # Validate all versions
                valid_versions = []
                for i, degraded in enumerate(degraded_versions):
                    if (not torch.isnan(degraded).any() and
                            not torch.isinf(degraded).any() and
                            degraded.shape[1] > 0):
                        valid_versions.append(degraded)

                if not valid_versions:
                    continue

                # Save pairs
                base_name = filename.replace('.wav', '')

                for i, degraded_audio in enumerate(valid_versions):
                    try:
                        if len(valid_versions) > 1:
                            clean_filename = f"{base_name}_v{i:02d}.wav"
                            noisy_filename = f"{base_name}_v{i:02d}_noisy.wav"
                        else:
                            clean_filename = f"{base_name}.wav"
                            noisy_filename = f"{base_name}_noisy.wav"

                        # Final normalization before saving
                        clean_save = clean_audio.clone()
                        noisy_save = degraded_audio.clone()

                        # Ensure both have same RMS level for fair comparison
                        clean_rms = torch.sqrt(torch.mean(clean_save ** 2))
                        noisy_rms = torch.sqrt(torch.mean(noisy_save ** 2))

                        if clean_rms > 1e-6 and noisy_rms > 1e-6:
                            # Normalize both to same RMS level
                            target_rms = 0.1
                            clean_save = clean_save * (target_rms / clean_rms)
                            noisy_save = noisy_save * (target_rms / noisy_rms)

                        # Save files
                        clean_path = f"{output_folder}/{split_name}/clean/{clean_filename}"
                        noisy_path = f"{output_folder}/{split_name}/noisy/{noisy_filename}"

                        torchaudio.save(clean_path, clean_save, 16000)
                        torchaudio.save(noisy_path, noisy_save, 16000)

                    except Exception as e:
                        print(f"    Error saving files: {e}")
                        continue

                processed_count += 1

            except Exception as e:
                print(f"    Unexpected error: {e}")
                continue

        processed_counts[split_name] = processed_count
        print(f"  {split_name}: {processed_count} files processed successfully")

    return processed_counts


def create_training_dataset_with_splits_safe(clean_folder: str, childes_folder: str,
                                             output_folder: str, n_analyze: int = 3000,
                                             train_split: float = 0.7, val_split: float = 0.15,
                                             test_split: float = 0.15, random_state: int = 42,
                                             versions_per_clean: int = 1):
    """Safe version of dataset creation with comprehensive error handling"""

    print("=== Creating CHILDES-matched training dataset (SAFE VERSION) ===")

    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Step 1: Analyze CHILDES noise characteristics
    print("Step 1: Analyzing CHILDES noise characteristics...")
    try:
        analyzer = CHILDESNoiseAnalyzer()
        noise_profile = analyzer.analyze_childes_noise(childes_folder, n_analyze)
    except Exception as e:
        print(f"Error analyzing CHILDES noise: {e}")
        raise

    # Step 2: Create simulator with validation
    print("Step 2: Creating noise simulator...")
    try:
        simulator = CHILDESNoiseSimulator(noise_profile)
    except Exception as e:
        print(f"Error creating simulator: {e}")
        raise

    # Step 3: Find all clean audio files
    print("Step 3: Finding clean audio files...")
    try:
        clean_files = analyzer.find_audio_folders(clean_folder)
        if not clean_files:
            clean_files = analyzer.find_all_wav_files(clean_folder)
        if not clean_files:
            raise ValueError(f"No wav files found in {clean_folder}")
        print(f"Found {len(clean_files)} clean audio files")
    except Exception as e:
        print(f"Error finding clean files: {e}")
        raise

    # Step 4: Split files into train/val/test
    print("Step 4: Splitting files...")
    try:
        np.random.seed(random_state)
        random.seed(random_state)

        train_files, temp_files = train_test_split(
            clean_files, test_size=(val_split + test_split), random_state=random_state
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_split / (val_split + test_split)), random_state=random_state
        )

        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    except Exception as e:
        print(f"Error splitting files: {e}")
        raise

    # Step 5: Create output directory structure
    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    try:
        for split_name in splits.keys():
            os.makedirs(f"{output_folder}/{split_name}/clean", exist_ok=True)
            os.makedirs(f"{output_folder}/{split_name}/noisy", exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        raise

    # Step 6: Process each split safely
    try:
        processed_counts = safe_audio_processing(splits, simulator, output_folder, versions_per_clean)
    except Exception as e:
        print(f"Error processing audio files: {e}")
        raise

    # Step 7: Save metadata safely
    print("\nStep 7: Saving metadata...")
    try:
        # Save noise profile with proper serialization
        noise_profile_serializable = {}
        for k, v in noise_profile.items():
            if isinstance(v, np.ndarray):
                # Handle potential NaN/inf values
                v_clean = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                noise_profile_serializable[k] = v_clean.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                # Handle numpy scalars
                if np.isfinite(v):
                    noise_profile_serializable[k] = float(v)
                else:
                    noise_profile_serializable[k] = 0.0
            else:
                noise_profile_serializable[k] = v

        with open(f"{output_folder}/noise_profile.json", 'w') as f:
            json.dump(noise_profile_serializable, f, indent=2)

        # Save split information
        splits_info = {
            'train_files': len(train_files),
            'val_files': len(val_files),
            'test_files': len(test_files),
            'processed_counts': processed_counts,
            'versions_per_clean': versions_per_clean,
            'random_state': random_state
        }

        with open(f"{output_folder}/splits_info.json", 'w') as f:
            json.dump(splits_info, f, indent=2)

    except Exception as e:
        print(f"Error saving metadata: {e}")
        raise

    print(f"\n=== Dataset Creation Complete ===")
    print(f"Output folder: {output_folder}")
    print(f"Files processed: {processed_counts}")

    return noise_profile, splits_info


def organize_audio_files(main_folder, clean_output_folder, noisy_output_folder):
    """
    Organizes audio files from a nested structure:
    - Files starting with 'P' are considered clean
    - All others are considered noisy

    Args:
        main_folder: Root folder containing subfolders with audio files
        clean_output_folder: Where to copy clean files
        noisy_output_folder: Where to copy noisy files
    """
    # Create output folders if they don't exist
    os.makedirs(clean_output_folder, exist_ok=True)
    os.makedirs(noisy_output_folder, exist_ok=True)

    # Walk through the main folder structure
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                src_path = os.path.join(root, file)

                # Check if file starts with 'P' (clean)
                if file.startswith('P'):
                    dest_folder = clean_output_folder
                else:
                    dest_folder = noisy_output_folder

                # Copy file to appropriate folder
                dest_path = os.path.join(dest_folder, file)

                # Handle duplicate filenames
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")


def create_noisy_dataset(clean_folder: str, childes_folder: str,
                         output_folder: str, n_analyze: int = 4000):
    """Create noisy versions of all clean files in a single output folder"""

    print("=== Creating Noisy Dataset ===")

    # Create Auris output folder
    auris_folder = os.path.join(output_folder, "Auris")
    os.makedirs(auris_folder, exist_ok=True)

    # 1. Analyze noise characteristics
    print("Analyzing noise samples...")
    try:
        analyzer = CHILDESNoiseAnalyzer()
        noise_profile = analyzer.analyze_childes_noise(childes_folder, n_analyze)
        simulator = CHILDESNoiseSimulator(noise_profile)
    except Exception as e:
        print(f"Error in noise analysis: {e}")
        raise

    # 2. Process all clean files
    clean_files = analyzer.find_all_wav_files(clean_folder)
    if not clean_files:
        raise ValueError(f"No WAV files found in {clean_folder}")

    processed_files = []

    for clean_file in tqdm(clean_files, desc="Processing files"):
        try:
            # Load clean audio
            clean_audio, sr = torchaudio.load(clean_file)
            if clean_audio.shape[0] > 1:
                clean_audio = clean_audio.mean(dim=0, keepdim=True)  # Convert to mono

            # Create noisy version
            noisy_audio = simulator.simulate_childes_conditions(clean_audio)

            # Save with same filename in Auris folder
            output_path = os.path.join(auris_folder, os.path.basename(clean_file))
            torchaudio.save(output_path, noisy_audio, sr)

            processed_files.append(clean_file)

        except Exception as e:
            print(f"\nError processing {clean_file}: {e}")
            continue

    # 3. Save metadata
    metadata = {
        'noise_profile': noise_profile,
        'processed_files': processed_files,
        'total_files': len(processed_files),
        'failed_files': len(clean_files) - len(processed_files)
    }

    with open(os.path.join(output_folder, 'noise_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Dataset Creation Complete ===")
    print(f"Successfully processed: {len(processed_files)}/{len(clean_files)} files")
    print(f"Noisy files saved to: {auris_folder}")

    return metadata


if __name__ == "__main__":
    main_folder = ""
    clean_folder = "./Dataset/Clean_Files"
    noisy_folder = "./Dataset/Noisy_Files"
    output_folder = "./Auris"

    # Step 1: Organize files into clean and noisy folders
    print("Organizing audio files...")
    #organize_audio_files(main_folder, clean_folder, noisy_folder)

    # Step 2: Create training dataset
    print("Creating training dataset...")
    noise_profile, splits_info = create_noisy_dataset(
        clean_folder=clean_folder,
        childes_folder=noisy_folder,  # Using noisy files as childes_folder
        output_folder=output_folder,
        n_analyze=4000,
    )

    print("\nDataset creation completed successfully!")
    print(f"Total files processed: {sum(splits_info['processed_counts'].values())}")
