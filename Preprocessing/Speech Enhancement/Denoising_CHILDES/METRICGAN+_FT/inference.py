"""
Inference and evaluation script for fine-tuned MetricGAN+ model
"""
import psutil
import torch
import torchaudio
import numpy as np
import os
import json
from speechbrain.inference.enhancement import SpectralMaskEnhancement
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import argparse

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _calculate_stoi_simple(clean, enhanced):
    """Simplified STOI-like score"""
    # Spectral correlation in frequency domain
    clean_fft = np.fft.fft(clean)
    enhanced_fft = np.fft.fft(enhanced)

    # Magnitude correlation
    clean_mag = np.abs(clean_fft)
    enhanced_mag = np.abs(enhanced_fft)

    correlation = np.corrcoef(clean_mag, enhanced_mag)[0, 1]
    return max(0, correlation)


def _calculate_pesq_simple(clean, enhanced):
    """Simplified PESQ-like score"""
    # Correlation-based quality measure
    correlation = np.corrcoef(clean, enhanced)[0, 1]
    return max(0, correlation)


def _calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio"""
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def _calculate_segmental_snr(clean, enhanced, frame_length=1024, overlap=0.5):
    """Calculate segmental SNR"""
    hop_length = int(frame_length * (1 - overlap))
    num_frames = (len(clean) - frame_length) // hop_length + 1

    segmental_snrs = []

    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length

        clean_frame = clean[start_idx:end_idx]
        enhanced_frame = enhanced[start_idx:end_idx]

        # Calculate SNR for this frame
        noise_frame = enhanced_frame - clean_frame
        signal_power = np.mean(clean_frame ** 2)
        noise_power = np.mean(noise_frame ** 2)

        if noise_power > 0 and signal_power > 0:
            frame_snr = 10 * np.log10(signal_power / noise_power)
            segmental_snrs.append(frame_snr)

    return np.mean(segmental_snrs) if segmental_snrs else 0


def plot_comparison_results(comparison_df):
    """Plot comparison results"""
    print("Plotting comparison results...")

    # Get improvement columns
    improvement_cols = [col for col in comparison_df.columns if col.startswith('improvement_')]

    if not improvement_cols:
        print("No improvement metrics found for plotting")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison Results', fontsize=16)

    # Plot 1: SNR Improvement
    if 'improvement_snr_improvement' in comparison_df.columns:
        axes[0, 0].hist(comparison_df['improvement_snr_improvement'], bins=20, alpha=0.7)
        axes[0, 0].set_title('SNR Improvement Distribution')
        axes[0, 0].set_xlabel('SNR Improvement (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # Plot 2: PESQ Score Improvement
    if 'improvement_pesq_score' in comparison_df.columns:
        axes[0, 1].hist(comparison_df['improvement_pesq_score'], bins=20, alpha=0.7)
        axes[0, 1].set_title('PESQ Score Improvement Distribution')
        axes[0, 1].set_xlabel('PESQ Score Improvement')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # Plot 3: Overall improvement metrics
    improvement_means = comparison_df[improvement_cols].mean()
    axes[1, 0].bar(range(len(improvement_means)), improvement_means.values)
    axes[1, 0].set_title('Average Improvement by Metric')
    axes[1, 0].set_xticks(range(len(improvement_means)))
    axes[1, 0].set_xticklabels([col.replace('improvement_', '') for col in improvement_means.index],
                               rotation=45, ha='right')
    axes[1, 0].set_ylabel('Average Improvement')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # Plot 4: Scatter plot of pretrained vs finetuned SNR
    if 'pretrained_snr_enhanced' in comparison_df.columns and 'finetuned_snr_enhanced' in comparison_df.columns:
        axes[1, 1].scatter(comparison_df['pretrained_snr_enhanced'],
                           comparison_df['finetuned_snr_enhanced'], alpha=0.7)
        axes[1, 1].set_title('SNR: Pretrained vs Finetuned')
        axes[1, 1].set_xlabel('Pretrained SNR (dB)')
        axes[1, 1].set_ylabel('Finetuned SNR (dB)')

        # Add diagonal line for reference
        min_val = min(comparison_df['pretrained_snr_enhanced'].min(),
                      comparison_df['finetuned_snr_enhanced'].min())
        max_val = max(comparison_df['pretrained_snr_enhanced'].max(),
                      comparison_df['finetuned_snr_enhanced'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved as 'model_comparison_plots.png'")


class SpeechEnhancementEvaluator:
    """Comprehensive evaluation of speech enhancement quality"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def evaluate_enhancement(self, clean_audio, noisy_audio, enhanced_audio):
        """
        Evaluate enhancement quality using multiple metrics

        Args:
            clean_audio (torch.Tensor): Original clean audio
            noisy_audio (torch.Tensor): Noisy version
            enhanced_audio (torch.Tensor): Enhanced version

        Returns:
            dict: Dictionary of evaluation metrics
        """

        # Convert to numpy for processing
        clean_np = clean_audio.squeeze().numpy()
        noisy_np = noisy_audio.squeeze().numpy()
        enhanced_np = enhanced_audio.squeeze().numpy()

        # Align lengths
        min_len = min(len(clean_np), len(noisy_np), len(enhanced_np))
        clean_np = clean_np[:min_len]
        noisy_np = noisy_np[:min_len]
        enhanced_np = enhanced_np[:min_len]

        metrics = {}

        # 1. Signal-to-Noise Ratio improvement
        metrics['snr_improvement'] = self._calculate_snr_improvement(
            clean_np, noisy_np, enhanced_np
        )

        # 2. Signal-to-Noise Ratio (absolute values)
        metrics['snr_original'] = self._calculate_snr(clean_np, noisy_np)
        metrics['snr_enhanced'] = self._calculate_snr(clean_np, enhanced_np)

        # 3. Perceptual Evaluation of Speech Quality (PESQ) - simplified
        metrics['pesq_score'] = self._calculate_pesq_simple(clean_np, enhanced_np)

        # 4. Short-Time Objective Intelligibility (STOI) - simplified
        metrics['stoi_score'] = self._calculate_stoi_simple(clean_np, enhanced_np)

        # 5. Spectral Distortion
        metrics['spectral_distortion'] = self._calculate_spectral_distortion(
            clean_np, enhanced_np
        )

        # 6. Mean Squared Error
        metrics['mse'] = mean_squared_error(clean_np, enhanced_np)

        # 7. Correlation coefficient
        metrics['correlation'] = np.corrcoef(clean_np, enhanced_np)[0, 1]

        # 8. Segmental SNR
        metrics['segmental_snr'] = self._calculate_segmental_snr(clean_np, enhanced_np)

        return metrics

    def _calculate_snr_improvement(self, clean, noisy, enhanced):
        """Calculate SNR improvement"""
        snr_before = self._calculate_snr(clean, noisy)
        snr_after = self._calculate_snr(clean, enhanced)
        return snr_after - snr_before

    def _calculate_spectral_distortion(self, clean, enhanced):
        """Calculate spectral distortion"""
        # Compute power spectral densities
        _, psd_clean = signal.welch(clean, fs=self.sample_rate)
        _, psd_enhanced = signal.welch(enhanced, fs=self.sample_rate)

        # Log spectral distortion
        lsd = np.mean((10 * np.log10(psd_clean + 1e-10) -
                       10 * np.log10(psd_enhanced + 1e-10)) ** 2)

        return np.sqrt(lsd)

    def evaluate_test_set(self, test_clean_folder, test_noisy_folder, test_enhanced_folder, output_file):
        """Evaluate all files in test set"""
        print(f"Evaluating test set, saving results to {output_file}")

        clean_files = sorted([f for f in os.listdir(test_clean_folder) if f.endswith('.wav')])
        results = []

        for clean_file in clean_files:
            print(f"Evaluating: {clean_file}")

            # Build file paths
            noisy_file = clean_file.replace('.wav', '_noisy.wav')
            enhanced_file = clean_file.replace('.wav', '_enhanced.wav')

            clean_path = os.path.join(test_clean_folder, clean_file)
            noisy_path = os.path.join(test_noisy_folder, noisy_file)
            enhanced_path = os.path.join(test_enhanced_folder, enhanced_file)

            # Check if all files exist
            if not all(os.path.exists(path) for path in [clean_path, noisy_path, enhanced_path]):
                print(f"Warning: Missing files for {clean_file}")
                continue

            try:
                # Load audio files
                clean_audio, _ = torchaudio.load(clean_path)
                noisy_audio, _ = torchaudio.load(noisy_path)
                enhanced_audio, _ = torchaudio.load(enhanced_path)

                # Convert to mono if needed
                if clean_audio.shape[0] > 1:
                    clean_audio = clean_audio.mean(dim=0, keepdim=True)
                if noisy_audio.shape[0] > 1:
                    noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
                if enhanced_audio.shape[0] > 1:
                    enhanced_audio = enhanced_audio.mean(dim=0, keepdim=True)

                # Evaluate
                metrics = self.evaluate_enhancement(clean_audio, noisy_audio, enhanced_audio)
                metrics['filename'] = clean_file
                results.append(metrics)

            except Exception as e:
                print(f"Error evaluating {clean_file}: {str(e)}")
                continue

        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

            # Print summary statistics
            print("\nSummary Statistics:")
            print(df.describe())

            return df
        else:
            print("No results to save")
            return None

    def compare_models(self, test_clean_folder, test_noisy_folder, pretrained_output_folder,
                       finetuned_output_folder, results_file):
        """Compare pretrained and finetuned models"""
        print(f"Comparing models, saving results to {results_file}")

        clean_files = sorted([f for f in os.listdir(test_clean_folder) if f.endswith('.wav')])
        comparison_results = []

        for clean_file in clean_files:
            print(f"Comparing: {clean_file}")

            # Build file paths
            noisy_file = clean_file.replace('.wav', '_noisy.wav')
            enhanced_file = clean_file.replace('.wav', '_enhanced.wav')

            clean_path = os.path.join(test_clean_folder, clean_file)
            noisy_path = os.path.join(test_noisy_folder, noisy_file)
            pretrained_path = os.path.join(pretrained_output_folder, enhanced_file)
            finetuned_path = os.path.join(finetuned_output_folder, enhanced_file)

            # Check if all files exist
            if not all(os.path.exists(path) for path in [clean_path, noisy_path, pretrained_path, finetuned_path]):
                print(f"Warning: Missing files for {clean_file}")
                continue

            try:
                # Load audio files
                clean_audio, _ = torchaudio.load(clean_path)
                noisy_audio, _ = torchaudio.load(noisy_path)
                pretrained_audio, _ = torchaudio.load(pretrained_path)
                finetuned_audio, _ = torchaudio.load(finetuned_path)

                # Convert to mono if needed
                if clean_audio.shape[0] > 1:
                    clean_audio = clean_audio.mean(dim=0, keepdim=True)
                if noisy_audio.shape[0] > 1:
                    noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
                if pretrained_audio.shape[0] > 1:
                    pretrained_audio = pretrained_audio.mean(dim=0, keepdim=True)
                if finetuned_audio.shape[0] > 1:
                    finetuned_audio = finetuned_audio.mean(dim=0, keepdim=True)

                # Evaluate both models
                pretrained_metrics = self.evaluate_enhancement(clean_audio, noisy_audio, pretrained_audio)
                finetuned_metrics = self.evaluate_enhancement(clean_audio, noisy_audio, finetuned_audio)

                # Create comparison result
                result = {'filename': clean_file}
                for metric in pretrained_metrics:
                    result[f'pretrained_{metric}'] = pretrained_metrics[metric]
                    result[f'finetuned_{metric}'] = finetuned_metrics[metric]
                    result[f'improvement_{metric}'] = finetuned_metrics[metric] - pretrained_metrics[metric]

                comparison_results.append(result)

            except Exception as e:
                print(f"Error comparing {clean_file}: {str(e)}")
                continue

        # Save results to CSV
        if comparison_results:
            df = pd.DataFrame(comparison_results)
            df.to_csv(results_file, index=False)
            print(f"Comparison results saved to {results_file}")

            # Print improvement summary
            print("\nImprovement Summary:")
            improvement_cols = [col for col in df.columns if col.startswith('improvement_')]
            print(df[improvement_cols].describe())

            return df
        else:
            print("No comparison results to save")
            return None


class ModelInference:
    """Handle model loading and inference"""

    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.model_type = "fine-tuned" if (model_path and os.path.exists(model_path)) else "pre-trained"
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            print(f"Loading fine-tuned model from: {model_path}")
            self.model = self._load_finetuned_model(model_path)
        else:
            # Load pre-trained model
            print("Loading pre-trained MetricGAN+ model...")
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": device}
            )

    def enhance_audio(self, noisy_audio):
        """Enhance a single audio file with proper preprocessing"""
        # Ensure we have the right tensor format
        if len(noisy_audio.shape) == 1:
            # Single channel, add batch and channel dimensions
            noisy_audio = noisy_audio.unsqueeze(0)  # [1, samples]
        elif len(noisy_audio.shape) == 2:
            # Check if it's [channels, samples] or [batch, samples]
            if noisy_audio.shape[0] == 2:  # Stereo audio
                # Convert stereo to mono
                noisy_audio = noisy_audio.mean(dim=0, keepdim=False)  # [samples]
                noisy_audio = noisy_audio.unsqueeze(0)  # [1, samples]
            # else: assume it's already [1, samples]

        # Process in chunks if too large (>4s at 16kHz)
        if noisy_audio.shape[-1] > 64000:
            chunk_size = 64000  # Process 4s chunks
            enhanced_chunks = []
            for i in range(0, noisy_audio.shape[-1], chunk_size):
                chunk = noisy_audio[..., i:i + chunk_size]
                with torch.no_grad():
                    enhanced_chunk = self.model.enhance_batch(
                        chunk.to(self.device),
                        lengths=torch.tensor([1.0])
                    )
                enhanced_chunks.append(enhanced_chunk.cpu())
                del chunk, enhanced_chunk  # Free memory
                torch.cuda.empty_cache() if self.device == 'cuda' else None

            enhanced = torch.cat(enhanced_chunks, dim=-1)
        else:
            with torch.no_grad():
                enhanced = self.model.enhance_batch(
                    noisy_audio.to(self.device),
                    lengths=torch.tensor([1.0])
                )

        return enhanced.squeeze()

    def process_single_file(self, clean_path, noisy_path, output_folder):
        """Process a single file pair with proper audio preprocessing"""
        print(f"Processing: {os.path.basename(clean_path)}")
        print_memory_usage()

        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(noisy_path):
            print(f"Warning: Noisy file not found: {noisy_path}")
            return False

        try:
            # Load audio file
            noisy_audio, sr = torchaudio.load(noisy_path, normalize=True)

            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                noisy_audio = resampler(noisy_audio)

            # Convert to mono if stereo
            if noisy_audio.shape[0] > 1:
                noisy_audio = noisy_audio.mean(dim=0, keepdim=True)  # [1, samples]

            # Ensure float32 type
            noisy_audio = noisy_audio.float()

            print(f"Audio shape after preprocessing: {noisy_audio.shape}")

            # Process with memory optimization
            enhanced_audio = self.enhance_audio(noisy_audio)

            # Save and clean up
            output_filename = os.path.basename(clean_path).replace('.wav', '_enhanced.wav')
            enhanced_path = os.path.join(output_folder, output_filename)

            # Ensure we have the right shape for saving
            if len(enhanced_audio.shape) == 1:
                enhanced_audio = enhanced_audio.unsqueeze(0)  # Add channel dimension

            torchaudio.save(enhanced_path, enhanced_audio, 16000)

            del noisy_audio, enhanced_audio
            torch.cuda.empty_cache() if self.device == 'cuda' else None

            print(f"Saved: {enhanced_path}")
            return True

        except Exception as e:
            print(f"Error processing {os.path.basename(clean_path)}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will help debug further issues
            return False

    def _load_finetuned_model(self, model_path):
        """Load fine-tuned model from .ckpt file with better error handling"""
        try:
            # Load pre-trained base model
            model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": self.device}
            )

            # Load fine-tuned weights from .ckpt file
            print(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint
                state_dict = checkpoint['state_dict']
                print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")

                # Remove 'model.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('model.', '')
                    new_state_dict[new_key] = v

                # Try to load the state dict
                try:
                    model.enhance_model.load_state_dict(new_state_dict, strict=False)
                    print("Successfully loaded fine-tuned weights")
                except Exception as e:
                    print(f"Error loading state dict: {e}")
                    print("Falling back to pretrained model")

            elif 'model' in checkpoint:
                # Direct model checkpoint
                model.enhance_model.load_state_dict(checkpoint['model'], strict=False)
                print("Successfully loaded fine-tuned weights from 'model' key")
            else:
                # Assume it's a direct state dict
                try:
                    model.enhance_model.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded fine-tuned weights (direct state dict)")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Falling back to pretrained model")

            return model

        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Falling back to pretrained model")
            # Return pretrained model if loading fails
            return SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": self.device}
            )

    def run_inference_on_test_set(self, test_clean_folder, test_noisy_folder, output_folder):
        print("=== Running inference on test set ===")
        os.makedirs(output_folder, exist_ok=True)

        clean_files = sorted([f for f in os.listdir(test_clean_folder) if f.endswith('.wav')])
        processed_files = []

        for i, clean_file in enumerate(clean_files):
            print(f"\nProcessing {i + 1}/{len(clean_files)}: {clean_file}")

            clean_path = os.path.join(test_clean_folder, clean_file)
            noisy_file = clean_file.replace('.wav', '_noisy.wav')
            noisy_path = os.path.join(test_noisy_folder, noisy_file)

            if self.process_single_file(clean_path, noisy_path, output_folder):
                processed_files.append(clean_file)

        print(f"\nProcessed {len(processed_files)} files")
        return processed_files


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def main():
    """Main execution function - PyCharm friendly version"""
    # ===== CONFIGURATION =====
    # Set  paths here directly
    test_clean_folder = "./Dataset_Improved/test/clean"  # Path to clean test files
    test_noisy_folder = "./Dataset_Improved/test/noisy"  # Path to noisy test files
    finetuned_model_path = "./pretrained_models/metricgan-plus-voicebank/enhance_model.ckpt"  # Path to fine-tuned model
    output_dir = "./enhanced_output"  # Output directory
    device = 'cpu'  # 'cpu' or 'cuda'

    # ===== EXECUTION =====
    # Create separate output folders
    pretrained_output = os.path.join(output_dir, 'pretrained')
    finetuned_output = os.path.join(output_dir, 'finetuned')

    # Initialize the evaluator
    evaluator = SpeechEnhancementEvaluator()

    # Run inference with pre-trained model
    print("\nRunning pre-trained model inference...")
    pretrained_inference = ModelInference(model_path=None, device=device)

    # Process files one by one to save memory
    clean_files = sorted([f for f in os.listdir(test_clean_folder) if f.endswith('.wav')])
    for clean_file in clean_files:
        clean_path = os.path.join(test_clean_folder, clean_file)
        noisy_path = os.path.join(test_noisy_folder, clean_file.replace('.wav', '_noisy.wav'))
        pretrained_inference.process_single_file(
            clean_path=clean_path,
            noisy_path=noisy_path,
            output_folder=pretrained_output
        )

    # Evaluate pre-trained model
    print("\nEvaluating pre-trained model...")
    pretrained_results = evaluator.evaluate_test_set(
        test_clean_folder=test_clean_folder,
        test_noisy_folder=test_noisy_folder,
        test_enhanced_folder=pretrained_output,
        output_file="pretrained_results.csv"
    )

    if finetuned_model_path and os.path.exists(finetuned_model_path):
        # Run inference with fine-tuned model
        print("\nRunning fine-tuned model inference...")
        finetuned_inference = ModelInference(model_path=finetuned_model_path, device=device)
        finetuned_processed = finetuned_inference.run_inference_on_test_set(
            test_clean_folder=test_clean_folder,
            test_noisy_folder=test_noisy_folder,
            output_folder=finetuned_output
        )

        # Evaluate fine-tuned model
        print("\nEvaluating fine-tuned model...")
        finetuned_results = evaluator.evaluate_test_set(
            test_clean_folder=test_clean_folder,
            test_noisy_folder=test_noisy_folder,
            test_enhanced_folder=finetuned_output,
            output_file="finetuned_results.csv"
        )

        # Compare models
        print("\nComparing models...")
        comparison_results = evaluator.compare_models(
            test_clean_folder=test_clean_folder,
            test_noisy_folder=test_noisy_folder,
            pretrained_output_folder=pretrained_output,
            finetuned_output_folder=finetuned_output,
            results_file="model_comparison.csv"
        )

        # Plot comparison results if available
        if comparison_results is not None:
            evaluator.plot_comparison_results(comparison_results)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
