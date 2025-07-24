import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import warnings
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pyannote.audio import Model
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
from textgrid import TextGrid
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    fold: int
    der: float
    speaker_mapping_accuracy: float
    confusion_matrix: np.ndarray
    detailed_metrics: Dict
    files_processed: int


def custom_collate_fn(batch):
    """Custom collate function to handle mixed data types"""
    audio_tensors = []
    speaker_types = []
    speaker_ids = []
    slp_ids = []

    for sample in batch:
        audio_tensors.append(sample['audio'])
        speaker_types.append(sample['speaker_type'])
        speaker_ids.append(sample['speaker_id'])
        slp_ids.append(sample['slp_id'])

    # Stack audio tensors
    audio_batch = torch.stack(audio_tensors)

    return {
        'audio': audio_batch,
        'speaker_type': speaker_types,  # Keep as list of strings
        'speaker_id': speaker_ids,  # Keep as list of strings
        'slp_id': slp_ids  # Keep as list of strings
    }


class PyAnnoteClinicFinetuner:
    """Main class for PyAnnote clinical fine-tuning"""

    def __init__(self, config: Dict):
        self.config = config
        self.pipeline = None
        self.results = []
        self.output_dir = Path(config.get('output_dir', 'pyannote_results'))
        self.output_dir.mkdir(exist_ok=True)

    def initialize_pyannote(self):
        """Initialize PyAnnote pipeline"""
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.config.get('model_name', "pyannote/speaker-diarization-3.1"),
                use_auth_token=self.config.get('hf_token')
            )
            print("PyAnnote pipeline initialized successfully")
            return self.pipeline
        except Exception as e:
            print(f"Error initializing PyAnnote: {e}")
            raise

    def load_slp_mapping(self, excel_path: str) -> Dict[str, str]:
        """Load the SLP mapping from Excel file"""
        try:
            df = pd.read_excel(excel_path)
            mapping = dict(zip(df['PP'], df['SLP_IDs']))
            print(f"Loaded SLP mapping for {len(mapping)} participants")
            return mapping
        except Exception as e:
            print(f"Error loading SLP mapping: {e}")
            raise

    def find_audio_textgrid_pairs(self, root_dir: str, slp_mapping: Dict[str, str]) -> List[Tuple]:
        """Find all valid audio-textgrid pairs with SLP IDs"""
        pairs = []
        age_groups = ['3yo', '4yo', '5yo', '6yo']

        stats = {
            'total_files': 0,
            'by_age_group': {},
            'by_slp': defaultdict(int),
            'missing_slp_mapping': [],
            'missing_textgrid': [],
            'found_pairs': []
        }

        print("=" * 60)
        print("SEARCHING FOR AUDIO-TEXTGRID PAIRS")
        print("=" * 60)

        for age_dir in Path(root_dir).iterdir():
            if not age_dir.is_dir() or age_dir.name not in age_groups:
                continue

            age_group = age_dir.name
            stats['by_age_group'][age_group] = {'audio_files': [], 'valid_pairs': 0}

            print(f"\nProcessing {age_group} folder...")

            for formatted_dir in age_dir.glob("**/Formatted/"):
                for audio_file in formatted_dir.rglob("*audio_chi_chins_adu.wav"):
                    stats['total_files'] += 1
                    participant_code = audio_file.stem.split('_')[0]

                    # Check SLP mapping
                    slp_id = slp_mapping.get(participant_code)
                    if slp_id is None:
                        stats['missing_slp_mapping'].append(participant_code)
                        continue

                    # Check for corresponding TextGrid
                    base_name = audio_file.stem.replace('audio_', 'wav_textgrid_')
                    textgrid_file = audio_file.parent / f"{base_name}.TextGrid"

                    if not textgrid_file.exists():
                        stats['missing_textgrid'].append(str(audio_file.relative_to(root_dir)))
                        continue

                    # Valid pair found
                    pairs.append((audio_file, textgrid_file, slp_id))
                    stats['by_age_group'][age_group]['valid_pairs'] += 1
                    stats['by_slp'][slp_id] += 1
                    stats['found_pairs'].append({
                        'audio': audio_file.name,
                        'textgrid': textgrid_file.name,
                        'participant': participant_code,
                        'slp_id': slp_id,
                        'age_group': age_group
                    })

        self._print_data_statistics(stats)
        return pairs

    def _print_data_statistics(self, stats: Dict):
        """Print detailed statistics about the data"""
        print(f"\nDATA STATISTICS")
        print(f"Total audio files found: {stats['total_files']}")
        print(f"Valid pairs created: {len(stats['found_pairs'])}")
        print(f"Missing SLP mappings: {len(stats['missing_slp_mapping'])}")
        print(f"Missing TextGrids: {len(stats['missing_textgrid'])}")

        print(f"\nSLP Distribution:")
        for slp_id, count in sorted(stats['by_slp'].items()):
            print(f"  SLP {slp_id}: {count} recordings")

        print(f"\nAge Group Distribution:")
        for age_group, data in stats['by_age_group'].items():
            print(f"  {age_group}: {data['valid_pairs']} valid pairs")

    def textgrid_to_rttm(self, textgrid_path: str, output_path: str, file_id: str) -> bool:
        """Convert TextGrid file to RTTM format"""
        try:
            tg = TextGrid.fromFile(str(textgrid_path))

            with open(output_path, 'w') as f:
                for tier in tg.tiers:
                    tier_name = tier.name.lower()

                    # Determine speaker label
                    if 'child' in tier_name or 'chi' in tier_name:
                        speaker = 'CHI'
                    elif 'adult' in tier_name or 'adu' in tier_name:
                        speaker = 'SLP'
                    else:
                        continue

                    for interval in tier.intervals:
                        if not interval.mark.strip():
                            continue

                        start = interval.minTime
                        duration = interval.maxTime - interval.minTime

                        # Skip very short segments
                        if duration < 0.1:
                            continue

                        # RTTM format: SPEAKER <file_id> 1 <start> <duration> <na> <na> <speaker> <na> <na>
                        f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

            return True

        except Exception as e:
            print(f"Error converting TextGrid to RTTM: {e}")
            return False

    def create_cross_validation_folds(self, pairs: List[Tuple], n_folds: int = 5) -> List[Tuple]:
        """Create cross-validation folds based on SLP IDs"""
        # Group pairs by SLP ID
        slp_groups = defaultdict(list)
        for pair in pairs:
            slp_id = pair[2]
            slp_groups[slp_id].append(pair)

        # Get unique SLP IDs
        slp_ids = list(slp_groups.keys())

        print(f"\nCreating {n_folds}-fold cross-validation")
        print(f"Total SLPs: {len(slp_ids)}")
        print(f"SLP distribution: {[len(slp_groups[slp_id]) for slp_id in slp_ids]}")

        # Create folds using KFold on SLP IDs
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []

        for fold_idx, (train_slp_indices, test_slp_indices) in enumerate(kfold.split(slp_ids)):
            train_slps = [slp_ids[i] for i in train_slp_indices]
            test_slps = [slp_ids[i] for i in test_slp_indices]

            # Collect pairs for train and test sets
            train_pairs = []
            test_pairs = []

            for slp_id in train_slps:
                train_pairs.extend(slp_groups[slp_id])

            for slp_id in test_slps:
                test_pairs.extend(slp_groups[slp_id])

            folds.append((train_pairs, test_pairs, train_slps, test_slps))

            print(f"Fold {fold_idx + 1}:")
            print(f"  Train SLPs: {train_slps} ({len(train_pairs)} recordings)")
            print(f"  Test SLPs: {test_slps} ({len(test_pairs)} recordings)")

        return folds

    def prepare_data_for_fold(self, pairs: List[Tuple], fold_idx: int, is_train: bool = True) -> str:
        """Prepare data (convert to RTTM) for a specific fold"""
        fold_type = "train" if is_train else "test"
        fold_dir = self.output_dir / f"fold_{fold_idx}" / fold_type
        fold_dir.mkdir(parents=True, exist_ok=True)

        rttm_dir = fold_dir / "rttm"
        rttm_dir.mkdir(exist_ok=True)

        # Convert TextGrids to RTTM
        for audio_file, textgrid_file, slp_id in pairs:
            file_id = audio_file.stem
            rttm_file = rttm_dir / f"{file_id}.rttm"

            if not self.textgrid_to_rttm(textgrid_file, rttm_file, file_id):
                print(f" Failed to convert {textgrid_file} to RTTM")
                continue

        # Create database file listing all files
        db_file = fold_dir / "database.txt"
        with open(db_file, 'w') as f:
            for audio_file, textgrid_file, slp_id in pairs:
                file_id = audio_file.stem
                f.write(f"{file_id}\t{audio_file}\t{rttm_dir / f'{file_id}.rttm'}\n")

        print(f"Prepared {len(pairs)} files for fold {fold_idx} ({fold_type})")
        return str(fold_dir)

    def load_reference_from_textgrid(self, textgrid_path: str) -> Optional[Annotation]:
        """Load reference diarization from TextGrid file"""
        try:
            tg = TextGrid.fromFile(str(textgrid_path))
            annotation = Annotation()

            for tier in tg.tiers:
                tier_name = tier.name.lower()
                for interval in tier.intervals:
                    if not interval.mark.strip():
                        continue

                    start = interval.minTime
                    end = interval.maxTime

                    # Map tier names to speaker labels
                    if 'child' in tier_name or 'chi' in tier_name:
                        speaker = 'CHI'
                    elif 'adult' in tier_name or 'adu' in tier_name:
                        speaker = 'SLP'
                    else:
                        continue

                    segment = Segment(start, end)
                    annotation[segment] = speaker

            return annotation
        except Exception as e:
            print(f"Error loading reference from {textgrid_path}: {e}")
            return None

    def _evaluate_slp_recognition(self, reference: Annotation, hypothesis: Annotation, true_slp_id: str) -> float:
        """Evaluate how well the model recognizes the specific SLP"""
        try:
            # Get SLP segments from reference
            slp_segments = None
            for label in reference.labels():
                if label == 'SLP':
                    slp_segments = reference.label_timeline(label)
                    break

            if slp_segments is None:
                return 0.0

            # Find the hypothesis speaker that overlaps most with SLP segments
            best_overlap = 0.0
            best_hyp_speaker = None

            for hyp_speaker in hypothesis.labels():
                hyp_segments = hypothesis.label_timeline(hyp_speaker)
                overlap = slp_segments.intersection(hyp_segments).duration()

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_hyp_speaker = hyp_speaker

            # Return overlap ratio as recognition accuracy
            total_slp_duration = slp_segments.duration()
            return best_overlap / total_slp_duration if total_slp_duration > 0 else 0.0

        except Exception as e:
            print(f"Error evaluating SLP recognition: {e}")
            return 0.0

    def create_clinical_pipeline(self, fine_tuned_model: Model) -> Pipeline:
        """Create a clinical-aware diarization pipeline using the fine-tuned model"""

        # Create a custom pipeline configuration
        pipeline_config = {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 10,
                "threshold": 0.7,
            },
            "segmentation": {
                "min_duration_on": 0.5,
                "min_duration_off": 0.5,
            }
        }

        # Load base pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.config.get('hf_token')
        )

        # Replace the embedding model with our fine-tuned version
        pipeline.embedding = fine_tuned_model

        # Configure for 2-speaker clinical setting
        pipeline.instantiate({
            "clustering": {
                "min_cluster_size": 10,
                "max_num_speakers": 2,  # Force 2 speakers
                "min_num_speakers": 2,
            }
        })

        return pipeline

    def run_diarization_on_fold_fixed(self, test_pairs: List[Tuple], train_pairs: List[Tuple], fold_idx: int) -> Dict:
        """Run diarization on test set with fine-tuned model"""

        print(f"Using {len(train_pairs)} training pairs for fold {fold_idx}")

        # Fine-tune the model if we have training pairs
        if train_pairs and len(train_pairs) > 0:
            print(f"Fine-tuning model with {len(train_pairs)} training pairs")
            fine_tuned_model = self.fine_tune_embedding_model(train_pairs, fold_idx)
            clinical_pipeline = self.create_clinical_pipeline(fine_tuned_model)
        else:
            print("No training pairs available, using pre-trained model")
            clinical_pipeline = self.pipeline

        # Run evaluation with the pipeline
        results = {
            'fold': fold_idx,
            'der_scores': [],
            'speaker_mappings': [],
            'slp_recognition_accuracy': [],
            'processed_files': 0,
            'failed_files': 0
        }

        print(f"Running diarization on fold {fold_idx} test set...")

        for audio_file, textgrid_file, slp_id in test_pairs:
            try:
                # Load reference annotation
                reference = self.load_reference_from_textgrid(textgrid_file)
                if reference is None:
                    results['failed_files'] += 1
                    continue

                # Run diarization
                diarization = clinical_pipeline(str(audio_file))

                # Compute DER
                der = DiarizationErrorRate()
                der_score = der(reference, diarization)
                results['der_scores'].append(der_score)

                # Evaluate speaker mapping
                speaker_mapping_acc = self._evaluate_speaker_mapping(reference, diarization)
                results['speaker_mappings'].append(speaker_mapping_acc)

                # Evaluate SLP recognition
                slp_recognition_acc = self._evaluate_slp_recognition(reference, diarization, slp_id)
                results['slp_recognition_accuracy'].append(slp_recognition_acc)

                results['processed_files'] += 1

                if results['processed_files'] % 5 == 0:
                    print(f"  Processed {results['processed_files']}/{len(test_pairs)} files...")

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results['failed_files'] += 1

        # Compute aggregate metrics
        if results['der_scores']:
            results['mean_der'] = np.mean(results['der_scores'])
            results['std_der'] = np.std(results['der_scores'])
            results['mean_speaker_mapping'] = np.mean(results['speaker_mappings'])
            results['std_speaker_mapping'] = np.std(results['speaker_mappings'])
            results['mean_slp_recognition'] = np.mean(results['slp_recognition_accuracy'])
            results['std_slp_recognition'] = np.std(results['slp_recognition_accuracy'])
        else:
            results['mean_der'] = float('inf')
            results['std_der'] = 0.0
            results['mean_speaker_mapping'] = 0.0
            results['std_speaker_mapping'] = 0.0
            results['mean_slp_recognition'] = 0.0
            results['std_slp_recognition'] = 0.0

        print(f"  Fold {fold_idx} results:")
        print(f"  Mean DER: {results['mean_der']:.4f}")
        print(f"  Mean SLP Recognition: {results['mean_slp_recognition']:.4f}")

        return results

    def fine_tune_embedding_model(self, train_pairs: List[Tuple], fold_idx: int) -> Model:
        """Fine-tune the PyAnnote embedding model for clinical speaker diarization"""

        print(f" Fine-tuning embedding model for fold {fold_idx}...")

        if not train_pairs:
            print("  No training pairs provided, returning pre-trained model")
            return Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=self.config.get('hf_token')
            )

        # Create dataset and check if it has samples
        dataset = ClinicalSpeakerDataset(train_pairs)

        if len(dataset) == 0:
            print("  No training samples created, returning pre-trained model")
            return Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=self.config.get('hf_token')
            )

        # Load pre-trained model
        model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=self.config.get('hf_token')
        )

        # model parameters to better understand the structure
        print(" Model parameters:")
        trainable_params = []
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Now selectively unfreeze parameters based on actual parameter names
        for name, param in model.named_parameters():
            # Common patterns for layers to fine-tune in embedding models:
            if any(pattern in name.lower() for pattern in ['fc', 'linear', 'head', 'classifier', 'final']):
                param.requires_grad = True
                trainable_params.append(name)
            # For ResNet-based models, you might want to unfreeze the last few layers
            elif any(pattern in name.lower() for pattern in ['layer4', 'layer3']):  # Last ResNet layers
                param.requires_grad = True
                trainable_params.append(name)

        # If no parameters were unfrozen with the above logic, unfreeze the last few layers
        if not trainable_params:
            print("  No parameters matched the unfreezing criteria, unfreezing last layers...")
            all_params = list(model.named_parameters())
            # Unfreeze last 20% of parameters
            num_to_unfreeze = max(1, len(all_params) // 5)
            for name, param in all_params[-num_to_unfreeze:]:
                param.requires_grad = True
                trainable_params.append(name)

        print(f" Trainable parameters ({len(trainable_params)}):")
        for param_name in trainable_params:
            print(f"  - {param_name}")

        # Create dataloader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.config.get('batch_size', 32), len(dataset)),
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            drop_last=False,
            collate_fn=custom_collate_fn  # Use custom collate function
        )

        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()

        # Loss function and optimizer
        criterion = ClinicalSpeakerLoss(margin=0.5, alpha=0.7)

        # Get only the parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if not trainable_params:
            raise ValueError("No trainable parameters found after parameter selection!")

        optimizer = optim.Adam(
            trainable_params,
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=1e-5
        )

        # Training loop
        num_epochs = self.config.get('num_epochs', 10)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                try:
                    # Move to device
                    audio = batch['audio'].to(device)
                    speaker_types = batch['speaker_type']
                    slp_ids = batch['slp_id']

                    # Forward pass
                    with torch.set_grad_enabled(True):
                        # Get embeddings
                        embeddings = model(audio.unsqueeze(1))  # Add channel dimension

                        # If embeddings are 3D, take mean over time
                        if len(embeddings.shape) == 3:
                            embeddings = embeddings.mean(dim=2)

                        # Compute loss
                        loss = criterion(embeddings, speaker_types, slp_ids)

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    if num_batches % 10 == 0:
                        print(f"  Epoch {epoch + 1}/{num_epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f" Error in batch: {e}")
                    continue

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f" Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save fine-tuned model
        model_path = self.output_dir / f"fold_{fold_idx}" / "fine_tuned_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)

        print(f" Fine-tuned model saved to {model_path}")

        return model

    def _evaluate_speaker_mapping(self, reference: Annotation, hypothesis: Annotation) -> float:
        """Evaluate speaker mapping accuracy"""
        try:
            # Get overlap between reference and hypothesis
            ref_labels = set(reference.labels())
            hyp_labels = set(hypothesis.labels())

            if len(ref_labels) == 0 or len(hyp_labels) == 0:
                return 0.0

            # For each hypothesis speaker, find the best matching reference speaker
            best_mapping = {}
            total_duration = 0
            correct_duration = 0

            for hyp_speaker in hyp_labels:
                hyp_segments = hypothesis.label_timeline(hyp_speaker)

                best_ref_speaker = None
                best_overlap = 0

                for ref_speaker in ref_labels:
                    ref_segments = reference.label_timeline(ref_speaker)
                    overlap = hyp_segments.intersection(ref_segments).duration()

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_ref_speaker = ref_speaker

                if best_ref_speaker:
                    best_mapping[hyp_speaker] = best_ref_speaker
                    total_duration += hyp_segments.duration()
                    correct_duration += best_overlap

            return correct_duration / total_duration if total_duration > 0 else 0.0

        except Exception as e:
            print(f" Error evaluating speaker mapping: {e}")
            return 0.0

    def run_cross_validation(self, pairs: List[Tuple], n_folds: int = 5) -> List[Dict]:
        """Run complete cross-validation experiment"""
        print(f"\n Starting {n_folds}-fold cross-validation...")

        # Create folds
        folds = self.create_cross_validation_folds(pairs, n_folds)

        results = []

        for fold_idx, (train_pairs, test_pairs, train_slps, test_slps) in enumerate(folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {fold_idx + 1}/{n_folds}")
            print(f"{'=' * 60}")

            # Prepare data for this fold
            train_dir = self.prepare_data_for_fold(train_pairs, fold_idx, is_train=True)
            test_dir = self.prepare_data_for_fold(test_pairs, fold_idx, is_train=False)

            # Store train_pairs for this fold (this was missing!)
            self.current_fold_train_pairs = train_pairs

            # Run evaluation on test set - now pass train_pairs explicitly
            fold_results = self.run_diarization_on_fold_fixed(test_pairs, train_pairs, fold_idx)
            fold_results['train_slps'] = train_slps
            fold_results['test_slps'] = test_slps

            results.append(fold_results)

            # Save fold results
            self._save_fold_results(fold_results, fold_idx)

            print(f" Fold {fold_idx + 1} completed:")
            print(f"  Mean DER: {fold_results['mean_der']:.4f} ± {fold_results['std_der']:.4f}")
            print(f"  Mean Speaker Mapping: {fold_results['mean_speaker_mapping']:.4f} ± {fold_results['std_speaker_mapping']:.4f}")
            print(f"  Processed: {fold_results['processed_files']}/{len(test_pairs)} files")

        # Compute overall results
        self._compute_overall_results(results)

        return results

    def _save_fold_results(self, results: Dict, fold_idx: int):
        """Save results for a specific fold"""
        fold_dir = self.output_dir / f"fold_{fold_idx}"

        # Save detailed results
        with open(fold_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary CSV
        summary_data = {
            'fold': fold_idx,
            'mean_der': results['mean_der'],
            'std_der': results['std_der'],
            'mean_speaker_mapping': results['mean_speaker_mapping'],
            'std_speaker_mapping': results['std_speaker_mapping'],
            'processed_files': results['processed_files'],
            'failed_files': results['failed_files'],
            'train_slps': ','.join(results['train_slps']),
            'test_slps': ','.join(results['test_slps'])
        }

        df = pd.DataFrame([summary_data])
        df.to_csv(fold_dir / "summary.csv", index=False)

    def _compute_overall_results(self, all_results: List[Dict]):
        """Compute and save overall cross-validation results"""
        print(f"\n{'=' * 60}")
        print("OVERALL CROSS-VALIDATION RESULTS")
        print(f"{'=' * 60}")

        # Aggregate metrics
        all_ders = []
        all_mappings = []

        for results in all_results:
            all_ders.extend(results['der_scores'])
            all_mappings.extend(results['speaker_mappings'])

        overall_results = {
            'n_folds': len(all_results),
            'total_files_processed': sum(r['processed_files'] for r in all_results),
            'total_files_failed': sum(r['failed_files'] for r in all_results),
            'overall_mean_der': np.mean(all_ders) if all_ders else float('inf'),
            'overall_std_der': np.std(all_ders) if all_ders else 0.0,
            'overall_mean_speaker_mapping': np.mean(all_mappings) if all_mappings else 0.0,
            'overall_std_speaker_mapping': np.std(all_mappings) if all_mappings else 0.0,
            'fold_results': all_results
        }

        # Print summary
        print(f" Overall DER: {overall_results['overall_mean_der']:.4f} ± {overall_results['overall_std_der']:.4f}")
        print(
            f" Overall Speaker Mapping: {overall_results['overall_mean_speaker_mapping']:.4f} ± {overall_results['overall_std_speaker_mapping']:.4f}")
        print(f" Files processed: {overall_results['total_files_processed']}")
        print(f" Files failed: {overall_results['total_files_failed']}")

        # Save overall results
        with open(self.output_dir / "overall_results.json", 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)

        # Create summary table
        summary_data = []
        for i, results in enumerate(all_results):
            summary_data.append({
                'fold': i + 1,
                'mean_der': results['mean_der'],
                'std_der': results['std_der'],
                'mean_speaker_mapping': results['mean_speaker_mapping'],
                'std_speaker_mapping': results['std_speaker_mapping'],
                'processed_files': results['processed_files'],
                'train_slps': ','.join(results['train_slps']),
                'test_slps': ','.join(results['test_slps'])
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "cross_validation_summary.csv", index=False)

        print(f"\n Results saved to {self.output_dir}")

        return overall_results


class ClinicalSpeakerDataset(Dataset):
    """Dataset for clinical speaker diarization fine-tuning"""

    def __init__(self, pairs: List[Tuple], sample_rate: int = 16000, chunk_duration: float = 3.0):
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)

        # Create training samples
        self.samples = self._create_training_samples()
        print(f" Created {len(self.samples)} training samples from {len(pairs)} pairs")

    def _create_training_samples(self):
        """Create training samples with speaker labels and SLP IDs"""
        samples = []

        for audio_file, textgrid_file, slp_id in self.pairs:
            try:
                print(f" Processing {audio_file.name}...")

                # Check if files exist
                if not audio_file.exists():
                    print(f"️  Audio file not found: {audio_file}")
                    continue

                if not textgrid_file.exists():
                    print(f"️  TextGrid file not found: {textgrid_file}")
                    continue

                # Load audio
                try:
                    waveform, sr = torchaudio.load(str(audio_file))
                    if sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

                    # Take first channel if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1, :]

                except Exception as e:
                    print(f" Error loading audio {audio_file}: {e}")
                    continue

                # Load TextGrid for annotations
                try:
                    from textgrid import TextGrid
                    tg = TextGrid.fromFile(str(textgrid_file))
                except Exception as e:
                    print(f" Error loading TextGrid {textgrid_file}: {e}")
                    continue

                # Extract segments for each speaker
                for tier in tg.tiers:
                    tier_name = tier.name.lower()

                    # Determine speaker type and label
                    if 'child' in tier_name or 'chi' in tier_name:
                        speaker_type = 'CHI'
                        speaker_id = 'CHILD'  # Generic child ID
                    elif 'adult' in tier_name or 'adu' in tier_name:
                        speaker_type = 'SLP'
                        speaker_id = slp_id  # Specific SLP ID
                    else:
                        continue

                    # Extract audio chunks from this tier
                    for interval in tier.intervals:
                        if not interval.mark or not interval.mark.strip():
                            continue

                        start_time = max(0, interval.minTime)
                        end_time = min(interval.maxTime, waveform.shape[1] / self.sample_rate)
                        duration = end_time - start_time

                        # Skip very short segments
                        if duration < 0.5:
                            continue

                        # Convert to sample indices
                        start_sample = int(start_time * self.sample_rate)
                        end_sample = int(end_time * self.sample_rate)

                        # Ensure indices are within bounds
                        start_sample = max(0, min(start_sample, waveform.shape[1] - 1))
                        end_sample = max(start_sample + 1, min(end_sample, waveform.shape[1]))

                        # Extract audio chunk
                        audio_chunk = waveform[:, start_sample:end_sample]

                        # Skip if chunk is too short
                        if audio_chunk.shape[1] < self.sample_rate * 0.5:  # Less than 0.5 seconds
                            continue

                        # Handle different chunk lengths
                        if audio_chunk.shape[1] < self.chunk_samples:
                            # Pad with zeros if shorter than target
                            padding = self.chunk_samples - audio_chunk.shape[1]
                            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, padding))

                            samples.append({
                                'audio': audio_chunk,
                                'speaker_type': speaker_type,
                                'speaker_id': speaker_id,
                                'slp_id': slp_id,
                                'file_id': audio_file.stem
                            })

                        elif audio_chunk.shape[1] > self.chunk_samples:
                            # Create multiple chunks if longer than target
                            num_chunks = max(1, audio_chunk.shape[1] // self.chunk_samples)
                            for i in range(num_chunks):
                                chunk_start = i * self.chunk_samples
                                chunk_end = min(chunk_start + self.chunk_samples, audio_chunk.shape[1])

                                chunk = audio_chunk[:, chunk_start:chunk_end]

                                # Pad if necessary
                                if chunk.shape[1] < self.chunk_samples:
                                    padding = self.chunk_samples - chunk.shape[1]
                                    chunk = torch.nn.functional.pad(chunk, (0, padding))

                                samples.append({
                                    'audio': chunk,
                                    'speaker_type': speaker_type,
                                    'speaker_id': speaker_id,
                                    'slp_id': slp_id,
                                    'file_id': audio_file.stem
                                })
                        else:
                            # Perfect length
                            samples.append({
                                'audio': audio_chunk,
                                'speaker_type': speaker_type,
                                'speaker_id': speaker_id,
                                'slp_id': slp_id,
                                'file_id': audio_file.stem
                            })

            except Exception as e:
                print(f" Error processing {audio_file}: {e}")
                continue

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio': sample['audio'].squeeze(0),  # Remove channel dimension
            'speaker_type': sample['speaker_type'],
            'speaker_id': sample['speaker_id'],
            'slp_id': sample['slp_id']
        }


class ClinicalSpeakerLoss(nn.Module):
    """Custom loss function for clinical speaker diarization"""

    def __init__(self, margin: float = 0.5, alpha: float = 0.7):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for SLP identification vs speaker type classification

    def forward(self, embeddings, speaker_types, slp_ids):
        """
        Compute contrastive loss for speaker diarization

        Args:
            embeddings: Speaker embeddings [batch_size, embedding_dim]
            speaker_types: Speaker type labels ('CHI' or 'SLP')
            slp_ids: SLP identification labels
        """
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t())

        # Create positive and negative masks
        positive_mask = torch.zeros_like(similarity_matrix, device=embeddings.device)
        negative_mask = torch.zeros_like(similarity_matrix, device=embeddings.device)

        # Fill masks
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    if speaker_types[i] == 'CHI' and speaker_types[j] == 'CHI':
                        positive_mask[i, j] = 1.0
                    elif speaker_types[i] == 'SLP' and speaker_types[j] == 'SLP' and slp_ids[i] == slp_ids[j]:
                        positive_mask[i, j] = 1.0
                    else:
                        negative_mask[i, j] = 1.0

        # Contrastive loss
        positive_loss = positive_mask * torch.clamp(self.margin - similarity_matrix, min=0.0)
        negative_loss = negative_mask * torch.clamp(similarity_matrix - self.margin, min=0.0)

        # Compute total loss
        total_positive = positive_mask.sum()
        total_negative = negative_mask.sum()

        if total_positive > 0 and total_negative > 0:
            contrastive_loss = (positive_loss.sum() + negative_loss.sum()) / (total_positive + total_negative)
        elif total_positive > 0:
            contrastive_loss = positive_loss.sum() / total_positive
        elif total_negative > 0:
            contrastive_loss = negative_loss.sum() / total_negative
        else:
            contrastive_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Additional loss for SLP identification
        slp_mask = torch.tensor([st == 'SLP' for st in speaker_types], device=embeddings.device)
        slp_embeddings = embeddings[slp_mask]
        slp_labels = [slp_ids[i] for i, mask in enumerate(slp_mask) if mask]

        slp_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        if len(slp_embeddings) > 1:
            slp_similarity = torch.mm(slp_embeddings, slp_embeddings.t())
            slp_positive_mask = torch.zeros_like(slp_similarity, device=embeddings.device)
            slp_negative_mask = torch.zeros_like(slp_similarity, device=embeddings.device)

            for i in range(len(slp_labels)):
                for j in range(len(slp_labels)):
                    if i != j:
                        if slp_labels[i] == slp_labels[j]:
                            slp_positive_mask[i, j] = 1.0
                        else:
                            slp_negative_mask[i, j] = 1.0

            slp_positive_loss = slp_positive_mask * torch.clamp(self.margin - slp_similarity, min=0.0)
            slp_negative_loss = slp_negative_mask * torch.clamp(slp_similarity - self.margin, min=0.0)

            slp_total_positive = slp_positive_mask.sum()
            slp_total_negative = slp_negative_mask.sum()

            if slp_total_positive > 0 and slp_total_negative > 0:
                slp_loss = (slp_positive_loss.sum() + slp_negative_loss.sum()) / (
                        slp_total_positive + slp_total_negative)
            elif slp_total_positive > 0:
                slp_loss = slp_positive_loss.sum() / slp_total_positive
            elif slp_total_negative > 0:
                slp_loss = slp_negative_loss.sum() / slp_total_negative

        # Combine both losses
        total_loss = (1 - self.alpha) * contrastive_loss + self.alpha * slp_loss

        return total_loss


def main():
    """Main function to run the complete pipeline"""

    # Configuration
    config = {
        'model_name': "pyannote/speaker-diarization-3.1",
        'embedding_model': "pyannote/wespeaker-voxceleb-resnet34-LM",
        'hf_token': "YOUR=HF=TOKEN=HERE",
        'output_dir': 'pyannote_clinical_results',
        'n_folds': 5,
        'data_root': "", # update path
        'slp_mapping_file': "DIARIZATION.xlsx",
        # Training parameters
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'chunk_duration': 3.0,
    }

    # Initialize fine-tuner
    finetuner = PyAnnoteClinicFinetuner(config)

    try:
        # Initialize PyAnnote
        finetuner.initialize_pyannote()

        # Load SLP mapping
        slp_mapping = finetuner.load_slp_mapping(config['slp_mapping_file'])

        # Find audio-textgrid pairs
        pairs = finetuner.find_audio_textgrid_pairs(config['data_root'], slp_mapping)

        if not pairs:
            print(" No valid audio-textgrid pairs found. Please check your data paths.")
            return

        # Run cross-validation
        results = finetuner.run_cross_validation(pairs, config['n_folds'])

        print(f"\n Cross-validation completed successfully!")
        print(f" Results saved to: {finetuner.output_dir}")

    except Exception as e:
        print(f" Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
