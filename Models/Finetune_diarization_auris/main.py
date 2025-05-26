import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pyannote.audio import Pipeline
from textgrid import TextGrid, IntervalTier, Interval
import pickle
import os
import warnings
import glob
import traceback

warnings.filterwarnings("ignore")


# --- Step 1: Initialize Pyannote Pipeline ---
def initialize_pyannote():
    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR=HUGGINGFACE=TOKEN=HERE"
        )
    except Exception as e:
        print(f"Pyannote initialization failed: {str(e)}")
        print("Check your Hugging Face token and internet connection")
        sys.exit(1)


def load_slp_mapping(excel_path):
    """Load the SLP mapping from Excel file"""
    df = pd.read_excel(excel_path)
    return dict(zip(df['PP'], df['SLP_IDs']))


def find_audio_textgrid_pairs(root_dir, slp_mapping):
    """Find all valid audio-textgrid pairs with SLP IDs"""
    pairs = []
    age_groups = ['3yo', '4yo']  # , '5yo', '6yo'

    for age_dir in Path(root_dir).iterdir():
        if not age_dir.is_dir() or age_dir.name not in age_groups:
            continue

        for formatted_dir in age_dir.glob("**/Formatted/"):
            for audio_file in formatted_dir.rglob("*audio_chi_chins_adu.wav"):
                # Extract participant code (e.g., "P01" from "P01_M1_audio_chi_chins_adu.wav")
                participant_code = audio_file.stem.split('_')[0]

                # Get SLP ID from mapping
                slp_id = slp_mapping.get(participant_code)
                if slp_id is None:
                    print(f"Warning: No SLP ID found for participant {participant_code}")
                    continue

                base_name = audio_file.stem.replace('audio_', 'wav_textgrid_')
                textgrid_file = audio_file.parent / f"{base_name}.TextGrid"

                if textgrid_file.exists():
                    pairs.append((audio_file, textgrid_file, slp_id))

    print(f"Found {len(pairs)} audio-textgrid pairs")
    return pairs


def extract_speaker_segments(pairs, pipeline=None):
    """
    Extract speaker segments from TextGrids with SLP IDs
    Returns:
    - slp_embeddings: List of (embedding, slp_id) tuples
    - child_embeddings: List of embeddings for children
    - all_segments: All segments with (file, start, end, speaker_type, slp_id)
    """
    if pipeline is None:
        pipeline = initialize_pyannote()

    slp_embeddings = []
    child_embeddings = []
    all_segments = []

    for audio_file, textgrid_file, slp_id in pairs:
        try:
            tg = TextGrid.fromFile(str(textgrid_file))
            for tier in tg.tiers:
                tier_name = tier.name.lower()
                for interval in tier.intervals:
                    if not interval.mark.strip():
                        continue

                    start = interval.minTime
                    end = interval.maxTime
                    duration = end - start

                    # Skip segments that are too short (< 0.5s)
                    if duration < 0.5:
                        continue

                    if 'child' in tier_name:
                        embedding = extract_embedding(pipeline, audio_file, (start, end))
                        if embedding is not None:
                            child_embeddings.append(embedding)
                            all_segments.append((audio_file, start, end, 'CHI', None))
                            all_segments.append((audio_file, start, end, 'CHI_NS', None))
                    elif 'adult' in tier_name:
                        embedding = extract_embedding(pipeline, audio_file, (start, end))
                        if embedding is not None:
                            slp_embeddings.append((embedding, slp_id))
                            all_segments.append((audio_file, start, end, 'SLP', slp_id))
                            all_segments.append((audio_file, start, end, 'ADU', slp_id))

        except Exception as e:
            print(f"Error processing {textgrid_file}: {e}")

    return slp_embeddings, child_embeddings, all_segments


# --- Step 2: Extract Speaker Embeddings ---
def extract_embedding(pipeline, file, segment):
    try:
        waveform, sample_rate = torchaudio.load(file)
        start, end = segment
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # Check if end_sample is greater than the length of the waveform
        if end_sample > waveform.shape[1]:
            end_sample = waveform.shape[1]

        # Make sure start_sample is less than end_sample
        if start_sample >= end_sample:
            return None

        chunk = waveform[:, start_sample:end_sample]

        # Skip if chunk is too short
        if chunk.shape[1] < 0.2 * sample_rate:  # Less than 0.2 seconds
            return None

        # Pad short segments (<1s) to 1s
        if chunk.shape[1] < sample_rate:  # 1 second
            padding = sample_rate - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding), mode="constant")

        # Use Pyannote's embedding model
        with torch.no_grad():
            embedding = pipeline._embedding(chunk.unsqueeze(0))
        embedding = embedding.squeeze(0)

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {str(e)}")
        return None


def train_test_split_data(slp_embeddings, child_embeddings, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    # For SLP vs Child classifier
    X_speaker_type = np.vstack([np.array([e[0] for e in slp_embeddings]), np.array(child_embeddings)])
    y_speaker_type = np.array([1] * len(slp_embeddings) + [0] * len(child_embeddings))

    X_speaker_train, X_speaker_test, y_speaker_train, y_speaker_test = train_test_split(
        X_speaker_type, y_speaker_type, test_size=test_size, random_state=random_state, stratify=y_speaker_type
    )

    # For SLP identifier
    X_slp = np.array([e[0] for e in slp_embeddings])
    y_slp = np.array([e[1] for e in slp_embeddings])

    # Stratify by SLP ID
    try:
        X_slp_train, X_slp_test, y_slp_train, y_slp_test = train_test_split(
            X_slp, y_slp, test_size=test_size, random_state=random_state, stratify=y_slp
        )
    except ValueError:
        # If we can't stratify (e.g., some classes have only one instance), just split regularly
        X_slp_train, X_slp_test, y_slp_train, y_slp_test = train_test_split(
            X_slp, y_slp, test_size=test_size, random_state=random_state
        )

    return {
        'speaker_type': (X_speaker_train, X_speaker_test, y_speaker_train, y_speaker_test),
        'slp': (X_slp_train, X_slp_test, y_slp_train, y_slp_test)
    }


def train_models(train_data):
    """Train SLP vs Child classifier and SLP identifier"""
    X_speaker_train, _, y_speaker_train, _ = train_data['speaker_type']
    X_slp_train, _, y_slp_train, _ = train_data['slp']

    # SLP vs Child classifier
    speaker_clf = SVC(kernel='rbf', probability=True)
    speaker_clf.fit(X_speaker_train, y_speaker_train)

    # SLP identifier
    slp_clf = SVC(kernel='rbf', probability=True)
    slp_clf.fit(X_slp_train, y_slp_train)

    return speaker_clf, slp_clf


# --- Main Workflow for Training ---
def train_workflow(root_dir, excel_path, test_size=0.2, random_state=42):
    # Load SLP mapping
    slp_mapping = load_slp_mapping(excel_path)

    # Find all audio-textgrid pairs with SLP IDs
    pairs = find_audio_textgrid_pairs(root_dir, slp_mapping)

    # Initialize Pyannote pipeline
    pipeline = initialize_pyannote()

    # Extract speaker segments and embeddings
    print("Extracting speaker segments and embeddings...")
    slp_embeds, child_embeds, all_segments = extract_speaker_segments(pairs, pipeline)

    print(f"Extracted {len(slp_embeds)} SLP segments")
    print(f"Extracted {len(child_embeds)} child segments")
    print(f"Unique SLP IDs found: {set(e[1] for e in slp_embeds)}")

    # Split data into train and test sets
    print("Splitting data into train and test sets...")
    train_data = train_test_split_data(slp_embeds, child_embeds, test_size, random_state)

    # Train models
    print("Training models...")
    models = train_models(train_data)

    # Evaluate models
    print("Evaluating models...")
    evaluate_models(models, train_data)

    # Save models
    save_models(models)

    return models, all_segments


def evaluate_models(models, train_data):
    """Evaluate models on test data"""
    speaker_clf, slp_clf = models
    _, X_speaker_test, _, y_speaker_test = train_data['speaker_type']
    _, X_slp_test, _, y_slp_test = train_data['slp']

    # Evaluate SLP vs Child classifier
    y_speaker_pred = speaker_clf.predict(X_speaker_test)
    speaker_report = classification_report(y_speaker_test, y_speaker_pred, target_names=['Child', 'SLP'])
    speaker_conf_matrix = confusion_matrix(y_speaker_test, y_speaker_pred)

    # Evaluate SLP identifier
    y_slp_pred = slp_clf.predict(X_slp_test)
    slp_report = classification_report(y_slp_test, y_slp_pred)
    slp_conf_matrix = confusion_matrix(y_slp_test, y_slp_pred)

    print("\n=== SLP vs Child Classifier ===")
    print(speaker_report)
    print("Confusion Matrix:")
    print(speaker_conf_matrix)

    print("\n=== SLP Identifier ===")
    print(slp_report)
    print("Confusion Matrix:")
    print(slp_conf_matrix)

    # Return metrics for further analysis
    return {
        'speaker_type': {
            'report': speaker_report,
            'conf_matrix': speaker_conf_matrix
        },
        'slp': {
            'report': slp_report,
            'conf_matrix': slp_conf_matrix
        }
    }


def save_models(models, output_dir="models"):
    """Save trained models to disk"""
    speaker_clf, slp_clf = models

    os.makedirs(output_dir, exist_ok=True)

    # Save models
    with open(os.path.join(output_dir, "speaker_classifier.pkl"), "wb") as f:
        pickle.dump(speaker_clf, f)

    with open(os.path.join(output_dir, "slp_identifier.pkl"), "wb") as f:
        pickle.dump(slp_clf, f)

    print(f"Models saved to {output_dir}")


def load_models(model_dir="models"):
    """Load trained models from disk"""
    speaker_clf_path = os.path.join(model_dir, "speaker_classifier.pkl")
    slp_clf_path = os.path.join(model_dir, "slp_identifier.pkl")

    if not os.path.exists(speaker_clf_path) or not os.path.exists(slp_clf_path):
        print("Model files not found. Train models first.")
        return None, None

    with open(speaker_clf_path, "rb") as f:
        speaker_clf = pickle.load(f)

    with open(slp_clf_path, "rb") as f:
        slp_clf = pickle.load(f)

    return speaker_clf, slp_clf


def perform_diarization_inference(audio_file, models, pipeline=None):
    """
    Perform diarization inference on a new audio file
    Returns segments with predicted speaker information
    """
    if pipeline is None:
        pipeline = initialize_pyannote()

    speaker_clf, slp_clf = models

    # Get base diarization from pyannote
    diarization = pipeline(audio_file)
    print(f"Processing {audio_file}...")


    # Process each segment
    result_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = segment.start
        end = segment.end

        # Extract embedding for this segment
        embedding = extract_embedding(pipeline, audio_file, (start, end))
        if embedding is None:
            continue

        # Predict if SLP or child
        is_slp = speaker_clf.predict([embedding])[0]

        if is_slp == 1:  # SLP
            # Predict which SLP
            slp_id = slp_clf.predict([embedding])[0]
            result_segments.append((start, end, "SLP", slp_id))
        else:  # Child
            result_segments.append((start, end, "CHI", None))

    return result_segments


def save_to_textgrid(segments, output_file):
    """Save diarization results to TextGrid format"""

    # Create a new TextGrid
    tg = TextGrid(name=Path(output_file).stem)

    # Find the max end time to set TextGrid maxTime
    max_time = max([end for start, end, _, _ in segments]) if segments else 0
    if max_time == 0:
        max_time = 1.0  # Default if no segments

    # Create tiers for child and SLP
    child_tier = IntervalTier(name="CHI", minTime=0, maxTime=max_time)
    slp_tier = IntervalTier(name="ADU", minTime=0, maxTime=max_time)

    # Sort segments by start time to process them in order
    sorted_segments = sorted(segments, key=lambda x: x[0])

    # Process child segments (handle overlaps by merging or prioritizing)
    child_segments = [seg for seg in sorted_segments if seg[2] == "CHI"]
    process_segments_for_tier(child_segments, child_tier, "Speaker_01")

    # Process SLP segments
    slp_segments = [seg for seg in sorted_segments if seg[2] == "SLP"]
    process_segments_for_tier(slp_segments, slp_tier, "Speaker_02")

    # Add tiers to TextGrid
    tg.append(child_tier)
    tg.append(slp_tier)

    # Save to file
    try:
        tg.write(output_file)
        print(f"TextGrid saved to {output_file}")
    except Exception as e:
        print(f"Error saving TextGrid: {str(e)}")


def process_segments_for_tier(segments, tier, label_prefix):
    """
    Process segments for a tier, handling overlaps by merging overlapping segments
    """
    if not segments:
        return

    # Initialize with first segment
    current_start = segments[0][0]
    current_end = segments[0][1]
    current_id = segments[0][3]  # This will be None for CHI

    for i in range(1, len(segments)):
        start, end, _, seg_id = segments[i]

        # If current segment overlaps with previous
        if start <= current_end:
            # Merge by extending end time
            current_end = max(current_end, end)
            # For SLP segments with IDs, keep the ID of the longer segment
            if label_prefix == "Speaker_02" and seg_id is not None:
                if end - start > current_end - current_start:
                    current_id = seg_id
        else:
            # No overlap, add the previous segment and start a new one
            label = f"{label_prefix}" if label_prefix == "Speaker_02" and current_id is not None else label_prefix # _{current_id}
            tier.addInterval(Interval(current_start, current_end, label))

            # Start new segment
            current_start = start
            current_end = end
            current_id = seg_id

    # Add the last segment
    label = f"{label_prefix}" if label_prefix == "Speaker_01" and current_id is not None else label_prefix  #f"{label_prefix}_{current_id}"
    tier.addInterval(Interval(current_start, current_end, label))


# --- Main Workflow for Inference ---
def inference_workflow(audio_file, output_dir="output"):
    """Run inference on a new audio file and save results to TextGrid"""
    try:
        # Load trained models
        models = load_models()
        if models[0] is None or models[1] is None:
            print("Error: Could not load models. Make sure to train models first.")
            return []

        # Initialize Pyannote pipeline
        pipeline = initialize_pyannote()

        # Perform diarization
        segments = perform_diarization_inference(audio_file, models, pipeline)

        if not segments:
            print("Warning: No segments were identified in the audio file.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save results to TextGrid
        output_file = os.path.join(output_dir, f"{Path(audio_file).stem}_diarization.TextGrid")
        save_to_textgrid(segments, output_file)

        return segments
    except Exception as e:
        print(f"Error in inference workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # ============= CONFIGURATION SECTION =============

    # Set this to True to train models, False to run inference
    TRAIN_MODE = False

    # Paths for training
    ROOT_DIR = "C:/Users/a.stasica.AURIS.000/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener"  # Root directory audio and TextGrid files
    EXCEL_PATH = "DIARIZATION.xlsx"  # Path to Excel file with SLP mapping
    TEST_SIZE = 0.2  # Proportion of data to use for testing

    # Paths for inference
    INPUT_DIR = "C:/Users/a.stasica.AURIS.000/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener/TD_spontanee/Audio"  # Path to new audio file for inference
    OUTPUT_DIR = "C:/Users/a.stasica.AURIS.000/OneDrive - Stichting Onderwijs Koninklijke Auris Groep - 01JO/Desktop/Python/Screener/TD_spontanee/TextGrid"  # Directory to save TextGrid outputs
    AUDIO_EXTENSIONS = ['.wav']

    # ============= END OF CONFIGURATION SECTION =============

    # Run the appropriate workflow based on mode
    if TRAIN_MODE:
        print(f"Starting training with data from {ROOT_DIR}")
        print(f"Using SLP mapping from {EXCEL_PATH}")
        models, segments = train_workflow(ROOT_DIR, EXCEL_PATH, TEST_SIZE)
        print("Training complete!")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Get all audio files in the input directory
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))

        if not audio_files:
            print(f"No audio files found in {INPUT_DIR}")
            sys.exit(1)

        print(f"Found {len(audio_files)} audio files for processing")

        # Initialize Pyannote and load models once for all files
        models = load_models()
        if models[0] is None or models[1] is None:
            print("Error: Could not load models. Make sure to train models first.")
            sys.exit(1)

        pipeline = initialize_pyannote()

        # Process each audio file
        for i, audio_file in enumerate(audio_files):
            print(f"[{i + 1}/{len(audio_files)}] Processing {os.path.basename(audio_file)}")
            try:
                # Perform diarization
                segments = perform_diarization_inference(audio_file, models, pipeline)

                if not segments:
                    print(f" Warning: No segments were identified in {os.path.basename(audio_file)}")
                    continue

                # Save results to TextGrid
                output_file = os.path.join(OUTPUT_DIR, f"{Path(audio_file).stem}_diarized.TextGrid")
                save_to_textgrid(segments, output_file)
                print(f"  Completed: TextGrid saved to {os.path.basename(output_file)}")

            except Exception as e:
                print(f"  Error processing {os.path.basename(audio_file)}: {str(e)}")
                traceback.print_exc()

        print(f"Batch processing complete! Results saved to {OUTPUT_DIR}")
