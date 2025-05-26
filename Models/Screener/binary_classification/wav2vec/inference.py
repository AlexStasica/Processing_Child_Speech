from datasets import load_dataset, load_from_disk, Audio, Dataset, concatenate_datasets
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import argparse
from huggingface_hub import login
from sklearn.metrics import classification_report
from finetune_wav2vec2 import balance_dataset

def main(args):

    # Load dataset, change sampling rate to 16000 Hz
    dataset = load_from_disk(args.data_id)

     # Encode labels
    dataset = dataset.class_encode_column("label")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split the dataset using stratified sampling (train_test_split on labels)
    split = dataset.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

    split['train'] = balance_dataset(split['train'])
    # split['test'] = balance_dataset(split['test'])

    # Verify the distribution in the splits
    print(f"Train set distribution: {split['train']['label'].count(0)}")
    print(f"Train set distribution: {split['train']['label'].count(1)}")
    print(f"Test set distribution: {split['test']['label'].count(0)}")
    print(f"Test set distribution: {split['test']['label'].count(1)}")

    preds, labels = [], []
    # Iterate over audio files
    for i, item in enumerate(split['test']):
        
        test_audio = item["audio"]["array"]
        label = item["label"]
        filename = item["audio"]["path"]

        # Preprocess inputs using the feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        inputs = feature_extractor(test_audio, sampling_rate=16000, return_tensors='pt')

        # Load model
        model = AutoModelForAudioClassification.from_pretrained(args.finetuned_model_id)

        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get predicted label
        predicted_class_ids = torch.argmax(logits).item()
        predicted_label = model.config.id2label[predicted_class_ids]
        true_label = model.config.id2label[label]
        print(f'File name: {filename}')
        print(f'Predicted label for item {i}: {predicted_label}')
        print(f'True label for item {i}: {true_label}')
        print()

        preds.append(predicted_label)
        labels.append(true_label)

    print(classification_report(labels, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="finetuning_data/4yo_childes_auris_denoised")
    parser.add_argument("--finetuned_model_id", type=str, default="finetuned_model/4yo_childes_auris_denoised")
    args = parser.parse_args()
    main(args)