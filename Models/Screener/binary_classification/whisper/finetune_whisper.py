from datasets import load_dataset, load_from_disk, Audio, DatasetDict, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, Trainer, TrainingArguments
import torch
import argparse
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np


def balance_dataset_exact(dataset, class_0_target, class_1_target, seed=42):
    """
    Balance the dataset to exact target counts for each class.

    Args:
        dataset: Input dataset to balance
        class_0_target: Desired number of samples for class 0
        class_1_target: Desired number of samples for class 1
        seed: Random seed for reproducibility

    Returns:
        Balanced dataset with exact counts for each class
    """
    # Separate the dataset into two classes
    class_0 = dataset.filter(lambda example: example["label"] == 0)
    class_1 = dataset.filter(lambda example: example["label"] == 1)

    # Shuffle and select the target number of samples
    class_0_samples = class_0.shuffle(seed=seed).select(range(min(class_0_target, len(class_0))))
    class_1_samples = class_1.shuffle(seed=seed).select(range(min(class_1_target, len(class_1))))

    # Combine and shuffle
    balanced_dataset = concatenate_datasets([class_0_samples, class_1_samples]).shuffle(seed=seed)
    return balanced_dataset


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        batch = next(iter(dataloader))
        # print("First batch shape:", batch['input_features'].shape)
        return dataloader


def main(args):

    def preprocess_function(examples, feature_extractor):

        audio_arrays = [np.array(x["array"]) for x in examples["audio"]]
        print(f"Audio batch shape: {[x.shape for x in audio_arrays]}")  # Debug

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,  # Ensure 16 kHz
            return_tensors="pt",
            padding=True,  # Auto-padding
            truncation=True,  # Auto-truncation
            max_length=None,  # Let god decide
        )

        examples["label"] = torch.tensor(examples["label"], dtype=torch.int64)
        return inputs

    # Extract original filename (without _partX suffix)
    def extract_original_filename(example):
        example["original_filename"] = example["audio"]["path"].rsplit('_part', 1)[0]
        return example

    # Split the dataset ensuring chunks from the same original file are not split
    def split_dataset_by_original_filename(dataset, test_size=0.1, seed=42):
        # Get unique original filenames
        original_filenames = list(set(dataset["original_filename"]))

        # Split the original filenames into train and test sets
        train_filenames, test_filenames = train_test_split(original_filenames, test_size=test_size, random_state=seed)

        # Create train and test datasets based on the split filenames
        train_data = dataset.filter(lambda example: example["original_filename"] in train_filenames)
        test_data = dataset.filter(lambda example: example["original_filename"] in test_filenames)

        return train_data, test_data

    split_dataset_path = "finetuning_data/split_dataset_denoised"

    if os.path.exists(split_dataset_path):
        print("Loading saved dataset split...")
        split = load_from_disk(split_dataset_path)
    else:
        print("Creating new dataset split...")
        # Load dataset
        data = load_from_disk(args.data_id)

        # Resample audio to 16 kHz (required by Whisper)
        data = data.cast_column("audio", Audio(sampling_rate=16_000))

        data = data.map(extract_original_filename)

        # Encode labels
        data = data.class_encode_column("label")

        # Initialize feature extractor
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.pretrained_model_id)
        print(f"Feature extractor settings: {feature_extractor}")

        # Bind feature_extractor to preprocess_function
        preprocess_fn = partial(preprocess_function, feature_extractor=feature_extractor)

        # Preprocess data in batches
        batch_size = 32  # Adjust based on memory constraints
        encoded_data = data.map(preprocess_fn, remove_columns="audio", batched=True, batch_size=batch_size)

        # Ensure that the label column is of type int64
        encoded_data = encoded_data.map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)},
                                        batched=True)

        # Split the dataset
        train_data, test_data = split_dataset_by_original_filename(encoded_data, test_size=0.2, seed=42)

        # Define your exact desired counts
        TRAIN_DLD_COUNT = 800  # DLD class (class 0)
        TRAIN_TD_COUNT = 1000  # TD class (class 1)
        TEST_DLD_COUNT = 100  # DLD test samples
        TEST_TD_COUNT = 100  # TD test samples

        # Balance the training set with exact counts
        train_data = balance_dataset_exact(train_data,
                                           class_0_target=TRAIN_DLD_COUNT,
                                           class_1_target=TRAIN_TD_COUNT,
                                           seed=42)

        # Balance the test set with exact counts
        test_data = balance_dataset_exact(test_data,
                                          class_0_target=TEST_DLD_COUNT,
                                          class_1_target=TEST_TD_COUNT,
                                          seed=42)

        # Add verification prints
        def print_dataset_stats(name, dataset):
            class_0 = dataset.filter(lambda example: example["label"] == 0)
            class_1 = dataset.filter(lambda example: example["label"] == 1)
            print(f"\n{name} set statistics:")
            print(f"  Class 0 (DLD) samples: {len(class_0)}")
            print(f"  Class 1 (TD) samples: {len(class_1)}")
            print(f"  Total samples: {len(dataset)}")

        print_dataset_stats("Training", train_data)
        print_dataset_stats("Test", test_data)

        # Create a DatasetDict for use during inference
        split = DatasetDict({
            'train': train_data,
            'test': test_data
        })

        # Save the DatasetDict to disk
        split.save_to_disk(f"finetuning_data/split_dataset_denoised")

    # Verify the distribution in the splits
    print(f"Train set DLD distribution: {split['train']['label'].count(0)}")
    print(f"Train set TD distribution: {split['train']['label'].count(1)}")
    print(f"Test set DLD distribution: {split['test']['label'].count(0)}")
    print(f"Test set TD distribution: {split['test']['label'].count(1)}")

    # Ensure labels are cast to int64 in both train and test splits
    split['train'] = split['train'].map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)},
                                        batched=True)
    split['test'] = split['test'].map(lambda example: {"label": torch.tensor(example["label"], dtype=torch.int64)},
                                      batched=True)

    # Verify the label type (it should be int64/Long)
    print(f"Label type (train): {split['train'].features['label'].dtype}")
    print(f"Label type (test): {split['test'].features['label'].dtype}")

    # Load model
    id2label = {0: 'DLD', 1: 'TD'}
    label2id = {'DLD': 0, 'TD': 1}
    num_labels = len(id2label)
    model = WhisperForAudioClassification.from_pretrained(
        args.pretrained_model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=8, 
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
    )


    # Define trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
    )

    # Check if checkpoints exist and resume training
    latest_checkpoint = None
    if os.path.exists(args.model_output_dir):
        checkpoints = [d for d in os.listdir(args.model_output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = os.path.join(args.model_output_dir,
                                             sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")

    # Train model (resume if checkpoint exists)
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # Ensure model output directory exists
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    # Check if the model output directory exists, if not, create it
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    # Once ready, save to disk
    trainer.save_model(args.model_output_dir)
    # Save the feature extractor
    feature_extractor.save_pretrained(args.model_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="finetuning_data/4yo_childes_auris_alldenoised_whisper")
    parser.add_argument("--pretrained_model_id", type=str, default="openai/whisper-small")
    parser.add_argument("--model_output_dir", type=str, default="finetuned_model/4yo_alldenoised_whisper")
    args = parser.parse_args()
    main(args)
