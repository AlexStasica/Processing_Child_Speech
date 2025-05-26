from datasets import load_dataset, load_from_disk, Audio, DatasetDict, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, Trainer, TrainingArguments
import torch
import argparse
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np


def balance_dataset(dataset, target_ratio=0.5, seed=42, oversample=False):
    """
    Balance the dataset by undersampling the majority class and optionally oversampling the minority class.

    Args:
        dataset: The input dataset to balance.
        target_ratio: The desired ratio of the minority class to the majority class (default: 0.5).
        seed: Random seed for reproducibility.
        oversample: If True, oversample the minority class. If False, only undersample the majority class.

    Returns:
        A balanced dataset.
    """
    # Separate the dataset into two classes based on the label
    class_0 = dataset.filter(lambda example: example["label"] == 0)
    class_1 = dataset.filter(lambda example: example["label"] == 1)

    # Get the sizes of both classes
    class_0_size = len(class_0)
    class_1_size = len(class_1)

    # Determine which class is the majority and which is the minority
    if class_0_size > class_1_size:
        majority_class, minority_class = class_0, class_1
        majority_size, minority_size = class_0_size, class_1_size
    else:
        majority_class, minority_class = class_1, class_0
        majority_size, minority_size = class_1_size, class_0_size

    # Calculate the target size for the majority class based on the target ratio
    target_majority_size = int(minority_size / target_ratio)

    # Undersample the majority class to the target size
    if majority_size > target_majority_size:
        undersampled_majority = majority_class.shuffle(seed=seed).select(range(target_majority_size))
    else:
        undersampled_majority = majority_class

    # Optionally oversample the minority class
    if oversample:
        oversampled_minority = minority_class.shuffle(seed=seed).select(
            np.random.choice(minority_size, size=target_majority_size, replace=True)
        )
    else:
        oversampled_minority = minority_class

    # Combine the undersampled majority class and minority class
    balanced_dataset = concatenate_datasets([undersampled_majority, oversampled_minority])

    # Shuffle the balanced dataset
    balanced_dataset = balanced_dataset.shuffle(seed=seed)

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
        print(f"Audio batch shape: {[x.shape for x in audio_arrays]}")

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=None,
        )
        examples["label"] = torch.tensor(examples["label"], dtype=torch.int64)
        return inputs

    def extract_original_filename(example):
        example["original_filename"] = example["audio"]["path"].rsplit('_part', 1)[0]
        return example

    def split_dataset_by_original_filename(dataset, test_size=0.2, seed=42):
        # Split TD and DLD separately
        td_data = dataset.filter(lambda x: x["is_dld"] == 0)
        dld_data = dataset.filter(lambda x: x["is_dld"] == 1)

        # Split TD files into train and test
        td_filenames = list(set(td_data["original_filename"]))
        train_filenames, test_filenames = train_test_split(
            td_filenames, test_size=test_size, random_state=seed
        )

        td_train = td_data.filter(lambda x: x["original_filename"] in train_filenames)
        td_test = td_data.filter(lambda x: x["original_filename"] in test_filenames)

        # Combine TD test with all DLD for final test set
        test_data = concatenate_datasets([td_test, dld_data])

        return td_train, test_data

    split_dataset_path = "finetuning_data/split_dataset_age"

    # Initialize feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.pretrained_model_id)
    print(f"Feature extractor settings: {feature_extractor}")


    if os.path.exists(split_dataset_path):
        print("Loading saved dataset split age...")
        split = load_from_disk(split_dataset_path)
    else:
        print("Creating new dataset split age...")
        # Load dataset
        data = load_from_disk(args.data_id)

        # Resample audio to 16 kHz
        data = data.cast_column("audio", Audio(sampling_rate=16_000))
        data = data.map(extract_original_filename)

        # Define age groups and create mapping
        age_groups = ['3_1', '3_2', '4_1', '4_2', '5_1', '5_2']
        age_to_id = {age: idx for idx, age in enumerate(age_groups)}

        # Convert string labels to numerical indices
        def map_labels(example):
            example["label"] = age_to_id[example["label"]]
            return example

        data = data.map(map_labels)
        data = data.class_encode_column("label")

        # Preprocess function
        preprocess_fn = partial(preprocess_function, feature_extractor=feature_extractor)

        # Preprocess data in batches
        batch_size = 32
        encoded_data = data.map(
            preprocess_fn,
            remove_columns="audio",
            batched=True,
            batch_size=batch_size
        )

        # Split the dataset
        train_data, test_data = split_dataset_by_original_filename(encoded_data)

        # Create DatasetDict
        split = DatasetDict({
            'train': train_data,
            'test': test_data
        })

        # Save to disk
        split.save_to_disk(split_dataset_path)

    # Define label mappings
    age_groups = ['3_1', '3_2', '4_1', '4_2', '5_1', '5_2']
    id2label = {i: age for i, age in enumerate(age_groups)}

    # Print dataset statistics with readable labels
    print("\n=== Dataset Statistics ===")

    # Training set stats
    train_df = split['train'].to_pandas()
    train_df['age_group'] = train_df['label'].map(id2label)
    print("\nTraining Set (TD children only):")
    print(train_df['age_group'].value_counts().sort_index())

    # Test set stats
    test_df = split['test'].to_pandas()
    test_df['age_group'] = test_df['label'].map(id2label)

    print("\nTest Set - TD children (unseen during training):")
    print(test_df[test_df['is_dld'] == 0]['age_group'].value_counts().sort_index())

    print("\nTest Set - DLD children (chronological age):")
    print(test_df[test_df['is_dld'] == 1]['age_group'].value_counts().sort_index())

    # Calculate and print percentages
    def print_percentages(series, title):
        print(f"\n{title} (Percentage):")
        percentages = (series.value_counts(normalize=True) * 100).round(1)
        print(percentages.sort_index())

    print_percentages(train_df['age_group'], "Training Set Distribution")
    print_percentages(test_df[test_df['is_dld'] == 0]['age_group'], "Test Set TD Distribution")
    print_percentages(test_df[test_df['is_dld'] == 1]['age_group'], "Test Set DLD Distribution")
    # Age group labels
    age_groups = ['3_1', '3_2', '4_1', '4_2', '5_1', '5_2']
    id2label = {i: age for i, age in enumerate(age_groups)}
    label2id = {age: i for i, age in enumerate(age_groups)}

    # Load model
    model = WhisperForAudioClassification.from_pretrained(
        args.pretrained_model_id,
        num_labels=len(age_groups),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification"
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        fp16=True,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Track loss but can be changed to accuracy
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
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

    # Check if the model output directory exists, if not, create it
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    trainer.save_model(args.model_output_dir)
    # Save the feature extractor
    feature_extractor.save_pretrained(args.model_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="finetuning_data/age_classification_childes_whisper")
    parser.add_argument("--pretrained_model_id", type=str, default="openai/whisper-small")
    parser.add_argument("--model_output_dir", type=str, default="finetuned_model/age_classification_childes_whisper")
    args = parser.parse_args()
    main(args)
